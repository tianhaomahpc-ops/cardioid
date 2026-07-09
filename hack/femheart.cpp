#include "mfem.hpp"
#include "object.h"
#include "object_cc.hh"
#include "ddcMalloc.h"
#include "pio.h"
#include "pioFixedRecordHelper.h"
#include "units.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <cassert>
#include <memory>
#include <set>
#include <dirent.h>
#include <regex.h>
#include <unistd.h>
#include <sys/stat.h>
#include "util.hpp"
#include "MatrixElementPiecewiseCoefficient.hpp"
#include "cardiac_coefficients.hpp"
#include "torsoSolver.hpp"
#include "scaled_asm.hpp"

#include <map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <mpi.h> // Include MPI header
#include <iomanip> // For formatted output

#include <string> 

#define StartTimer(x)
#define EndTimer()

using namespace mfem;

MPI_Comm COMM_LOCAL = MPI_COMM_WORLD;


const double DEFAULT_HEART_POTENTIAL = -83.0;  // 默认心脏电位值（仅作为备用）

static void ConvertHypreToPetscAIJSafe(HypreParMatrix &hypre_mat,
                                       PetscParMatrix &petsc_mat,
                                       const char *name,
                                       int my_rank)
{
    MPI_Comm comm = hypre_mat.GetComm();
    const HYPRE_BigInt row_start_big = hypre_mat.GetRowStarts()[0];
    const HYPRE_BigInt row_end_big = hypre_mat.GetRowStarts()[1];
    const HYPRE_BigInt col_start_big = hypre_mat.GetColStarts()[0];
    const HYPRE_BigInt col_end_big = hypre_mat.GetColStarts()[1];

    const PetscInt local_rows =
        static_cast<PetscInt>(row_end_big - row_start_big);
    const PetscInt local_cols =
        static_cast<PetscInt>(col_end_big - col_start_big);
    const PetscInt global_rows =
        static_cast<PetscInt>(hypre_mat.GetGlobalNumRows());
    const PetscInt global_cols =
        static_cast<PetscInt>(hypre_mat.GetGlobalNumCols());
    const PetscInt col_start = static_cast<PetscInt>(col_start_big);
    const PetscInt col_end = static_cast<PetscInt>(col_end_big);

    SparseMatrix merged;
    hypre_mat.MergeDiagAndOffd(merged);
    MFEM_VERIFY(merged.Height() == local_rows,
                "Unexpected local row count in Hypre to PETSc conversion.");

    const int *I = merged.HostReadI();
    const int *J = merged.HostReadJ();
    const real_t *data = merged.HostReadData();

    std::vector<PetscInt> d_nnz(static_cast<size_t>(local_rows), 0);
    std::vector<PetscInt> o_nnz(static_cast<size_t>(local_rows), 0);
    for (PetscInt i = 0; i < local_rows; ++i)
    {
        for (int p = I[i]; p < I[i + 1]; ++p)
        {
            const PetscInt col = static_cast<PetscInt>(J[p]);
            if (col_start <= col && col < col_end)
            {
                ++d_nnz[static_cast<size_t>(i)];
            }
            else
            {
                ++o_nnz[static_cast<size_t>(i)];
            }
        }
    }

    Mat mat = NULL;
    PetscErrorCode ierr = MatCreateAIJ(
        comm, local_rows, local_cols, global_rows, global_cols,
        0, local_rows ? d_nnz.data() : NULL,
        0, local_rows ? o_nnz.data() : NULL, &mat);
    MFEM_VERIFY(ierr == 0, "MatCreateAIJ failed in safe Hypre conversion.");

    if (name)
    {
        ierr = PetscObjectSetName((PetscObject) mat, name);
        MFEM_VERIFY(ierr == 0, "PetscObjectSetName failed in safe Hypre conversion.");
    }

    for (PetscInt i = 0; i < local_rows; ++i)
    {
        const PetscInt row = static_cast<PetscInt>(row_start_big) + i;
        for (int p = I[i]; p < I[i + 1]; ++p)
        {
            const PetscInt col = static_cast<PetscInt>(J[p]);
            const PetscScalar value = static_cast<PetscScalar>(data[p]);
            ierr = MatSetValues(mat, 1, &row, 1, &col, &value, INSERT_VALUES);
            MFEM_VERIFY(ierr == 0,
                        "MatSetValues failed in safe Hypre conversion.");
        }
    }

    ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
    MFEM_VERIFY(ierr == 0, "MatAssemblyBegin failed in safe Hypre conversion.");
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
    MFEM_VERIFY(ierr == 0, "MatAssemblyEnd failed in safe Hypre conversion.");

    petsc_mat.SetMat(mat);
    ierr = MatDestroy(&mat);
    MFEM_VERIFY(ierr == 0, "MatDestroy failed after safe Hypre conversion.");

    if (my_rank == 0)
    {
        std::cout << "[PETSc] safe AIJ conversion for " << name
                  << ": global " << global_rows << "x" << global_cols
                  << ", local rows on rank 0 = " << local_rows << std::endl;
    }
}




// 调试版 heart -> torso 边界传输类。
// 设计目标：
// 1) 不做全局边界点副本；
// 2) 不使用 MPI_Alltoall / MPI_Alltoallv；
// 3) 每个 phase 都用 "root 只收元数据 + 稀疏点对点 payload" 的闭合协议；
// 4) 为 SparseExchangeViaRootMeta 提供充分的调试插桩，便于定位 1536/3072 ranks 下的通信不匹配。
//
// [FIX] 2024: 将 root 顺序 Send/Recv 分发元数据改为 MPI_Scatter/Scatterv，
//       消除大规模并行下的死锁风险和 O(P) 串行瓶颈。
class HeartTorsoBoundaryTransfer {
private:
    struct InterfacePointRecord {
        double coords[3];
        int home_rank;
        int local_id;
    };

    struct MatchProposal {
        int torso_home_rank;
        int torso_local_id;
        int heart_home_rank;
        int heart_local_id;
        double dist2;
    };

    struct ValueRequest {
        int torso_home_rank;
        int torso_local_id;
        int heart_local_id;
    };

    struct MetaEntry {
        int peer;
        int count;
    };

    struct QuantizedKey {
        long long ix;
        long long iy;
        long long iz;

        bool operator==(const QuantizedKey& other) const {
            return ix == other.ix && iy == other.iy && iz == other.iz;
        }
    };

    struct QuantizedKeyHash {
        std::size_t operator()(const QuantizedKey& key) const {
            std::size_t h1 = std::hash<long long>{}(key.ix);
            std::size_t h2 = std::hash<long long>{}(key.iy);
            std::size_t h3 = std::hash<long long>{}(key.iz);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    ParMesh* heart_mesh_;
    ParFiniteElementSpace* heart_fes_;
    ParGridFunction* heart_gf_;

    ParMesh* torso_mesh_;
    ParFiniteElementSpace* torso_fes_;
    ParGridFunction* torso_gf_;

    double spatial_tol_;
    int my_rank_;
    int num_ranks_;

    double global_xmin_;
    double global_xmax_;

    std::vector<int> local_torso_dirichlet_vdofs_;

    std::vector<int> send_peers_;
    std::vector<std::vector<int> > send_heart_local_ids_per_peer_;

    std::vector<int> recv_peers_;
    std::vector<std::vector<int> > recv_torso_local_ids_per_peer_;

    std::vector<int> self_heart_local_ids_;
    std::vector<int> self_torso_local_ids_;

    static QuantizedKey MakeQuantizedKey(const double* coords, double tol,
                                         int dx = 0, int dy = 0, int dz = 0)
    {
        QuantizedKey key;
        key.ix = static_cast<long long>(std::llround(coords[0] / tol)) + dx;
        key.iy = static_cast<long long>(std::llround(coords[1] / tol)) + dy;
        key.iz = static_cast<long long>(std::llround(coords[2] / tol)) + dz;
        return key;
    }

    static double DistanceSquared(const double* a, const double* b)
    {
        const double dx = a[0] - b[0];
        const double dy = a[1] - b[1];
        const double dz = a[2] - b[2];
        return dx * dx + dy * dy + dz * dz;
    }

    static MPI_Datatype BuildPointRecordMPIType()
    {
        MPI_Datatype mpi_type;
        const int nitems = 3;
        int blocklengths[3] = {3, 1, 1};
        MPI_Datatype types[3] = {MPI_DOUBLE, MPI_INT, MPI_INT};
        MPI_Aint offsets[3];

        InterfacePointRecord dummy;
        MPI_Aint base_address;
        MPI_Get_address(&dummy, &base_address);
        MPI_Get_address(&dummy.coords[0], &offsets[0]);
        MPI_Get_address(&dummy.home_rank, &offsets[1]);
        MPI_Get_address(&dummy.local_id, &offsets[2]);
        for (int i = 0; i < nitems; i++) { offsets[i] -= base_address; }

        MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_type);
        MPI_Type_commit(&mpi_type);
        return mpi_type;
    }

    static MPI_Datatype BuildMatchProposalMPIType()
    {
        MPI_Datatype mpi_type;
        const int nitems = 5;
        int blocklengths[5] = {1, 1, 1, 1, 1};
        MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE};
        MPI_Aint offsets[5];

        MatchProposal dummy;
        MPI_Aint base_address;
        MPI_Get_address(&dummy, &base_address);
        MPI_Get_address(&dummy.torso_home_rank, &offsets[0]);
        MPI_Get_address(&dummy.torso_local_id, &offsets[1]);
        MPI_Get_address(&dummy.heart_home_rank, &offsets[2]);
        MPI_Get_address(&dummy.heart_local_id, &offsets[3]);
        MPI_Get_address(&dummy.dist2, &offsets[4]);
        for (int i = 0; i < nitems; i++) { offsets[i] -= base_address; }

        MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_type);
        MPI_Type_commit(&mpi_type);
        return mpi_type;
    }

    static MPI_Datatype BuildValueRequestMPIType()
    {
        MPI_Datatype mpi_type;
        const int nitems = 3;
        int blocklengths[3] = {1, 1, 1};
        MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
        MPI_Aint offsets[3];

        ValueRequest dummy;
        MPI_Aint base_address;
        MPI_Get_address(&dummy, &base_address);
        MPI_Get_address(&dummy.torso_home_rank, &offsets[0]);
        MPI_Get_address(&dummy.torso_local_id, &offsets[1]);
        MPI_Get_address(&dummy.heart_local_id, &offsets[2]);
        for (int i = 0; i < nitems; i++) { offsets[i] -= base_address; }

        MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_type);
        MPI_Type_commit(&mpi_type);
        return mpi_type;
    }

    static MPI_Datatype BuildMetaEntryMPIType()
    {
        MPI_Datatype mpi_type;
        const int nitems = 2;
        int blocklengths[2] = {1, 1};
        MPI_Datatype types[2] = {MPI_INT, MPI_INT};
        MPI_Aint offsets[2];

        MetaEntry dummy;
        MPI_Aint base_address;
        MPI_Get_address(&dummy, &base_address);
        MPI_Get_address(&dummy.peer, &offsets[0]);
        MPI_Get_address(&dummy.count, &offsets[1]);
        for (int i = 0; i < nitems; i++) { offsets[i] -= base_address; }

        MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_type);
        MPI_Type_commit(&mpi_type);
        return mpi_type;
    }

    void DebugLog(const std::string& msg) const
    {
        std::ostringstream oss;
        oss << "transfer_debug_rank_" << std::setfill('0') << std::setw(6) << my_rank_ << ".log";
        std::ofstream ofs(oss.str().c_str(), std::ios::app);
        ofs << msg << std::endl;
    }

    int GeomOwner(const double* coords) const
    {
        const double denom = std::max(global_xmax_ - global_xmin_, 1e-30);
        double xi = (coords[0] - global_xmin_) / denom;
        int owner = static_cast<int>(std::floor(xi * num_ranks_));
        if (owner < 0) { owner = 0; }
        if (owner >= num_ranks_) { owner = num_ranks_ - 1; }
        return owner;
    }

    void InitializeGlobalBBox()
    {
        double local_min[3] = { std::numeric_limits<double>::max(),
                                std::numeric_limits<double>::max(),
                                std::numeric_limits<double>::max() };
        double local_max[3] = { -std::numeric_limits<double>::max(),
                                -std::numeric_limits<double>::max(),
                                -std::numeric_limits<double>::max() };

        auto update_bbox = [&](ParMesh* mesh) {
            for (int i = 0; i < mesh->GetNV(); i++) {
                double c[3];
                mesh->GetNode(i, c);
                for (int d = 0; d < 3; d++) {
                    local_min[d] = std::min(local_min[d], c[d]);
                    local_max[d] = std::max(local_max[d], c[d]);
                }
            }
        };

        update_bbox(heart_mesh_);
        update_bbox(torso_mesh_);

        double global_min[3], global_max[3];
        MPI_Allreduce(local_min, global_min, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(local_max, global_max, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        global_xmin_ = global_min[0];
        global_xmax_ = global_max[0];
    }

    std::vector<InterfacePointRecord> CollectLocalHeartBoundaryPoints() const
    {
        std::vector<InterfacePointRecord> points;
        std::set<int> processed_vertices;

        for (int i = 0; i < heart_mesh_->GetNBE(); i++) {
            Array<int> vertices;
            heart_mesh_->GetBdrElementVertices(i, vertices);
            for (int j = 0; j < vertices.Size(); j++) {
                int vdof = vertices[j];
                if (!processed_vertices.insert(vdof).second) { continue; }

                InterfacePointRecord rec;
                heart_mesh_->GetNode(vdof, rec.coords);
                rec.home_rank = my_rank_;
                rec.local_id = vdof;
                points.push_back(rec);
            }
        }
        return points;
    }

    std::vector<InterfacePointRecord> CollectLocalTorsoDirichletPoints()
    {
        std::vector<InterfacePointRecord> points;
        std::set<int> processed_vertices;
        local_torso_dirichlet_vdofs_.clear();

        for (int i = 0; i < torso_mesh_->GetNBE(); i++) {
            if (torso_mesh_->GetBdrAttribute(i) != 100) { continue; }
            Array<int> vertices;
            torso_mesh_->GetBdrElementVertices(i, vertices);
            for (int j = 0; j < vertices.Size(); j++) {
                int vdof = vertices[j];
                if (!processed_vertices.insert(vdof).second) { continue; }

                InterfacePointRecord rec;
                torso_mesh_->GetNode(vdof, rec.coords);
                rec.home_rank = my_rank_;
                rec.local_id = vdof;
                points.push_back(rec);
                local_torso_dirichlet_vdofs_.push_back(vdof);
            }
        }
        return points;
    }

    static int FindBestHeartPoint(
        const InterfacePointRecord& target,
        const std::vector<InterfacePointRecord>& heart_points,
        const std::unordered_map<QuantizedKey, std::vector<int>, QuantizedKeyHash>& bins,
        double tol,
        double& best_dist_sq)
    {
        const double tol_sq = tol * tol;
        best_dist_sq = std::numeric_limits<double>::max();
        int best_idx = -1;

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    QuantizedKey key = MakeQuantizedKey(target.coords, tol, dx, dy, dz);
                    std::unordered_map<QuantizedKey, std::vector<int>, QuantizedKeyHash>::const_iterator it = bins.find(key);
                    if (it == bins.end()) { continue; }
                    for (size_t k = 0; k < it->second.size(); k++) {
                        int idx = it->second[k];
                        double d2 = DistanceSquared(target.coords, heart_points[idx].coords);
                        if (d2 < best_dist_sq) {
                            best_dist_sq = d2;
                            best_idx = idx;
                        }
                    }
                }
            }
        }

        if (best_idx >= 0 && best_dist_sq <= tol_sq) { return best_idx; }

        for (size_t i = 0; i < heart_points.size(); i++) {
            double d2 = DistanceSquared(target.coords, heart_points[i].coords);
            if (d2 < best_dist_sq) {
                best_dist_sq = d2;
                best_idx = static_cast<int>(i);
            }
        }
        return best_idx;
    }

    // =========================================================================
    // [FIX v4] NBX (Non-blocking Barrier eXchange) 稀疏交换:
    //   只在有实际数据的 rank 对之间建立 MPI 连接，避免 Alltoall/Alltoallv
    //   在 1536 ranks 下耗尽 InfiniBand queue pairs。
    //   算法: Hoefler et al., "Scalable Communication Protocols for
    //          Dynamic Sparse Data Exchange", PPoPP 2010.
    // =========================================================================
    template <class T>
    std::map<int, std::vector<T> > SparseExchangeViaRootMeta(
        const std::map<int, std::vector<T> >& outgoing,
        MPI_Datatype mpi_type,
        int tag_data,
        const std::string& phase_name) const
    {
        std::map<int, std::vector<T> > incoming;

        // --- Step 1: Post all non-blocking sends (sparse, only to actual peers) ---
        std::vector<MPI_Request> send_reqs;
        for (typename std::map<int, std::vector<T> >::const_iterator it = outgoing.begin();
             it != outgoing.end(); ++it)
        {
            if (it->second.empty()) { continue; }
            if (it->first == my_rank_) {
                // self-send: direct copy
                incoming[my_rank_] = it->second;
                continue;
            }
            MPI_Request r;
            MPI_Issend(const_cast<T*>(it->second.data()),
                       static_cast<int>(it->second.size()), mpi_type,
                       it->first, tag_data, MPI_COMM_WORLD, &r);
            send_reqs.push_back(r);
        }

        // --- Step 2: Probe-recv loop with non-blocking barrier for termination ---
        //   Phase A: probe & recv until all sends globally are matched (Ibarrier).
        //   Phase B: after barrier, drain any remaining messages that arrived
        //            between the last probe and barrier completion.
        bool barrier_posted = false;
        bool barrier_done = false;
        MPI_Request barrier_req = MPI_REQUEST_NULL;

        while (!barrier_done) {
            // Probe for incoming messages with this tag
            int flag = 0;
            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, tag_data, MPI_COMM_WORLD, &flag, &status);

            if (flag) {
                int count = 0;
                MPI_Get_count(&status, mpi_type, &count);
                int src = status.MPI_SOURCE;
                incoming[src].resize(count);
                MPI_Recv(incoming[src].data(), count, mpi_type,
                         src, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                continue; // check for more messages before testing barrier
            }

            // No pending message; check if we can post or test the barrier
            if (!barrier_posted) {
                // Post barrier only after all local sends are locally complete
                int all_sent = 0;
                if (send_reqs.empty()) {
                    all_sent = 1;
                } else {
                    MPI_Testall(static_cast<int>(send_reqs.size()),
                                send_reqs.data(), &all_sent, MPI_STATUSES_IGNORE);
                }
                if (all_sent) {
                    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_req);
                    barrier_posted = true;
                }
            } else {
                int bdone = 0;
                MPI_Test(&barrier_req, &bdone, MPI_STATUS_IGNORE);
                if (bdone) {
                    barrier_done = true;
                }
            }
        }

        // --- Step 3: Drain any remaining messages after barrier ---
        // (messages sent before barrier but arriving after it completes)
        while (true) {
            int flag = 0;
            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, tag_data, MPI_COMM_WORLD, &flag, &status);
            if (!flag) { break; }
            int count = 0;
            MPI_Get_count(&status, mpi_type, &count);
            int src = status.MPI_SOURCE;
            incoming[src].resize(count);
            MPI_Recv(incoming[src].data(), count, mpi_type,
                     src, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Ensure all sends are complete before returning (buffer safety)
        if (!send_reqs.empty()) {
            MPI_Waitall(static_cast<int>(send_reqs.size()),
                        send_reqs.data(), MPI_STATUSES_IGNORE);
        }

        // --- Debug logging (optional, lightweight) ---
        {
            long long local_send = 0, local_recv = 0;
            for (typename std::map<int, std::vector<T> >::const_iterator it = outgoing.begin();
                 it != outgoing.end(); ++it) {
                local_send += static_cast<long long>(it->second.size());
            }
            for (typename std::map<int, std::vector<T> >::const_iterator it = incoming.begin();
                 it != incoming.end(); ++it) {
                local_recv += static_cast<long long>(it->second.size());
            }
            long long global_send = 0, global_recv = 0;
            MPI_Allreduce(&local_send, &global_send, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_recv, &global_recv, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            std::ostringstream oss;
            oss << "[DBG] phase=" << phase_name
                << " local_send=" << local_send
                << " local_recv=" << local_recv
                << " global_send=" << global_send
                << " global_recv=" << global_recv;
            DebugLog(oss.str());
            if (global_send != global_recv && my_rank_ == 0) {
                std::cerr << "[ERR] phase=" << phase_name
                          << " global_send=" << global_send
                          << " != global_recv=" << global_recv << std::endl;
            }
        }

        return incoming;
    }

    void BuildInterfaceMap()
    {
        InitializeGlobalBBox();

        if (my_rank_ == 0) { std::cout << "[XFER] begin collect local interface points" << std::endl; }
        std::vector<InterfacePointRecord> local_heart_points = CollectLocalHeartBoundaryPoints();
        std::vector<InterfacePointRecord> local_torso_points = CollectLocalTorsoDirichletPoints();
        if (my_rank_ == 0) { std::cout << "[XFER] end   collect local interface points" << std::endl; }

        MPI_Datatype point_type = BuildPointRecordMPIType();
        MPI_Datatype match_type = BuildMatchProposalMPIType();
        MPI_Datatype request_type = BuildValueRequestMPIType();

        std::map<int, std::vector<InterfacePointRecord> > heart_to_owner;
        for (size_t i = 0; i < local_heart_points.size(); i++) {
            int owner = GeomOwner(local_heart_points[i].coords);
            heart_to_owner[owner].push_back(local_heart_points[i]);
        }

        std::map<int, std::vector<InterfacePointRecord> > torso_to_owner;
        for (size_t i = 0; i < local_torso_points.size(); i++) {
            int owner = GeomOwner(local_torso_points[i].coords);
            torso_to_owner[owner].push_back(local_torso_points[i]);
        }

        if (my_rank_ == 0) { std::cout << "[XFER] begin heart_to_owner" << std::endl; }
        std::map<int, std::vector<InterfacePointRecord> > owner_received_heart =
            SparseExchangeViaRootMeta(heart_to_owner, point_type, 8101, "heart_to_owner");
        if (my_rank_ == 0) { std::cout << "[XFER] end   heart_to_owner" << std::endl; }

        if (my_rank_ == 0) { std::cout << "[XFER] begin torso_to_owner" << std::endl; }
        std::map<int, std::vector<InterfacePointRecord> > owner_received_torso =
            SparseExchangeViaRootMeta(torso_to_owner, point_type, 8201, "torso_to_owner");
        if (my_rank_ == 0) { std::cout << "[XFER] end   torso_to_owner" << std::endl; }

        std::vector<InterfacePointRecord> owner_heart_points;
        for (std::map<int, std::vector<InterfacePointRecord> >::const_iterator it = owner_received_heart.begin(); it != owner_received_heart.end(); ++it) {
            owner_heart_points.insert(owner_heart_points.end(), it->second.begin(), it->second.end());
        }
        std::vector<InterfacePointRecord> owner_torso_points;
        for (std::map<int, std::vector<InterfacePointRecord> >::const_iterator it = owner_received_torso.begin(); it != owner_received_torso.end(); ++it) {
            owner_torso_points.insert(owner_torso_points.end(), it->second.begin(), it->second.end());
        }

        std::unordered_map<QuantizedKey, std::vector<int>, QuantizedKeyHash> heart_bins;
        for (size_t i = 0; i < owner_heart_points.size(); i++) {
            heart_bins[MakeQuantizedKey(owner_heart_points[i].coords, spatial_tol_)].push_back(static_cast<int>(i));
        }

        std::map<int, std::vector<MatchProposal> > matches_to_torso_home;
        int local_unmatched = 0;
        for (size_t i = 0; i < owner_torso_points.size(); i++) {
            double best_dist_sq = 0.0;
            int best_idx = FindBestHeartPoint(owner_torso_points[i], owner_heart_points, heart_bins, spatial_tol_, best_dist_sq);
            if (best_idx < 0) {
                local_unmatched++;
                continue;
            }
            const InterfacePointRecord &hp = owner_heart_points[best_idx];
            const InterfacePointRecord &tp = owner_torso_points[i];
            MatchProposal mp;
            mp.torso_home_rank = tp.home_rank;
            mp.torso_local_id = tp.local_id;
            mp.heart_home_rank = hp.home_rank;
            mp.heart_local_id = hp.local_id;
            mp.dist2 = best_dist_sq;
            matches_to_torso_home[tp.home_rank].push_back(mp);
        }

        int global_unmatched = 0;
        MPI_Allreduce(&local_unmatched, &global_unmatched, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (my_rank_ == 0) {
            std::cout << "[XFER] owner matching unmatched points = " << global_unmatched << std::endl;
        }

        if (my_rank_ == 0) { std::cout << "[XFER] begin match_to_torso" << std::endl; }
        std::map<int, std::vector<MatchProposal> > torso_received_matches =
            SparseExchangeViaRootMeta(matches_to_torso_home, match_type, 8301, "match_to_torso");
        if (my_rank_ == 0) { std::cout << "[XFER] end   match_to_torso" << std::endl; }

        std::map<int, std::vector<ValueRequest> > requests_to_heart;
        for (std::map<int, std::vector<MatchProposal> >::const_iterator it = torso_received_matches.begin(); it != torso_received_matches.end(); ++it) {
            for (size_t k = 0; k < it->second.size(); k++) {
                const MatchProposal &mp = it->second[k];
                ValueRequest req;
                req.torso_home_rank = my_rank_;
                req.torso_local_id = mp.torso_local_id;
                req.heart_local_id = mp.heart_local_id;
                requests_to_heart[mp.heart_home_rank].push_back(req);
            }
        }

        recv_peers_.clear();
        recv_torso_local_ids_per_peer_.clear();
        self_torso_local_ids_.clear();
        self_heart_local_ids_.clear();
        for (std::map<int, std::vector<ValueRequest> >::const_iterator it = requests_to_heart.begin(); it != requests_to_heart.end(); ++it) {
            if (it->first == my_rank_) {
                for (size_t k = 0; k < it->second.size(); k++) {
                    self_torso_local_ids_.push_back(it->second[k].torso_local_id);
                    self_heart_local_ids_.push_back(it->second[k].heart_local_id);
                }
            } else {
                recv_peers_.push_back(it->first);
                std::vector<int> tdofs;
                tdofs.reserve(it->second.size());
                for (size_t k = 0; k < it->second.size(); k++) {
                    tdofs.push_back(it->second[k].torso_local_id);
                }
                recv_torso_local_ids_per_peer_.push_back(tdofs);
            }
        }

        if (my_rank_ == 0) { std::cout << "[XFER] begin request_to_heart" << std::endl; }
        std::map<int, std::vector<ValueRequest> > heart_received_requests =
            SparseExchangeViaRootMeta(requests_to_heart, request_type, 8401, "request_to_heart");
        if (my_rank_ == 0) { std::cout << "[XFER] end   request_to_heart" << std::endl; }

        send_peers_.clear();
        send_heart_local_ids_per_peer_.clear();
        for (std::map<int, std::vector<ValueRequest> >::const_iterator it = heart_received_requests.begin(); it != heart_received_requests.end(); ++it) {
            if (it->first == my_rank_) { continue; }
            send_peers_.push_back(it->first);
            std::vector<int> hdofs;
            hdofs.reserve(it->second.size());
            for (size_t k = 0; k < it->second.size(); k++) {
                hdofs.push_back(it->second[k].heart_local_id);
            }
            send_heart_local_ids_per_peer_.push_back(hdofs);
        }

        MPI_Type_free(&point_type);
        MPI_Type_free(&match_type);
        MPI_Type_free(&request_type);

        {
            std::ostringstream oss;
            oss << "[DBG] final_plan send_peers=" << send_peers_.size()
                << " recv_peers=" << recv_peers_.size()
                << " self_pairs=" << self_heart_local_ids_.size();
            DebugLog(oss.str());
        }
    }

public:
    HeartTorsoBoundaryTransfer(ParMesh* heart_mesh,
                               ParFiniteElementSpace* heart_fes,
                               ParGridFunction* heart_gf,
                               ParMesh* torso_mesh,
                               ParFiniteElementSpace* torso_fes,
                               ParGridFunction* torso_gf,
                               double tol)
        : heart_mesh_(heart_mesh),
          heart_fes_(heart_fes),
          heart_gf_(heart_gf),
          torso_mesh_(torso_mesh),
          torso_fes_(torso_fes),
          torso_gf_(torso_gf),
          spatial_tol_(tol),
          my_rank_(0),
          num_ranks_(1),
          global_xmin_(0.0),
          global_xmax_(1.0)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks_);
        BuildInterfaceMap();
    }

    void TransferValuesToTorsoBoundary()
    {
        for (size_t i = 0; i < local_torso_dirichlet_vdofs_.size(); i++) {
            (*torso_gf_)(local_torso_dirichlet_vdofs_[i]) = DEFAULT_HEART_POTENTIAL;
        }

        for (size_t i = 0; i < self_heart_local_ids_.size() && i < self_torso_local_ids_.size(); i++) {
            (*torso_gf_)(self_torso_local_ids_[i]) = (*heart_gf_)(self_heart_local_ids_[i]);
        }

        std::vector<std::vector<double> > recv_buffers(recv_peers_.size());
        std::vector<std::vector<double> > send_buffers(send_peers_.size());
        std::vector<MPI_Request> reqs;

        for (size_t i = 0; i < recv_peers_.size(); i++) {
            recv_buffers[i].resize(recv_torso_local_ids_per_peer_[i].size(), DEFAULT_HEART_POTENTIAL);
            if (!recv_buffers[i].empty()) {
                MPI_Request r;
                MPI_Irecv(recv_buffers[i].data(), static_cast<int>(recv_buffers[i].size()), MPI_DOUBLE,
                          recv_peers_[i], 8501, MPI_COMM_WORLD, &r);
                reqs.push_back(r);
            }
        }

        for (size_t i = 0; i < send_peers_.size(); i++) {
            send_buffers[i].resize(send_heart_local_ids_per_peer_[i].size(), 0.0);
            for (size_t k = 0; k < send_heart_local_ids_per_peer_[i].size(); k++) {
                send_buffers[i][k] = (*heart_gf_)(send_heart_local_ids_per_peer_[i][k]);
            }
            if (!send_buffers[i].empty()) {
                MPI_Request r;
                MPI_Isend(send_buffers[i].data(), static_cast<int>(send_buffers[i].size()), MPI_DOUBLE,
                          send_peers_[i], 8501, MPI_COMM_WORLD, &r);
                reqs.push_back(r);
            }
        }

        if (!reqs.empty()) {
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        }

        for (size_t i = 0; i < recv_buffers.size(); i++) {
            for (size_t k = 0; k < recv_buffers[i].size() && k < recv_torso_local_ids_per_peer_[i].size(); k++) {
                (*torso_gf_)(recv_torso_local_ids_per_peer_[i][k]) = recv_buffers[i][k];
            }
        }
    }
};

// 定义边界条件类型的映射
enum BoundaryType {
    NEUMANN_ZERO = 1,
    DIRICHLET = 100,
    SOURCE = 100
};

// 用于比较两个点坐标是否相等的辅助结构体
struct Point3D {
   double x, y, z;
   
   Point3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
   
   // 重载小于运算符以便用于map
   bool operator<(const Point3D& other) const {
       if (x != other.x) return x < other.x;
       if (y != other.y) return y < other.y;
       return z < other.z;
   }
   
   // 点坐标相等性检查（带容差）
   bool equals(const Point3D& other, double tolerance = 1e-4) const {
       return std::abs(x - other.x) < tolerance && 
              std::abs(y - other.y) < tolerance && 
              std::abs(z - other.z) < tolerance;
   }
};




/**
 * 检查电导率张量是否已正确设置
 * 
 * @param sigma 要检查的电导率张量
 * @throws 如果电导率张量为空则抛出异常
 */
void checkConductivityTensors(MatrixElementPiecewiseCoefficient& sigma) {
    // 检查 heartConductivities_ 中是否有条目
    if (sigma.heartConductivities_.empty()) {
        throw std::runtime_error("错误：电导率张量为空!");
    }
}



//Stolen from SingleCell
class Timeline
{
 public:
   Timeline(double dt, double duration)
   {
      maxTimesteps_ = round(duration/dt);
      dt_ = duration/maxTimesteps_;
   }
   int maxTimesteps() const { return maxTimesteps_; };
   double dt() const { return dt_; }
   double maxTime() const { return dt_*maxTimesteps_; }
   double realTimeFromTimestep(int timestep) const
   {
      return timestep*dt_;
   }
   int timestepFromRealTime(double realTime) const
   {
      return round(realTime/dt_);
   }
   std::string outputIdFromTimestep(const int timestep) const
   {
      double resolution = 1e-3;
      int width = 8;
      while (resolution > dt_) {
         resolution /= 10;
         width++;
      }
      std::stringstream ss;
      ss << std::setfill('0') << std::setw(width)
         << int(round(dt_*timestep/resolution));
      return ss.str();
   }

 private:
   double dt_;
   int maxTimesteps_;
};

class OutputCoordinator
{

 private:
   
};

void recursive_mkdir(const std::string dirname, mode_t mode=S_IRWXU|S_IRWXG)
{
   int startSearch=0;
   do
   {
      int endSearch = dirname.find("/", startSearch);
      //if directory doesn't exist
      if (endSearch < 0) {
         endSearch = dirname.length();
      }
      std::string thisDirname = dirname.substr(0, endSearch);
      DIR* dir = opendir(thisDirname.c_str());
      if (dir)
      {
         closedir(dir);
      }
      else if (ENOENT == errno) {
         //make the directory
         int ret = mkdir(thisDirname.c_str(), mode);
         assert(ret == 0);
      }
      startSearch=endSearch+1;
   } while (startSearch < dirname.length());
}



int main(int argc, char *argv[])
{
   MPI_Init(NULL,NULL);
   int num_ranks, my_rank;

   MPI_Comm_size(COMM_LOCAL,&num_ranks);
   MPI_Comm_rank(COMM_LOCAL,&my_rank);


   units_internal(1e-3, 1e-9, 1e-3, 1e-3, 1, 1e-9, 1);
   units_external(1e-3, 1e-9, 1e-3, 1e-3, 1, 1e-9, 1);

   bool use_petsc = true;//false;true
   const char *petscrc_file = "rc_fem_heart";
   MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

   // Optional PETSc runtime cap for number of time steps.
   // -max_time_steps <N>, N >= 0; when unset, run all timeline steps.
   PetscInt max_time_steps_opt = -1;
   PetscBool has_max_time_steps_opt = PETSC_FALSE;
   PetscErrorCode petsc_opt_ierr =
       PetscOptionsGetInt(NULL, NULL, "-max_time_steps",
                          &max_time_steps_opt, &has_max_time_steps_opt);
   if (petsc_opt_ierr)
   {
      if (my_rank == 0)
      {
         std::cerr << "Warning: failed to parse PETSc option -max_time_steps; "
                   << "ignoring step cap." << std::endl;
      }
      has_max_time_steps_opt = PETSC_FALSE;
      max_time_steps_opt = -1;
   }
   int user_max_time_steps = -1;
   if (has_max_time_steps_opt && max_time_steps_opt >= 0)
   {
      user_max_time_steps = static_cast<int>(max_time_steps_opt);
   }

   PetscInt mesh_refine_levels_opt = 0;
   PetscOptionsGetInt(NULL, NULL, "-mesh_refine_levels",
                      &mesh_refine_levels_opt, NULL);
   const int mesh_refine_levels =
      std::max(0, static_cast<int>(mesh_refine_levels_opt));

{

       // --- Timer Variable Declarations ---
    double t_start, t_end; // Temporary start/end times
    double t_total_elapsed = 0.0;
    double t_total = 0.0;
    double t_problem1_total = 0.0; // e.g., Heart Solve
    double t_problem2_total = 0.0; // e.g., BC Transfer
    double t_problem3_total = 0.0; // e.g., Torso Solve
    double t_ksp1_total = 0.0;     // e.g., Heart KSP
    double t_ksp2_total = 0.0;     // e.g., Torso KSP
    double t_ksp3_total = 0.0;     // e.g., Other KSP (if applicable)
    double t_ionic_model_total = 0.0;
    double t_ionic_start, t_ksp1_start, t_ksp2_start, t_ksp3_start;
    double t_ionic_end, t_ksp1_end, t_ksp2_end, t_ksp3_end;

    static int total_iterations_monodomain = 0;
    static int solve_count_monodomain = 0;
    static int total_iterations_recoverue = 0;
    static int solve_count_recoverue = 0;
    static int total_iterations_torso = 0;
    static int solve_count_torso = 0;




   if (my_rank == 0)
   {
      std::cout << "Initializing with " << num_ranks << " MPI ranks." << std::endl;
   }
   
   int order = 1;

   std::vector<std::string> objectFilenames;
   if (argc == 1)
      objectFilenames.push_back("femheart.data");

   for (int iargCursor=1; iargCursor<argc; iargCursor++)
      objectFilenames.push_back(argv[iargCursor]);

   if (my_rank == 0) {
      for (int ii=0; ii<objectFilenames.size(); ii++)
	 object_compilefile(objectFilenames[ii].c_str());
   }
   object_Bcast(0,MPI_COMM_WORLD);

   OBJECT* obj = object_find("femheart", "HEART");
   assert(obj != NULL);

   StartTimer("Read the mesh");

   //const char *coll_name = "par-data-" << std::setfill('0') << std::setw(6) << num_ranks;

   Mesh* torso_mesh = nullptr;
   ParMesh* pmesh_torso = nullptr;
   ParFiniteElementSpace* pfespace_torso = nullptr;
   ParGridFunction* gf_ue_torso = nullptr;


std::string data_path = "parData/";
std::ostringstream oss_coll_name;
oss_coll_name << "par-data-" << std::setfill('0') << std::setw(6) << num_ranks;
std::string coll_name = oss_coll_name.str();
VisItDataCollection visit_dc(MPI_COMM_WORLD, coll_name);
visit_dc.SetPrefixPath(data_path);




   ParMesh *pmesh;
   ParGridFunction *saved_fiber;
   ParGridFunction *saved_sheet;
   ParGridFunction *saved_transverse;

   visit_dc.Load();
    //cout << "visit_dc Loaded;" << endl;
    pmesh = dynamic_cast<ParMesh*>(visit_dc.GetMesh());

    int ne_before_ = pmesh->GetNE();
for (int ilev = 0; ilev < mesh_refine_levels; ++ilev)
{
    pmesh->UniformRefinement();
}
int ne_after_ = pmesh->GetNE();

if (my_rank == 0) {
    cout << "mesh_refine_levels: " << mesh_refine_levels << endl;
    cout << "heart细化前单元数: " << ne_before_ << endl;
    cout << "heart细化后单元数: " << ne_after_ << endl;
    cout << "增长倍数: " << (double)ne_after_/ne_before_ << endl;
}

   //saved_fiber = visit_dc.GetParField("fiber");
   //saved_sheet = visit_dc.GetParField("sheet");
   //saved_transverse = visit_dc.GetParField("trans");
   //cout << "visit_dc Got ParField" << endl;


    std::ostringstream oss_coll_name_torso;
    oss_coll_name_torso << "par-torso-data-" << std::setfill('0') << std::setw(6) << num_ranks;
    std::string coll_name_torso = oss_coll_name_torso.str();
    VisItDataCollection visit_dc_torso(MPI_COMM_WORLD, coll_name_torso);
    visit_dc_torso.SetPrefixPath(data_path);
    visit_dc_torso.Load();
    //cout << "visit_dc_torso Loaded;" << endl;
    pmesh_torso = dynamic_cast<ParMesh*>(visit_dc_torso.GetMesh());
    
    // 验证细化前后的单元数量
int ne_before = pmesh_torso->GetNE();
for (int ilev = 0; ilev < mesh_refine_levels; ++ilev)
{
    pmesh_torso->UniformRefinement();
}
int ne_after = pmesh_torso->GetNE();

if (my_rank == 0) {
    cout << "细化前单元数: " << ne_before << endl;
    cout << "细化后单元数: " << ne_after << endl;
    cout << "增长倍数: " << (double)ne_after/ne_before << endl;
}


   // Read shared global mesh
   //mfem::Mesh *mesh = nullptr;
   if (my_rank == 0)
   {
    //mesh = ecg_readMeshptr(obj, "mesh");
   }
   //mfem::Mesh *mesh = ecg_readMeshptr(obj, "mesh");
   EndTimer();
   int dim = pmesh->Dimension();

   std::string torso_mesh_file;
   objectGet(obj, "torso_mesh", torso_mesh_file, "");

   //Fill in the MatrixElementPiecewiseCoefficients
   std::vector<int> heartRegions;
   objectGet(obj,"heart_regions", heartRegions);

   std::vector<double> sigma_m;
   objectGet(obj,"sigma_m",sigma_m);
   assert(heartRegions.size()*3 == sigma_m.size());

   // 读取细胞内电导率
   std::vector<double> sigma_i_values;
   objectGet(obj, "sigma_i", sigma_i_values);
   assert(heartRegions.size()*3 == sigma_i_values.size());

   // 读取细胞外电导率
   std::vector<double> sigma_e_values;
   objectGet(obj, "sigma_e", sigma_e_values);
   assert(heartRegions.size()*3 == sigma_e_values.size());

   double sigma_torso;
   objectGet(obj, "sigma_torso", sigma_torso, "0.2");  // 默认值0.2 mS/mm

   // 检查是否应该求解细胞外电位
   bool solveForUe;
   objectGet(obj, "solve_for_ue", solveForUe, "1");  // 默认启用

   bool solve_torso_model;
   objectGet(obj, "solve_torso", solve_torso_model, "1");  // 默认开启

   int heart_boundary_marker;
   objectGet(obj, "heart_bdry_marker", heart_boundary_marker, "1");  // 默认为1

   double tolerance;
   objectGet(obj, "mesh_tolerance", tolerance, "1e-6");  // 读取容差设置

   double dt;
   objectGet(obj,"dt",dt,"0.01 ms");
   double Bm;
   objectGet(obj,"Bm",Bm,"140"); // 1/mm
   double Cm;
   objectGet(obj,"Cm",Cm,"0.01"); // 1 uF/cm^2 = 0.01 uF/mm^2
 
   std::string reactionName;
   objectGet(obj, "reaction", reactionName, "BetterTT06");

   std::string outputDir;
   objectGet(obj, "outdir", outputDir, ".");
   
   double endTime;
   objectGet(obj, "end_time", endTime, "0 ms");

   double outputRate;
   objectGet(obj, "output_rate", outputRate, "1 ms");

   //double checkpointRate;
   //objectGet(obj, "checkpoint_rate", checkpointRate, "100 ms");

   double initVm;
   objectGet(obj, "init_vm", initVm, "-83");

   bool useNodalIion;
   objectGet(obj, "nodal_ion", useNodalIion, "1");


   StimulusCollection stims(dt);
   {
      std::vector<std::string> stimulusNames;
      objectGet(obj, "stimulus", stimulusNames);
      for (auto name : stimulusNames)
      {
         OBJECT* stimobj = object_find(name.c_str(), "STIMULUS");
         assert(stimobj != NULL);
         int numTimes;
         objectGet(stimobj, "n", numTimes, "1");
         double bcl;
         objectGet(stimobj, "bcl", bcl, "0 ms");
         assert(numTimes == 1 || bcl != 0);
         double startTime;
         objectGet(stimobj, "start", startTime, "0 ms");
         double duration;
         objectGet(stimobj, "duration", duration, "1 ms");
         double strength;
         objectGet(stimobj, "strength", strength, "0"); //uA/uF
         assert(strength >= 0);
         std::string location;
         objectGet(stimobj, "where", location, "");
         assert(!location.empty());
         OBJECT* locobj = object_find(location.c_str(), "REGION");
         assert(locobj != NULL);
         std::string regionType;
         objectGet(locobj, "type", regionType, "");
         assert(!regionType.empty());
         shared_ptr<StimulusLocation> stimLoc;
         if (regionType == "ball")
         {
            std::vector<double> center;
            objectGet(locobj, "center", center);
            assert(center.size() == 3);
            double radius;
            objectGet(locobj, "radius", radius, "-1");
            assert(radius >= 0);
            stimLoc = std::make_shared<CenterBallStimulus>(center[0],center[1],center[2],radius);
         }
         else if (regionType == "box")
         {
            std::vector<double> lower;
            objectGet(locobj, "lower", lower);
            assert(lower.size() == 3);
            vector<double> upper;
            objectGet(locobj, "upper", upper);
            assert(upper.size() == 3);
            stimLoc = std::make_shared<BoxStimulus>
               (lower[0], upper[0],
                lower[1], upper[1],
                lower[2], upper[2]);
         }
         shared_ptr<StimulusWaveform> stimWave(new SquareWaveform());
         stims.add(Stimulus(numTimes, startTime, duration, bcl, strength, stimLoc, stimWave));
      }
   }
   
   Timeline timeline(dt, endTime);  
   int max_time_steps = timeline.maxTimesteps();
   if (user_max_time_steps >= 0)
   {
      max_time_steps = std::min(max_time_steps, user_max_time_steps);
   }
   if (my_rank == 0 && user_max_time_steps >= 0)
   {
      std::cout << "PETSc max_time_steps limit enabled: " << max_time_steps
                << " (timeline default: " << timeline.maxTimesteps() << ")"
                << std::endl;
   }

   // Scaled additive Schwarz (sASM) configuration for the strong-scaling study.
   // All three linear systems (monodomain, u_e recovery, torso) are solved with
   // CG preconditioned by  D^{-1/2} M_ASM(BASIC)^{-1} D^{-1/2},  where D is the
   // overlap multiplicity. The preconditioner is installed per solver via
   // AttachScaledASM() right after each SetOperator(). Overlap and ICC fill
   // levels are shared, runtime-tunable knobs:
   //   -asm_overlap <N>     (default 1)
   //   -asm_icc_levels <L>  (default 0)
   PetscInt asm_overlap = 1;
   PetscOptionsGetInt(NULL, NULL, "-asm_overlap", &asm_overlap, NULL);
   asm_overlap = std::max<PetscInt>(0, asm_overlap);

   PetscInt asm_icc_levels = 0;
   PetscOptionsGetInt(NULL, NULL, "-asm_icc_levels", &asm_icc_levels, NULL);
   asm_icc_levels = std::max<PetscInt>(0, asm_icc_levels);

   if (my_rank == 0)
   {
      std::cout << "[sASM] scaled ASM+CG for all systems: overlap = "
                << asm_overlap << ", ICC levels = " << asm_icc_levels
                << std::endl;
   }

   StartTimer("Setting Attributes");
   pmesh->SetAttributes();
   EndTimer();

   //StartTimer("Partition Mesh");
   // If I read correctly, pmeshpart will now point to an integer array
   //  containing a partition ID (rank!) for every element ID.
   //int *pmeshpart = mesh->GeneratePartitioning(num_ranks);



   
   if (my_rank == 0)
   {
      for(int i=0; i<num_ranks; i++) {
         //std::cout << "Rank " << i << " has " << local_extents[i+1]-local_extents[i] << " nodes!" << std::endl;
      }
   }
   //ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, pmeshpart);
   
   // Build a new FEC...
   FiniteElementCollection *fec;
   if (my_rank == 0) { std::cout << "Creating new FEC..." << std::endl; }
   fec = new H1_FECollection(order, dim);
   // ...and corresponding FES
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec);
   //FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   std::cout << "[" << my_rank << "] Number of finite element unknowns: "
	     << pfespace->GetTrueVSize() << std::endl;

   // 5. Determine the list of true (i.e. conforming) essential boundary DOFs
   Array<int> ess_tdof_list;   // Essential true degrees of freedom
   // "true" takes into account shared vertices.
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
      pfespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   

// 1. Create a new VECTOR finite element space for the 3D fiber directions.
//    We use the same FE collection but specify a vector dimension of 3.
const int vdim = 3;
ParFiniteElementSpace *pfespace_vec = new ParFiniteElementSpace(pmesh, fec, vdim);

// 2. Create shared_ptrs for the ParGridFunctions that will hold the direction vectors.
//    These are created on the vector FESpace.
auto fiber_quat = std::make_shared<ParGridFunction>(pfespace_vec);
auto sheet_quat = std::make_shared<ParGridFunction>(pfespace_vec);
auto transverse_quat = std::make_shared<ParGridFunction>(pfespace_vec);

// 3. Define the constant vectors for each direction.
Vector fiber_direction(vdim);
fiber_direction(0) = 1.0; fiber_direction(1) = 0.0; fiber_direction(2) = 0.0;
VectorConstantCoefficient fiber_coeff(fiber_direction);

Vector sheet_direction(vdim);
sheet_direction(0) = 0.0; sheet_direction(1) = 1.0; sheet_direction(2) = 0.0;
VectorConstantCoefficient sheet_coeff(sheet_direction);

Vector trans_direction(vdim);
trans_direction(0) = 0.0; trans_direction(1) = 0.0; trans_direction(2) = 1.0;
VectorConstantCoefficient trans_coeff(trans_direction);

// 4. Project these constant vector coefficients onto the grid functions.
//    This assigns the specified vector to every node in the mesh.
fiber_quat->ProjectCoefficient(fiber_coeff);
sheet_quat->ProjectCoefficient(sheet_coeff);
transverse_quat->ProjectCoefficient(trans_coeff);












   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to pfespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction gf_Vm(pfespace);
   ParGridFunction gf_ue(pfespace);  // 用于细胞外电位的网格函数
   ParGridFunction gf_b(pfespace);
   gf_Vm = initVm;
   gf_ue = 0.0;  // 初始化为零
   gf_b = 0.0;


   
   

    if (my_rank == 0) {
        std::cout << "\n===== 设置Torso模型 =====\n" << std::endl;
        std::cout << "Torso电导率: " << sigma_torso << " mS/mm" << std::endl;
        std::cout << "点匹配容差: " << tolerance << std::endl;
    }
    
    // 使用改进的函数读取torso网格并识别边界
    //torso_mesh = new Mesh(torso_mesh_file, 1, 1);

            // 设置躯干网格的边界属性
           // for (int i = 0; i < torso_mesh->GetNBE(); i++) {
           //   int bid = torso_mesh->GetBdrAttribute(i);
            //  if (bid == 1) {
             //     torso_mesh->SetBdrAttribute(i, BoundaryType::NEUMANN_ZERO);
             // } else if (bid == 100) {
              //    torso_mesh->SetBdrAttribute(i, BoundaryType::DIRICHLET);
             // }}
             // 设置躯干网格的边界属性
for (int i = 0; i < pmesh_torso->GetNBE(); i++) {
    int bid = pmesh_torso->GetBdrAttribute(i);
    if (bid == 1) {
        pmesh_torso->SetBdrAttribute(i, BoundaryType::NEUMANN_ZERO);
    } else if (bid == 100) {
        pmesh_torso->SetBdrAttribute(i, BoundaryType::DIRICHLET);
    }
}
    
    if (my_rank == 0) {
        std::cout << "Torso网格准备完成，进行分区..." << std::endl;
    }
    



    // 在时间循环之前初始化交界面处理对象 - 添加这段代码
    HeartTorsoBoundaryTransfer* interface_transfer = nullptr;
        
    
    if (my_rank == 0) {
        std::cout << "创建Torso有限元空间..." << std::endl;
    }
    
    // 为torso创建有限元空间，与heart使用相同的有限元类型
    FiniteElementCollection* torso_fec = new H1_FECollection(1, 3);
    pfespace_torso = new ParFiniteElementSpace(pmesh_torso, torso_fec);

    
    if (my_rank == 0) {
        std::cout << "Torso有限元空间创建完成，自由度数量: " << pfespace_torso->GetTrueVSize() << std::endl;
    }
    


    
    // 创建torso解向量
    gf_ue_torso = new ParGridFunction(pfespace_torso);
    *gf_ue_torso = 0.0;
    

    
    if (my_rank == 0) {
        std::cout << "\n===== Torso模型设置完成 =====\n" << std::endl;
    }



   //std::shared_ptr<ParGridFunction> fiber_quat  = std::make_shared<mfem::ParGridFunction>(*saved_fiber);
   //std::shared_ptr<ParGridFunction> sheet_quat  = std::make_shared<mfem::ParGridFunction>(*saved_sheet);
   //std::shared_ptr<ParGridFunction> transverse_quat  = std::make_shared<mfem::ParGridFunction>(*saved_transverse);

   
   // Load conductivity data
   MatrixElementPiecewiseCoefficient sigma_m_pos_coeffs(fiber_quat, sheet_quat, transverse_quat);
   MatrixElementPiecewiseCoefficient sigma_m_neg_coeffs(fiber_quat, sheet_quat, transverse_quat);
   for (int ii=0; ii<heartRegions.size(); ii++) {
      int heartCursor=3*ii;
      Vector sigma_m_vec(&sigma_m[heartCursor],3);
      Vector sigma_m_pos_vec(3);
      Vector sigma_m_neg_vec(3);
      for (int jj=0; jj<3; jj++)
      {
         double value = sigma_m[jj]*dt/2/Bm/Cm;
         sigma_m_pos_vec[jj] = value;
         sigma_m_neg_vec[jj] = -value;
      }
    
      sigma_m_pos_coeffs.heartConductivities_[heartRegions[ii]] = sigma_m_pos_vec;
      sigma_m_neg_coeffs.heartConductivities_[heartRegions[ii]] = sigma_m_neg_vec;
   }

   // 准备求解细胞外电位
   if (solveForUe) {
      if (my_rank == 0) {
         std::cout << "准备求解细胞外电位..." << std::endl;
         
         // 打印电导率信息
         std::cout << "心脏区域数量: " << heartRegions.size() << std::endl;
         std::cout << "细胞内电导率值数量: " << sigma_i_values.size() << std::endl;
         std::cout << "细胞外电导率值数量: " << sigma_e_values.size() << std::endl;
         
         // 打印一些样本值
         if (!heartRegions.empty()) {
            std::cout << "第一个心脏区域: " << heartRegions[0] << std::endl;
         }
         if (sigma_i_values.size() >= 3) {
            std::cout << "第一组细胞内电导率: " 
                      << sigma_i_values[0] << ", " 
                      << sigma_i_values[1] << ", " 
                      << sigma_i_values[2] << std::endl;
         }
      }
   }
   auto DebugMatrixStage = [&](const char *msg)
   {
      if (my_rank == 0)
      {
         std::cout << "[DBG matrix] " << msg << std::endl;
      }
   };

   StartTimer("Forming bilinear system (RHS)");


   ConstantCoefficient one(1.0);
   ParBilinearForm *b = new ParBilinearForm(pfespace);
   b->AddDomainIntegrator(new DiffusionIntegrator(sigma_m_neg_coeffs));
   b->AddDomainIntegrator(new MassIntegrator(one));
   b->Assemble();
   // This creates the linear algebra problem.
   HypreParMatrix RHS_mat;
   DebugMatrixStage("RHS FormSystemMatrix begin");
   b->FormSystemMatrix(ess_tdof_list, RHS_mat);
   DebugMatrixStage("RHS FormSystemMatrix done");
   EndTimer();

   StartTimer("Forming bilinear system (LHS)");
   
   // Brought out of loop to avoid unnecessary duplication
   ParBilinearForm *a = new ParBilinearForm(pfespace);   // defines a.
   a->AddDomainIntegrator(new DiffusionIntegrator(sigma_m_pos_coeffs));
   a->AddDomainIntegrator(new MassIntegrator(one));
   a->Update(pfespace);
   a->Assemble(use_petsc ? 0 : 1);

   HypreParMatrix LHS_mat;
   HyprePCG* pcg = nullptr;
   HypreSolver *M_test = nullptr;

   PetscPCGSolver* pcg_monodomain_petsc = nullptr;
   PetscParMatrix *LHS_monodomain_petsc = new PetscParMatrix;
if(!use_petsc)//use petsc
{
   a->FormSystemMatrix(ess_tdof_list,LHS_mat);
   pcg = new HyprePCG(LHS_mat);
   pcg->SetTol(1e-6);
   pcg->SetMaxIter(1000);
   pcg->SetPrintLevel(2);
   M_test = new HypreBoomerAMG(LHS_mat);
   pcg->SetPreconditioner(*M_test);
}
else
{
    DebugMatrixStage("monodomain Hypre FormSystemMatrix begin");
    a->FormSystemMatrix(ess_tdof_list, LHS_mat);
    DebugMatrixStage("monodomain Hypre FormSystemMatrix done");
    DebugMatrixStage("monodomain safe PETSc conversion begin");
    ConvertHypreToPetscAIJSafe(LHS_mat, *LHS_monodomain_petsc,
                               "monodomain_LHS", my_rank);
    DebugMatrixStage("monodomain safe PETSc conversion done");
    pcg_monodomain_petsc = new PetscPCGSolver(MPI_COMM_WORLD, "monodomain_", true);
   DebugMatrixStage("monodomain PETSc SetOperator begin");
   pcg_monodomain_petsc->SetOperator(*LHS_monodomain_petsc);
   DebugMatrixStage("monodomain PETSc SetOperator done");
   pcg_monodomain_petsc->SetRelTol(1e-6);
   //pcg_monodomain_petsc->SetAbsTol(1e-12);
   pcg_monodomain_petsc->SetMaxIter(1000);
   pcg_monodomain_petsc->SetPrintLevel(2);
    pcg_monodomain_petsc->iterative_mode = true;
   pcg_monodomain_petsc->Customize(true);
   AttachScaledASM(pcg_monodomain_petsc, MPI_COMM_WORLD,
                   static_cast<int>(asm_overlap),
                   static_cast<int>(asm_icc_levels));
}
   EndTimer();








   //Set up the ionic models
   ParLinearForm *c = new ParLinearForm(pfespace);
   //positive dt here because the reaction models use dVm = -Iion
   c->AddDomainIntegrator(new DomainLFIntegrator(stims));


   
   
   ThreadServer& threadServer = ThreadServer::getInstance();
   ThreadTeam defaultGroup = threadServer.getThreadTeam(vector<unsigned>());
   std::vector<std::string> reactionNames;
   objectGet(obj, "reaction", reactionNames);
   //reactionNames.push_back(reactionName);
   std::vector<int> cellTypes;

   //int Iion_order = 2*order+3;
   int Iion_order = 2*order-1;
   QuadratureSpace quadSpace(pmesh, Iion_order);
   if (useNodalIion)
   {
      //for (int ranklookup=local_extents[my_rank]; ranklookup<local_extents[my_rank+1]; ranklookup++)
      for (int i = 0; i < pfespace->GetNE(); i++)
      {
         //cellTypes.push_back(material_from_ranklookup[ranklookup]);
         cellTypes.push_back(1);
      }
   }
   else
   {
      for (int i = 0; i < pfespace->GetNE(); ++i)
      {
         ElementTransformation *T = pfespace->GetElementTransformation(i);
         //This is a hack.  There's no way to get access to the offsets() array
         //in Quadrature Space without declaring ourselves to be a friend class.
         //This is broken and I hope it is fixed in 4.0
         Vector localVm;
         int NNN = quadSpace.GetElementIntRule(i).GetNPoints();
         for ( int j=0; j<NNN; j++)
         {
            cellTypes.push_back(T->Attribute);
         }
      }
   }

   ReactionWrapper reactionWrapper(dt,reactionNames,defaultGroup,cellTypes);
   reactionWrapper.Initialize();
   cellTypes.clear();
   reactionNames.clear();
   
   ParBilinearForm *Iion_blf;
   HypreParMatrix Iion_mat;
   ConstantCoefficient dt_coeff(dt);
   ReactionFunction* rf = NULL;
   if (useNodalIion) {
      Iion_blf = new ParBilinearForm(pfespace);
      Iion_blf->AddDomainIntegrator(new MassIntegrator(dt_coeff));
      Iion_blf->Update(pfespace);
      Iion_blf->Assemble();
      DebugMatrixStage("Iion FormSystemMatrix begin");
      Iion_blf->FormSystemMatrix(ess_tdof_list,Iion_mat);
      DebugMatrixStage("Iion FormSystemMatrix done");
   } else {
      Iion_blf = NULL;
      
      rf = new ReactionFunction(&quadSpace,pfespace,&reactionWrapper); 
      c->AddDomainIntegrator(new QuadratureIntegrator(rf, dt)); 
   }

   Vector actual_Vm(pfespace->GetTrueVSize()), actual_b(pfespace->GetTrueVSize()), actual_old(pfespace->GetTrueVSize());
   Vector actual_Iion(pfespace->GetTrueVSize());
   bool first=true;

   if (useNodalIion)
   {
      actual_Vm = reactionWrapper.getVmReadonly();
   }



 // 创建电导率系数（在时间循环外，只创建一次）
MatrixElementPiecewiseCoefficient sigma_i(fiber_quat, sheet_quat, transverse_quat);
MatrixElementPiecewiseCoefficient sigma_sum(fiber_quat, sheet_quat, transverse_quat);

for (int ii = 0; ii < heartRegions.size(); ii++) {
    int heartCursor = 3 * ii;
    
    Vector sigma_i_vec(3);
    Vector sigma_e_vec(3);
    Vector sigma_sum_vec(3);
    
    for (int jj = 0; jj < 3; jj++) {
        sigma_i_vec[jj] = sigma_i_values[heartCursor + jj];
        sigma_e_vec[jj] = sigma_e_values[heartCursor + jj];
        sigma_sum_vec[jj] = (sigma_i_vec[jj] + sigma_e_vec[jj]);
    }
    
    sigma_i.heartConductivities_[heartRegions[ii]] = sigma_i_vec;
    sigma_sum.heartConductivities_[heartRegions[ii]] = sigma_sum_vec;
}

// 设置左侧矩阵: -∇·((σ_i + σ_e)∇u_e)
ParBilinearForm *a_pblf_recoverue = new ParBilinearForm(pfespace);
a_pblf_recoverue->AddDomainIntegrator(new DiffusionIntegrator(sigma_sum));
a_pblf_recoverue->Assemble(use_petsc ? 0 : 1);  // 注意这里的参数，与torso一致
a_pblf_recoverue->Finalize();

// 设置右侧向量的临时双线性形式: ∇·(σ_i∇V_m)
ParBilinearForm* temp_form = new ParBilinearForm(pfespace);  // 改为指针，与torso风格一致
temp_form->AddDomainIntegrator(new DiffusionIntegrator(sigma_i));
temp_form->Assemble(use_petsc ? 0 : 1);
temp_form->Finalize();

// 声明矩阵和求解器（与torso完全一致的风格）
HypreBoomerAMG* precond_recoverue_hypre = nullptr;
HyprePCG* pcg_recoverue_hypre = nullptr;
HypreParMatrix A_recoverue_hypre;  // 栈对象，与torso一致

PetscPCGSolver* pcg_recoverue_petsc = nullptr;
PetscParMatrix *A_recoverue_petsc = new PetscParMatrix;

// 临时矩阵也用相同风格
HypreParMatrix A_temp_hypre;
PetscParMatrix *A_temp_petsc = new PetscParMatrix;

Vector B_recoverue, X_recoverue;
Vector vm_true(pfespace->GetTrueVSize());
Vector rhs_recoverue(pfespace->GetTrueVSize());

// 初始化求解器
if (!use_petsc)
{
    precond_recoverue_hypre = new HypreBoomerAMG;
    pcg_recoverue_hypre = new HyprePCG(MPI_COMM_WORLD);
    
    // 形成系统矩阵
    a_pblf_recoverue->FormSystemMatrix(ess_tdof_list, A_recoverue_hypre);
    temp_form->FormSystemMatrix(ess_tdof_list, A_temp_hypre);
    
    precond_recoverue_hypre->SetPrintLevel(0);
    pcg_recoverue_hypre->SetPreconditioner(*precond_recoverue_hypre);
    pcg_recoverue_hypre->SetOperator(A_recoverue_hypre);
    pcg_recoverue_hypre->SetTol(1e-6);
    pcg_recoverue_hypre->SetMaxIter(1000);
    pcg_recoverue_hypre->SetPrintLevel(1);
}
else
{
    // 形成系统矩阵（PETSc版本）
    DebugMatrixStage("recoverue A2 Hypre FormSystemMatrix begin");
    a_pblf_recoverue->FormSystemMatrix(ess_tdof_list, A_recoverue_hypre);
    DebugMatrixStage("recoverue A2 Hypre FormSystemMatrix done");
    DebugMatrixStage("recoverue A2 safe PETSc conversion begin");
    ConvertHypreToPetscAIJSafe(A_recoverue_hypre, *A_recoverue_petsc,
                               "recoverue_A2", my_rank);
    DebugMatrixStage("recoverue A2 safe PETSc conversion done");
    DebugMatrixStage("recoverue temp Hypre FormSystemMatrix begin");
    temp_form->FormSystemMatrix(ess_tdof_list, A_temp_hypre);
    DebugMatrixStage("recoverue temp Hypre FormSystemMatrix done");
    DebugMatrixStage("recoverue temp safe PETSc conversion begin");
    ConvertHypreToPetscAIJSafe(A_temp_hypre, *A_temp_petsc,
                               "recoverue_temp", my_rank);
    DebugMatrixStage("recoverue temp safe PETSc conversion done");

    if (ess_tdof_list.Size() == 0)
    {
        DebugMatrixStage("recoverue MatNullSpace attach begin");
        MatNullSpace nsp = NULL;
        PetscErrorCode ierr = MatNullSpaceCreate(MPI_COMM_WORLD, PETSC_TRUE,
                                                 0, NULL, &nsp);
        MFEM_VERIFY(ierr == 0, "MatNullSpaceCreate failed for recoverue A2.");
        Mat A2_petsc = *A_recoverue_petsc;
        ierr = MatSetNullSpace(A2_petsc, nsp);
        MFEM_VERIFY(ierr == 0, "MatSetNullSpace failed for recoverue A2.");
        ierr = MatSetTransposeNullSpace(A2_petsc, nsp);
        MFEM_VERIFY(ierr == 0, "MatSetTransposeNullSpace failed for recoverue A2.");
        ierr = MatNullSpaceDestroy(&nsp);
        MFEM_VERIFY(ierr == 0, "MatNullSpaceDestroy failed for recoverue A2.");
        if (my_rank == 0)
        {
            std::cout << "[recoverue] MatSetNullSpace(span{1}) attached to A2."
                      << std::endl;
        }
        DebugMatrixStage("recoverue MatNullSpace attach done");
    }
    
    pcg_recoverue_petsc = new PetscPCGSolver(MPI_COMM_WORLD, "recoverue_", true);
    // 如果PetscPCGSolver需要SetOperator，添加：
    DebugMatrixStage("recoverue PETSc SetOperator begin");
    pcg_recoverue_petsc->SetOperator(*A_recoverue_petsc);
    DebugMatrixStage("recoverue PETSc SetOperator done");
    pcg_recoverue_petsc->Customize(true);
    AttachScaledASM(pcg_recoverue_petsc, MPI_COMM_WORLD,
                    static_cast<int>(asm_overlap),
                    static_cast<int>(asm_icc_levels));
}

X_recoverue.SetSize(pfespace->GetTrueVSize());
X_recoverue = 0.0;  // 初始猜测为零

ParBilinearForm m_form_recoverue(pfespace);
m_form_recoverue.AddDomainIntegrator(new MassIntegrator(one));
m_form_recoverue.Assemble();
m_form_recoverue.Finalize();
HypreParMatrix M_recoverue;
{
    Array<int> empty_ess;
    m_form_recoverue.FormSystemMatrix(empty_ess, M_recoverue);
}

Vector ones_true_recoverue(pfespace->GetTrueVSize());
ones_true_recoverue = 1.0;
Vector M_ones_recoverue(pfespace->GetTrueVSize());
M_recoverue.Mult(ones_true_recoverue, M_ones_recoverue);
double M_total_recoverue =
    InnerProduct(MPI_COMM_WORLD, ones_true_recoverue, M_ones_recoverue);
MFEM_VERIFY(M_total_recoverue > 0.0, "Recoverue mass matrix has zero total mass.");

if (my_rank == 0)
{
    std::cout << "[recoverue] mass gauge ready, 1^T M 1 = "
              << std::scientific << M_total_recoverue << std::defaultfloat
              << std::endl;
}





        // 5. 创建双线性型和线性型
        ParBilinearForm* a_pblf_torso = new ParBilinearForm(pfespace_torso);

        ConstantCoefficient sigma_torso_coeff(sigma_torso);
        a_pblf_torso->AddDomainIntegrator(new DiffusionIntegrator(sigma_torso_coeff));
        a_pblf_torso->Assemble(use_petsc ? 0 : 1);
        a_pblf_torso->Finalize();

        // 创建线性型
        ParLinearForm* b_plf_torso = new ParLinearForm(pfespace_torso);
                // 为源边界创建边界属性数组
                Array<int> source_bdr_torso(pmesh_torso->bdr_attributes.Max());
                source_bdr_torso = 0;
                source_bdr_torso[BoundaryType::SOURCE-1] = 1;
        // 6. 应用Dirichlet边界条件
        Array<int> ess_bdr_torso(pmesh_torso->bdr_attributes.Max());
        ess_bdr_torso = 0;
        ess_bdr_torso[BoundaryType::DIRICHLET-1] = 1; 
        Array<int> ess_tdof_list_torso;
        pfespace_torso->GetEssentialTrueDofs(ess_bdr_torso, ess_tdof_list_torso);
// 创建包含边界值的系数函数

    if (solve_torso_model && pmesh && pfespace && pfespace_torso) {
        if (my_rank == 0) {
            std::cout << "初始化心脏-躯干交界面传输对象..." << std::endl;
        }
        
        // 创建一次接口映射，之后每个时间步只传递边界值
            interface_transfer = new HeartTorsoBoundaryTransfer(
                pmesh, pfespace, &gf_ue,
                pmesh_torso, pfespace_torso, gf_ue_torso,
                tolerance);
        
        if (my_rank == 0) {
            std::cout << "心脏-躯干交界面传输对象初始化完成。" << std::endl;
        }
    }

//Set matrix and vector for Torso model
HypreBoomerAMG* precond_hypre = nullptr;
HyprePCG* pcg_hypre = nullptr;
HypreParMatrix A_torso_hypre;

PetscPCGSolver* pcg_petsc = nullptr;
PetscParMatrix *A_torso_petsc = new PetscParMatrix;

Vector B_torso, X_torso;

   if (!use_petsc)
   {
        precond_hypre = new HypreBoomerAMG;
        pcg_hypre = new HyprePCG(MPI_COMM_WORLD);    
        //parcsr_A_hypre = parcsr_A_hypre.As<HypreParMatrix>();
        a_pblf_torso->FormLinearSystem(ess_tdof_list_torso, *gf_ue_torso, *b_plf_torso, 
             A_torso_hypre, X_torso, B_torso);

        precond_hypre->SetPrintLevel(0);
        pcg_hypre->SetPreconditioner(*precond_hypre);
        pcg_hypre->SetOperator(A_torso_hypre);
        pcg_hypre->SetTol(1e-4);
        pcg_hypre->SetMaxIter(1000);
        pcg_hypre->SetPrintLevel(1);
   }
   else
   {
        pcg_petsc = new PetscPCGSolver(MPI_COMM_WORLD, "torso_", true);
        pcg_petsc->SetRelTol(1e-4);
        pcg_petsc->SetAbsTol(1e-10);
        pcg_petsc->SetMaxIter(3000);
        pcg_petsc->SetPrintLevel(0);

        b_plf_torso->Assemble();
        a_pblf_torso->FormLinearSystem(ess_tdof_list_torso, *gf_ue_torso, *b_plf_torso,
                A_torso_hypre, X_torso, B_torso);
        ConvertHypreToPetscAIJSafe(A_torso_hypre, *A_torso_petsc,
                                   "torso_A", my_rank);
        pcg_petsc->SetOperator(*A_torso_petsc);
        pcg_petsc->Customize(true);
        AttachScaledASM(pcg_petsc, MPI_COMM_WORLD,
                        static_cast<int>(asm_overlap),
                        static_cast<int>(asm_icc_levels));
   }




    MPI_Barrier(MPI_COMM_WORLD); // Sync before starting overall timer
    t_start = MPI_Wtime();
    t_total_elapsed = t_start; // Store start time here initially
double t_total_start;
double t_total_end;



ParaViewDataCollection paraview_dc("potentials_data", pmesh);
paraview_dc.SetPrefixPath(outputDir);
paraview_dc.RegisterField("Vm", &gf_Vm);
paraview_dc.RegisterField("ue", &gf_ue);
paraview_dc.SetDataFormat(VTKFormat::ASCII);
paraview_dc.SetCycle(0);
paraview_dc.SetTime(0.0);

ParaViewDataCollection paraview_dc_torso("torso_data", pmesh_torso);
paraview_dc_torso.SetPrefixPath(outputDir);
paraview_dc_torso.RegisterField("uT", gf_ue_torso);
paraview_dc_torso.SetDataFormat(VTKFormat::ASCII);
paraview_dc_torso.SetCycle(0);
paraview_dc_torso.SetTime(0.0);

   
   int itime=0;
   while (1)
   {
      // Strong-scaling timing mode: disable all per-step file output and related barriers.
      //if end time, then exit
      if (itime == max_time_steps) { break; }


t_total_start = MPI_Wtime();


t_ionic_start = MPI_Wtime();
      //calculate the ionic contribution.
      if (useNodalIion) {
         reactionWrapper.getVmReadwrite() = actual_Vm; //should be a memcpy
         reactionWrapper.Calc();
      } else {
         rf->Calc(gf_Vm);
      }
 t_ionic_end = MPI_Wtime();
    t_ionic_model_total += (t_ionic_end - t_ionic_start);
      
      //add stimulii
      stims.updateTime(timeline.realTimeFromTimestep(itime));
      
      //compute the Iion and stimulus contribution
      c->Update();
      c->Assemble();

   if (!use_petsc)
   {
      a->FormLinearSystem(ess_tdof_list, gf_Vm, *c, LHS_mat, actual_Vm, actual_b, 1);
    }
    else
    {
        a->FormLinearSystem(ess_tdof_list, gf_Vm, *c, LHS_mat, actual_Vm, actual_b, 1);
    }


      //compute the RHS matrix contribution
      RHS_mat.Mult(actual_Vm, actual_old);
      actual_b += actual_old;

      if (useNodalIion)
      {
         Iion_mat.Mult(reactionWrapper.getIionReadonly(), actual_old);
         actual_b += actual_old;
      }
      //solve the matrix
      t_ksp1_start = MPI_Wtime();
         if (!use_petsc)
   {
      pcg->Mult(actual_b, actual_Vm);
    }
    else
    {
        
        pcg_monodomain_petsc->Mult(actual_b, actual_Vm);
                int current_iterations = pcg_monodomain_petsc->GetNumIterations();
    total_iterations_monodomain += current_iterations;
    solve_count_monodomain++;
    

    }
      t_ksp1_end = MPI_Wtime();
      t_ksp1_total += (t_ksp1_end - t_ksp1_start);

      a->RecoverFEMSolution(actual_Vm, *c, gf_Vm);
      
      // 求解伪双域模型以恢复细胞外电位u_e
      if (solveForUe) {

    gf_Vm.GetTrueDofs(vm_true);
    
    // 计算右侧向量: -∇·(σ_i∇V_m)
    if (!use_petsc) {
        A_temp_hypre.Mult(-1.0, vm_true, 0.0, rhs_recoverue);
    } else {
        A_temp_petsc->Mult(-1.0, vm_true, 0.0, rhs_recoverue);
    }
    
    // 记录求解时间
    t_ksp2_start = MPI_Wtime();

    // 求解线性系统
    if (!use_petsc) {
        pcg_recoverue_hypre->Mult(rhs_recoverue, X_recoverue);
    } else {
        pcg_recoverue_petsc->Mult(rhs_recoverue, X_recoverue);
        int current_iterations = pcg_recoverue_petsc->GetNumIterations();
        total_iterations_recoverue += current_iterations;
        solve_count_recoverue++;
    }
    
    t_ksp2_end = MPI_Wtime();
    t_ksp2_total += (t_ksp2_end - t_ksp2_start);
    
    if (ess_tdof_list.Size() == 0)
    {
        Vector Mu(X_recoverue.Size());
        M_recoverue.Mult(X_recoverue, Mu);
        const double gauge_before =
            InnerProduct(MPI_COMM_WORLD, ones_true_recoverue, Mu);
        X_recoverue.Add(-gauge_before / M_total_recoverue, ones_true_recoverue);
    }
    gf_ue.SetFromTrueDofs(X_recoverue);


      }
      
// 在时间迭代循环中，找到求解伪双域模型的部分后面
if (solve_torso_model  && pmesh_torso && pfespace_torso && gf_ue_torso) {
   
double t_boundary_start_1, t_boundary_end_1, t_boundary_duration_1;
double t_boundary_start_2, t_boundary_end_2, t_boundary_duration_2;
t_boundary_start_1 = MPI_Wtime();

 interface_transfer->TransferValuesToTorsoBoundary();
 t_boundary_end_1 = MPI_Wtime();
t_boundary_duration_1 = t_boundary_end_1 - t_boundary_start_1;

    b_plf_torso->Assemble();

    t_boundary_start_2 = MPI_Wtime();
  // 应用边界条件 - 仅在Dirichlet边界上使用心脏网格的值

 t_boundary_end_2 = MPI_Wtime();
t_boundary_duration_2 = t_boundary_end_2 - t_boundary_start_2;


   if (!use_petsc)
   {
        a_pblf_torso->FormLinearSystem(ess_tdof_list_torso, *gf_ue_torso, *b_plf_torso, 
             A_torso_hypre, X_torso, B_torso);
        t_ksp3_start = MPI_Wtime();
        pcg_hypre->Mult(B_torso, X_torso);
        t_ksp3_end = MPI_Wtime();
        t_ksp3_total += (t_ksp3_end - t_ksp3_start);
   }
   else
   {
      a_pblf_torso->FormLinearSystem(ess_tdof_list_torso, *gf_ue_torso, *b_plf_torso,
                A_torso_hypre, X_torso, B_torso);
      t_ksp3_start = MPI_Wtime();
      pcg_petsc->Mult(B_torso, X_torso);
      t_ksp3_end = MPI_Wtime();
      t_ksp3_total += (t_ksp3_end - t_ksp3_start);
          // 获取PETSc求解器的迭代次数
    int current_iterations = pcg_petsc->GetNumIterations();
    total_iterations_torso += current_iterations;
    solve_count_torso++;
    


   }

        
        // 8. 恢复解
        a_pblf_torso->RecoverFEMSolution(X_torso, *b_plf_torso, *gf_ue_torso);

 t_total_end = MPI_Wtime();
    t_total += (t_total_end - t_total_start);


}



      itime++;
      first=false;
   }



    MPI_Barrier(MPI_COMM_WORLD); // Sync before stopping overall timer
    t_end = MPI_Wtime();
    t_total_elapsed = t_end - t_total_elapsed; // Calculate total duration



    double t_total_max;//, t_problem1_max, t_problem2_max, t_problem3_max;
    double t_ksp1_max, t_ksp2_max, t_ksp3_max, t_ionic_model_max;

    MPI_Reduce(&t_total_elapsed, &t_total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    //MPI_Reduce(&t_problem1_total, &t_problem1_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    //MPI_Reduce(&t_problem2_total, &t_problem2_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    //MPI_Reduce(&t_problem3_total, &t_problem3_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_ksp1_total, &t_ksp1_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_ksp2_total, &t_ksp2_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_ksp3_total, &t_ksp3_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Remember this is placeholder
    MPI_Reduce(&t_ionic_model_total, &t_ionic_model_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        std::cout << "monodomain 平均迭代次数: " << (double)total_iterations_monodomain / solve_count_monodomain << std::endl;
        std::cout << "re ue 平均迭代次数: " << (double)total_iterations_recoverue / solve_count_recoverue << std::endl;
        std::cout << "torso 平均迭代次数: " << (double)total_iterations_torso / solve_count_torso << std::endl;
    }



    if (my_rank == 0) {
        std::cout << "\n--- Parallel Timing Results (Max Across " << num_ranks << " Ranks) ---" << std::endl;
        std::cout << std::fixed << std::setprecision(6); // Format output
        std::cout << "Total Simulation Time: " << t_total_max << " s" << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        //std::cout << "Problem 1 (e.g., Heart): " << t_problem1_max << " s" << std::endl;
        //std::cout << "Problem 2 (e.g., BC Tx): " << t_problem2_max << " s" << std::endl;
        //std::cout << "Problem 3 (e.g., Torso): " << t_problem3_max << " s" << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "KSP 1 (e.g., Heart LinSolv): " << t_ksp1_max << " s" << std::endl;
        std::cout << "KSP 2 (e.g., Torso LinSolv): " << t_ksp2_max << " s" << std::endl;
        std::cout << "KSP 3 (e.g., Other LinSolv): " << t_ksp3_max << " s" << std::endl; // Adjust name
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "Ionic Model Calculation:     " << t_ionic_model_max << " s" << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;

        // Optional: Calculate percentage of total time
        if (t_total_max > 1e-9) { // Avoid division by zero
           double ksp_total_max = t_ksp1_max + t_ksp2_max + t_ksp3_max;
           //double problem_sum_max = t_problem1_max + t_problem2_max + t_problem3_max;
           std::cout << "\n--- Percentage of Total Time (Max) ---" << std::endl;
           //std::cout << "Problem 1: " << (t_problem1_max / t_total_max) * 100.0 << "%" << std::endl;
           //std::cout << "Problem 2: " << (t_problem2_max / t_total_max) * 100.0 << "%" << std::endl;
           //std::cout << "Problem 3: " << (t_problem3_max / t_total_max) * 100.0 << "%" << std::endl;
           std::cout << "Ionic Model: " << (t_ionic_model_max / t_total_max) * 100.0 << "%" << std::endl;
           std::cout << "Total KSP: " << (ksp_total_max / t_total_max) * 100.0 << "%" << std::endl;
           std::cout << "--------------------------------------------------" << std::endl;
           // Note: Sum of percentages might not be 100% due to overhead not timed
           //       and KSP/Ionic times being *part of* Problem times.
           //std::cout << "Debug: Sum of Problems: " << problem_sum_max << " s" << std::endl;
           //std::cout << "Debug: KSP1 within Problem1: " << (t_ksp1_max / t_problem1_max) * 100.0 << "%" << std::endl;
           //std::cout << "Debug: Ionic within Problem1: " << (t_ionic_model_max / t_problem1_max) * 100.0 << "%" << std::endl;
           //std::cout << "Debug: KSP2 within Problem3: " << (t_ksp2_max / t_problem3_max) * 100.0 << "%" << std::endl;

        }
    }






   // 14. Free the used memory.
   delete M_test;
   delete pcg;
   if (pcg_recoverue_petsc) { delete pcg_recoverue_petsc; pcg_recoverue_petsc = nullptr; }
if (pcg_monodomain_petsc) { delete pcg_monodomain_petsc; pcg_monodomain_petsc = nullptr; }
delete pcg_petsc;
pcg_petsc = nullptr;
   if (pcg_recoverue_hypre) { delete pcg_recoverue_hypre; pcg_recoverue_hypre = nullptr; }
   if (precond_recoverue_hypre) { delete precond_recoverue_hypre; precond_recoverue_hypre = nullptr; }
delete LHS_monodomain_petsc;
LHS_monodomain_petsc = nullptr;
delete A_recoverue_petsc;
A_recoverue_petsc = nullptr;
delete A_temp_petsc;
A_temp_petsc = nullptr;
delete A_torso_petsc;
A_torso_petsc = nullptr;
   delete a;
   delete b;
   delete c;
   if (rf) delete rf;
   if (Iion_blf) delete Iion_blf;
   if (a_pblf_recoverue) { delete a_pblf_recoverue; a_pblf_recoverue = nullptr; }
   if (temp_form) { delete temp_form; temp_form = nullptr; }
   delete pfespace;
   //delete fespace;
delete torso_fec;
   if (order > 0) { delete fec; }
   //delete mesh;
   //delete[] pmeshpart;

   if (gf_ue_torso) delete gf_ue_torso;
    if (pfespace_torso) delete pfespace_torso;
    //if (torso_mesh) delete torso_mesh;
    // 主函数末尾资源清理部分应该还需要添加：
//if (transfer) delete transfer;
if (interface_transfer) delete interface_transfer;


if (a_pblf_torso) delete a_pblf_torso;
if (b_plf_torso) delete b_plf_torso;
delete pcg_hypre;
delete precond_hypre;

delete pfespace_vec;

}

MFEMFinalizePetsc();
MPI_Finalize();
   
   return 0;
}
