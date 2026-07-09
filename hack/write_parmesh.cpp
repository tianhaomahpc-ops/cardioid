// write_parmesh -- one-time offline converter.
//
// Reads the serial mesh and the serial fiber field named in a femheart object
// file, partitions the mesh across `num_ranks`, and writes a parallel VisIt data
// collection (parData/par-data-NNNNNN) holding, per rank, that rank's ParMesh
// piece plus the distributed "fibers" field.
//
// femheart then loads that collection in parallel -- one rank, one partition,
// never the full serial mesh.  This is the ONLY place the serial mesh is
// replicated; it runs once, offline, and must be run with the SAME rank count
// that the solve will use (the collection is keyed on num_ranks).
//
// Usage:  mpirun -np <N> write_parmesh [femheart.data ...]
//
// NOTE: uncompiled in this environment (no MFEM/PETSc); compile on the cluster.

#include "mfem.hpp"
#include "object.h"
#include "object_cc.hh"
#include "pio.h"
#include "pioFixedRecordHelper.h"
#include "units.h"
#include "util.hpp"

#include <mpi.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;

int main(int argc, char *argv[])
{
   MPI_Init(NULL, NULL);
   int num_ranks, my_rank;
   MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   // Same unit system femheart uses.
   units_internal(1e-3, 1e-9, 1e-3, 1e-3, 1, 1e-9, 1);
   units_external(1e-3, 1e-9, 1e-3, 1e-3, 1, 1e-9, 1);

   std::vector<std::string> objectFilenames;
   if (argc == 1) { objectFilenames.push_back("femheart.data"); }
   for (int i = 1; i < argc; i++) { objectFilenames.push_back(argv[i]); }

   if (my_rank == 0)
   {
      for (size_t i = 0; i < objectFilenames.size(); i++)
      {
         object_compilefile(objectFilenames[i].c_str());
      }
   }
   object_Bcast(0, MPI_COMM_WORLD);

   OBJECT *obj = object_find("femheart", "HEART");
   MFEM_VERIFY(obj != NULL, "HEART object not found.");

   // --- The serial replication lives here, and ONLY here (offline, one-time) ---
   Mesh *mesh = ecg_readMeshptr(obj, "mesh");
   mesh->SetAttributes();
   int *pmeshpart = mesh->GeneratePartitioning(num_ranks);
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, pmeshpart);

   // Distribute the serial fiber GridFunction onto the partitioned mesh.
   std::shared_ptr<GridFunction> flat_fiber_quat;
   ecg_readGF(obj, "fibers", mesh, flat_fiber_quat);
   ParGridFunction *fiber_quat =
      new ParGridFunction(pmesh, flat_fiber_quat.get(), pmeshpart);

   // --- Write the per-rank parallel collection femheart will load ---
   std::ostringstream oss_coll_name;
   oss_coll_name << "par-data-" << std::setfill('0') << std::setw(6) << num_ranks;

   VisItDataCollection visit_dc(MPI_COMM_WORLD, oss_coll_name.str());
   visit_dc.SetPrefixPath("parData/");
   visit_dc.SetMesh(pmesh);
   visit_dc.RegisterField("fibers", fiber_quat);
   visit_dc.SetCycle(0);
   visit_dc.SetTime(0.0);
   visit_dc.Save();

   if (my_rank == 0)
   {
      std::cout << "Wrote parData/" << oss_coll_name.str()
                << " (" << num_ranks << " ranks, "
                << pmesh->GetGlobalNE() << " global elements)." << std::endl;
   }

   delete fiber_quat;
   delete pmesh;
   delete[] pmeshpart;
   delete mesh;

   MPI_Finalize();
   return 0;
}
