#include "scaled_asm.hpp"

#include <iostream>
#include <vector>

using namespace mfem;

namespace
{

// PCSHELL context: apply(r) = D^{-1/2} * M_ASM(BASIC)^{-1} * ( D^{-1/2} * r ).
struct SASMCtx
{
   PC  inner_pc;   // child PCASM_BASIC (overlap + ICC block solves)
   Vec w;          // diagonal weight D^{-1/2} (w_k = 1/sqrt(m_k))
   Vec tmp;        // scratch, matches the KSP vector layout
};

extern "C" PetscErrorCode SASMApply(PC pc, Vec r, Vec z)
{
   SASMCtx *ctx = nullptr;
   PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
   PetscCall(VecPointwiseMult(ctx->tmp, ctx->w, r));   // tmp = D^{-1/2} r
   PetscCall(PCApply(ctx->inner_pc, ctx->tmp, z));     // z   = M_ASM^{-1} tmp
   PetscCall(VecPointwiseMult(z, ctx->w, z));          // z   = D^{-1/2} z  (aliased ok)
   return PETSC_SUCCESS;
}

extern "C" PetscErrorCode SASMDestroy(PC pc)
{
   SASMCtx *ctx = nullptr;
   PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
   if (ctx)
   {
      PetscCall(PCDestroy(&ctx->inner_pc));
      PetscCall(VecDestroy(&ctx->w));
      PetscCall(VecDestroy(&ctx->tmp));
      delete ctx;
   }
   PetscCall(PCShellSetContext(pc, nullptr));
   return PETSC_SUCCESS;
}

} // namespace

void AttachScaledASM(PetscPCGSolver *ksp_wrapper,
                     MPI_Comm comm,
                     int asm_overlap,
                     int icc_levels)
{
   int my_rank = 0;
   MPI_Comm_rank(comm, &my_rank);

   KSP ksp = static_cast<KSP>(*ksp_wrapper);
   Mat A = nullptr;
   PetscErrorCode ierr = KSPGetOperators(ksp, &A, NULL);
   MFEM_VERIFY(ierr == 0, "KSPGetOperators failed in AttachScaledASM.");

   ierr = KSPSetType(ksp, KSPCG);
   MFEM_VERIFY(ierr == 0, "KSPSetType(KSPCG) failed in AttachScaledASM.");

   // 1. Inner PCASM_BASIC, fully self-contained (no options prefix), so that
   //    a later KSPSetFromOptions on the outer KSP cannot touch it.
   PC inner_pc = nullptr;
   ierr = PCCreate(PetscObjectComm((PetscObject) A), &inner_pc);
   MFEM_VERIFY(ierr == 0, "PCCreate(inner ASM) failed.");
   ierr = PCSetType(inner_pc, PCASM);
   MFEM_VERIFY(ierr == 0, "PCSetType(PCASM) failed.");
   ierr = PCASMSetType(inner_pc, PC_ASM_BASIC);
   MFEM_VERIFY(ierr == 0, "PCASMSetType(PC_ASM_BASIC) failed.");
   ierr = PCASMSetOverlap(inner_pc, asm_overlap);
   MFEM_VERIFY(ierr == 0, "PCASMSetOverlap failed.");
   ierr = PCSetOperators(inner_pc, A, A);
   MFEM_VERIFY(ierr == 0, "PCSetOperators(inner ASM) failed.");
   ierr = PCSetUp(inner_pc);
   MFEM_VERIFY(ierr == 0, "PCSetUp(inner ASM) failed.");

   // 2. Overlap multiplicity m_k = number of overlapping subdomains that
   //    contain global DOF k. The with-overlap index sets carry GLOBAL
   //    indices, so ADD_VALUES + assembly accumulates cross-rank counts.
   Vec mult = nullptr;
   ierr = MatCreateVecs(A, &mult, NULL);
   MFEM_VERIFY(ierr == 0, "MatCreateVecs(mult) failed.");
   ierr = VecSet(mult, 0.0);
   MFEM_VERIFY(ierr == 0, "VecSet(mult) failed.");

   PetscInt n_sub = 0;
   IS *is_with_overlap = nullptr;
   IS *is_local_only = nullptr;
   ierr = PCASMGetLocalSubdomains(inner_pc, &n_sub, &is_with_overlap,
                                  &is_local_only);
   MFEM_VERIFY(ierr == 0, "PCASMGetLocalSubdomains failed.");
   for (PetscInt i = 0; i < n_sub; ++i)
   {
      const PetscInt *idx = nullptr;
      PetscInt n = 0;
      ierr = ISGetLocalSize(is_with_overlap[i], &n);
      MFEM_VERIFY(ierr == 0, "ISGetLocalSize failed.");
      ierr = ISGetIndices(is_with_overlap[i], &idx);
      MFEM_VERIFY(ierr == 0, "ISGetIndices failed.");
      std::vector<PetscScalar> ones(static_cast<size_t>(n), 1.0);
      ierr = VecSetValues(mult, n, idx, ones.data(), ADD_VALUES);
      MFEM_VERIFY(ierr == 0, "VecSetValues(mult) failed.");
      ierr = ISRestoreIndices(is_with_overlap[i], &idx);
      MFEM_VERIFY(ierr == 0, "ISRestoreIndices failed.");
   }
   ierr = VecAssemblyBegin(mult);
   MFEM_VERIFY(ierr == 0, "VecAssemblyBegin(mult) failed.");
   ierr = VecAssemblyEnd(mult);
   MFEM_VERIFY(ierr == 0, "VecAssemblyEnd(mult) failed.");

   // 3. Block solves: preonly + ICC(icc_levels).
   KSP *sub_ksps = nullptr;
   PetscInt n_local = 0;
   PetscInt first = 0;
   ierr = PCASMGetSubKSP(inner_pc, &n_local, &first, &sub_ksps);
   MFEM_VERIFY(ierr == 0, "PCASMGetSubKSP failed.");
   for (PetscInt i = 0; i < n_local; ++i)
   {
      ierr = KSPSetType(sub_ksps[i], KSPPREONLY);
      MFEM_VERIFY(ierr == 0, "KSPSetType(KSPPREONLY) failed for sub-KSP.");
      PC sub_pc = nullptr;
      ierr = KSPGetPC(sub_ksps[i], &sub_pc);
      MFEM_VERIFY(ierr == 0, "KSPGetPC failed for sub-KSP.");
      ierr = PCSetType(sub_pc, PCICC);
      MFEM_VERIFY(ierr == 0, "PCSetType(PCICC) failed for sub-PC.");
      ierr = PCFactorSetLevels(sub_pc, icc_levels);
      MFEM_VERIFY(ierr == 0, "PCFactorSetLevels failed for sub-PC.");
   }
   ierr = PCSetUpOnBlocks(inner_pc);
   MFEM_VERIFY(ierr == 0, "PCSetUpOnBlocks(inner ASM) failed.");

   // 4. Diagonal weight w = D^{-1/2} = 1/sqrt(m_k).
   Vec w = nullptr;
   ierr = VecDuplicate(mult, &w);
   MFEM_VERIFY(ierr == 0, "VecDuplicate(w) failed.");
   ierr = VecCopy(mult, w);
   MFEM_VERIFY(ierr == 0, "VecCopy(mult->w) failed.");
   ierr = VecReciprocal(w);     // 1/m_k
   MFEM_VERIFY(ierr == 0, "VecReciprocal(w) failed.");
   ierr = VecSqrtAbs(w);        // 1/sqrt(m_k)
   MFEM_VERIFY(ierr == 0, "VecSqrtAbs(w) failed.");

   PetscReal mmin = 0.0;
   PetscReal mmax = 0.0;
   VecMin(mult, NULL, &mmin);
   VecMax(mult, NULL, &mmax);
   ierr = VecDestroy(&mult);
   MFEM_VERIFY(ierr == 0, "VecDestroy(mult) failed.");

   // 5. Outer PC = PCSHELL that wraps the inner ASM with the D^{-1/2} scaling.
   PC pc_outer = nullptr;
   ierr = KSPGetPC(ksp, &pc_outer);
   MFEM_VERIFY(ierr == 0, "KSPGetPC(outer) failed.");
   ierr = PCSetType(pc_outer, PCSHELL);
   MFEM_VERIFY(ierr == 0, "PCSetType(PCSHELL) failed.");

   SASMCtx *ctx = new SASMCtx;
   ctx->inner_pc = inner_pc;
   ctx->w = w;
   ierr = MatCreateVecs(A, &ctx->tmp, NULL);
   MFEM_VERIFY(ierr == 0, "MatCreateVecs(tmp) failed.");

   ierr = PCShellSetContext(pc_outer, ctx);
   MFEM_VERIFY(ierr == 0, "PCShellSetContext failed.");
   ierr = PCShellSetApply(pc_outer, SASMApply);
   MFEM_VERIFY(ierr == 0, "PCShellSetApply failed.");
   ierr = PCShellSetDestroy(pc_outer, SASMDestroy);
   MFEM_VERIFY(ierr == 0, "PCShellSetDestroy failed.");
   ierr = PCShellSetName(pc_outer, "scaled_ASM_Dm1o2");
   MFEM_VERIFY(ierr == 0, "PCShellSetName failed.");
   ierr = PCSetUp(pc_outer);
   MFEM_VERIFY(ierr == 0, "PCSetUp(outer shell) failed.");

   if (my_rank == 0)
   {
      std::cout << "[sASM] PCSHELL installed: D^-1/2 M_ASM(BASIC, overlap = "
                << asm_overlap << ", ICC levels = " << icc_levels
                << ") D^-1/2; multiplicity range [" << static_cast<int>(mmin)
                << ", " << static_cast<int>(mmax) << "]" << std::endl;
   }
}
