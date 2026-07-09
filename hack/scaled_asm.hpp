#ifndef CARDIOID_HACK_SCALED_ASM_HPP
#define CARDIOID_HACK_SCALED_ASM_HPP

#include "mfem.hpp"

#include <mpi.h>
#include <petscksp.h>

// Install a symmetric scaled additive Schwarz (sASM) preconditioner on the
// KSP wrapped by the given MFEM PetscPCGSolver:
//
//     M_sASM^{-1} = D^{-1/2} ( sum_i R_i^T A_i^{-1} R_i ) D^{-1/2}
//
// i.e. D^{-1/2} M_ASM(BASIC)^{-1} D^{-1/2}, where D = diag(m_k) is the overlap
// multiplicity (m_k = number of overlapping subdomains that contain DOF k).
// The two-sided 1/sqrt(m_k) scaling exactly cancels PC_ASM_BASIC's
// over-counting on the overlap while keeping the operator symmetric, so the
// outer CG remains valid. At overlap 0 every DOF has multiplicity 1, so D = I
// and sASM reduces to plain BASIC ASM.
//
// The inner block solves are preonly + ICC(icc_levels).
//
// IMPORTANT: call PetscPCGSolver::Customize(true) on the wrapper BEFORE this
// function so MFEM runs KSPSetFromOptions once; otherwise MFEM's first Mult
// would re-run it and clobber the PCSHELL installed here.
void AttachScaledASM(mfem::PetscPCGSolver *ksp_wrapper,
                     MPI_Comm comm,
                     int asm_overlap,
                     int icc_levels);

#endif
