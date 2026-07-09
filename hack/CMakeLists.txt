
blt_add_executable(
  NAME femheart
  INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCES
     femheart.cpp
     scaled_asm.cpp
     MatrixElementPiecewiseCoefficient.cpp
     util.cpp
     cardiac_coefficients.cpp
     torsoSolver.cpp
     DEPENDS_ON mfem ode_gpu_aware simUtil mpi openmp
  )
