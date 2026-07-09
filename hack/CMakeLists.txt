
blt_add_executable(
  NAME femheart
  INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCES
     femheart.cpp
     MatrixElementPiecewiseCoefficient.cpp
     util.cpp
     cardiac_coefficients.cpp
     DEPENDS_ON mfem ode_gpu_aware simUtil mpi openmp
  )

# One-time offline converter: serial mesh + fibers -> parallel VisIt collection
# (parData/par-data-NNNNNN) that femheart loads per-rank.  This is the only
# place the serial mesh is replicated.
blt_add_executable(
  NAME write_parmesh
  INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCES
     write_parmesh.cpp
     util.cpp
     DEPENDS_ON mfem simUtil mpi
  )
