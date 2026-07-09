# Spack dev-build record

Date: 2026-07-09
Source tree: `/home/thma/git-tianhaomahpc-ops/cardioid`

Concretize:

```sh
spack spec cardioid@elecfem+mfem ^mfem@4.7.0 ^petsc target=x86_64_v3
```

Configure/install command:

```sh
spack --debug dev-build -u cmake cardioid@elecfem+mfem ^mfem@4.7.0 ^petsc target=x86_64_v3
```

Then build from the generated build directory:

```sh
cd build
make
```
