## Matrix-free solvers (Deal.II)

This repository collects the lab codes and scaling experiments for matrix-based and matrix-free ADR problems (steady and time-dependent). Each lab folder is self contained and builds its own executables.

### Repository layout and hygiene
- All C++ sources live in the respective `*/src` folders (keep new code there). The top-level `src` is available for shared code if needed.
- Do not commit binaries or build artifacts.
- Meshes should be generated otherwise you need to modify the code
### Prerequisites
The project targets Deal.II with MPI, p4est, Trilinos, HDF5, and METIS.

1) If your environment already provides Deal.II:
```bash
module load gcc-glibc dealii          # or the equivalent modules on your system
```

2) If you are on a cluster without Deal.II modules, start from:
```bash
module purge
module load gcc/11.2.0
module load openmpi/4.1.2-gcc11.2.0
module load mkl/2022.1.0
module load cmake/3.23.1
module load boost/1.77.0   # sets BOOST_ROOT
```
If you need to build the full stack yourself, the script below builds METIS, ParMETIS, p4est, HDF5, Trilinos, and Deal.II into `$WORK/libs/install` (adjust `CORES_*` for your memory budget):
```bash
#!/bin/bash
set -e

BASE_DIR="$WORK/libs"
SRC_DIR="$BASE_DIR/src"
BUILD_DIR="$BASE_DIR/build"
INSTALL_DIR="$BASE_DIR/install"
CORES_FAST=20
CORES_SLOW=4
VER_METIS="v5.1.0"
VER_PARMETIS="v4.0.3"
VER_P4EST="2.3.2"
VER_HDF5="1.12.0"
VER_TRILINOS="13.4.1"
VER_DEALII="v9.5.1"

mkdir -p "$SRC_DIR" "$BUILD_DIR" "$INSTALL_DIR"

# METIS
cd "$SRC_DIR"
if [ ! -d METIS-5.1.0 ]; then
  wget https://github.com/KarypisLab/METIS/archive/refs/tags/$VER_METIS.tar.gz -O metis.tar.gz
  tar xzf metis.tar.gz
fi
cd METIS-5.1.0
make config shared=1 prefix="$INSTALL_DIR/metis-5.1.0" cc=gcc gklib_cflags="-fPIC"
make -j $CORES_FAST install

# ParMETIS
cd "$SRC_DIR"
if [ ! -d ParMETIS-4.0.3 ]; then
  wget https://github.com/KarypisLab/ParMETIS/archive/refs/tags/$VER_PARMETIS.tar.gz -O parmetis.tar.gz
  tar xzf parmetis.tar.gz
fi
cd ParMETIS-4.0.3
make config shared=1 prefix="$INSTALL_DIR/parmetis-4.0.3" cc=mpicc cxx=mpicxx metispath="$INSTALL_DIR/metis-5.1.0" gklib_cflags="-fPIC"
make -j $CORES_FAST install

# p4est
cd "$SRC_DIR"
if [ ! -d p4est-$VER_P4EST ]; then
  wget https://p4est.github.io/release/p4est-$VER_P4EST.tar.gz
  tar xzf p4est-$VER_P4EST.tar.gz
fi
mkdir -p "$BUILD_DIR/p4est" && cd "$BUILD_DIR/p4est"
"$SRC_DIR/p4est-$VER_P4EST/configure" --prefix="$INSTALL_DIR/p4est-$VER_P4EST" --enable-mpi --enable-shared --disable-vtk-binary CFLAGS="-O3 -fPIC" CC=mpicc CXX=mpicxx FC=mpif90 F77=mpif77 BLAS_LIBS="-lmkl_rt" LDFLAGS="-L$MKLROOT/lib/intel64"
make -j $CORES_FAST install
rm -rf "$BUILD_DIR/p4est"

# HDF5
cd "$SRC_DIR"
if [ ! -d hdf5-$VER_HDF5 ]; then
  wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-$VER_HDF5/src/hdf5-$VER_HDF5.tar.gz
  tar xzf hdf5-$VER_HDF5.tar.gz
fi
mkdir -p "$BUILD_DIR/hdf5" && cd "$BUILD_DIR/hdf5"
"$SRC_DIR/hdf5-$VER_HDF5/configure" --prefix="$INSTALL_DIR/hdf5-$VER_HDF5" --enable-parallel --enable-shared CC=mpicc CXX=mpicxx FC=mpif90
make -j $CORES_FAST install
rm -rf "$BUILD_DIR/hdf5"

# Trilinos
cd "$SRC_DIR"
if [ ! -d trilinos-$VER_TRILINOS ]; then
  git clone --branch trilinos-release-13-4-1 https://github.com/trilinos/Trilinos.git trilinos-$VER_TRILINOS
fi
mkdir -p "$BUILD_DIR/trilinos" && cd "$BUILD_DIR/trilinos"
cmake -D CMAKE_INSTALL_PREFIX="$INSTALL_DIR/trilinos-$VER_TRILINOS" -D CMAKE_C_COMPILER=mpicc -D CMAKE_CXX_COMPILER=mpicxx -D CMAKE_Fortran_COMPILER=mpif90 -D CMAKE_BUILD_TYPE=Release -D BUILD_SHARED_LIBS=ON -D CMAKE_CXX_FLAGS="-O3 -march=native -fPIC" -D CMAKE_C_FLAGS="-O3 -march=native -fPIC" -D CMAKE_Fortran_FLAGS="-O3 -march=native -fPIC" -D TPL_ENABLE_MPI=ON -D TPL_ENABLE_BLAS=ON -D TPL_BLAS_LIBRARIES="-lmkl_rt" -D TPL_ENABLE_LAPACK=ON -D TPL_LAPACK_LIBRARIES="-lmkl_rt" -D Trilinos_ENABLE_Amesos=ON -D Trilinos_ENABLE_AztecOO=ON -D Trilinos_ENABLE_Epetra=ON -D Trilinos_ENABLE_EpetraExt=ON -D Trilinos_ENABLE_Ifpack=ON -D Trilinos_ENABLE_ML=ON -D Trilinos_ENABLE_MueLu=ON -D Trilinos_ENABLE_Rol=ON -D Trilinos_ENABLE_Sacado=ON -D Trilinos_ENABLE_Teuchos=ON -D Trilinos_ENABLE_Zoltan=ON -D Trilinos_ENABLE_TESTS=OFF "$SRC_DIR/trilinos-$VER_TRILINOS"
make -j $CORES_FAST install
rm -rf "$BUILD_DIR/trilinos"

# Deal.II
cd "$SRC_DIR"
if [ ! -d dealii ]; then
  git clone --branch $VER_DEALII https://github.com/dealii/dealii.git dealii
fi
mkdir -p "$BUILD_DIR/dealii" && cd "$BUILD_DIR/dealii"
cmake -D CMAKE_INSTALL_PREFIX="$INSTALL_DIR/dealii-$VER_DEALII" -D DEAL_II_WITH_MPI=ON -D CMAKE_CXX_FLAGS="-O3 -march=native" -D DEAL_II_WITH_BOOST=ON -D BOOST_DIR="$BOOST_ROOT" -D DEAL_II_WITH_TRILINOS=ON -D TRILINOS_DIR="$INSTALL_DIR/trilinos-$VER_TRILINOS" -D DEAL_II_WITH_P4EST=ON -D P4EST_DIR="$INSTALL_DIR/p4est-$VER_P4EST" -D DEAL_II_WITH_HDF5=ON -D HDF5_DIR="$INSTALL_DIR/hdf5-$VER_HDF5" -D DEAL_II_WITH_METIS=ON -D METIS_DIR="$INSTALL_DIR/parmetis-4.0.3" "$SRC_DIR/dealii"
make -j $CORES_SLOW install
rm -rf "$BUILD_DIR/dealii"
```

### Building the labs
Each lab has its own `CMakeLists.txt`; build inside the target folder:
```bash
cd nmpde-matrix-free-solvers/lab-05           # or lab-05_mf, lab-05_adr, dt_poisson, dt_poisson_mf, lab-05_mf_adr
mkdir -p build && cd build

# If a Deal.II module provides DEAL_II_DIR, a plain cmake .. is enough.
cmake -DDEAL_II_DIR=$WORK/libs/install/dealii-v9.5.1 ..
make -j 20
```
Main executables:
- `lab-05`: `lab-05_parallel`, `poisson_mat_strong`, `lab-05_weak_mat`
- `lab-05_mf`: `lab-05_parallel`, `poisson_mf_strong`, `lab-05_weak_mf`
- `lab-05_adr`: `lab-05_parallel`, `poisson_mat_strong`, `lab-05_weak_mat` (ADR variant)
- `lab-05_mf_adr`: matrix-free ADR analogues
- `dt_poisson`: `lab-06`, `sparse_time_convergence`
- `dt_poisson_mf`: `lab-06`, `heat_mf_scaling`

### Running and scaling
- Typical MPI run: `mpirun -np 4 ./lab-05_parallel`.
- Strong-scaling scripts live in `strong_scaling_steady/srun_strong_scaling.sh` and `strong_scaling_time_dependent/run_strong_scaling.sh`.
- Weak-scaling scripts live in `weak_scaling_steady/run_weak_scaling.sh` and `weak_scaling_time_depedent/run_weak_scaling.sh`.
- On Slurm, you can confirm node distribution with `srun -n 128 --distribution=block:block,Pack hostname | sort | uniq -c`.

### Notes
- Keep new source files inside `src` folders and avoid tracking generated meshes or binaries.
- When adding meshes, prefer committing only the `*.geo` generator and document the `gmsh` command used to produce the `.msh`.
