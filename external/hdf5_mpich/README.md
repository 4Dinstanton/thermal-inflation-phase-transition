# Parallel HDF5 built against MPICH

Homebrew’s `hdf5` bottle is **serial** (no `H5Pset_fapl_mpio`).  
Homebrew’s `hdf5-mpi` is built against **OpenMPI**, which we do not mix with MPICH on this Mac.

CosmoLattice MPI + HDF5 snapshots need this tree.

Rebuild (after `brew link mpich` and `brew install libaec`):

```bash
REPO=$PWD/..   # PhaseTransition/external
PREFIX=$PWD
SRC=$REPO/hdf5_mpich_src
MPICH=/opt/homebrew/opt/mpich
AEC=/opt/homebrew/opt/libaec
mkdir -p "$SRC" && cd "$SRC"
curl -L -o hdf5-2.2.0-src.tar.gz https://github.com/HDFGroup/hdf5/archive/refs/tags/2.2.0.tar.gz
tar xf hdf5-2.2.0-src.tar.gz && cd hdf5-2.2.0
mkdir -p build_mpi && cd build_mpi
export PATH="$MPICH/bin:$PATH" CC=mpicc CXX=mpicxx
cmake .. \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DHDF5_ENABLE_PARALLEL=ON \
  -DHDF5_BUILD_FORTRAN=OFF -DHDF5_BUILD_JAVA=OFF -DHDF5_BUILD_CPP_LIB=OFF \
  -DHDF5_BUILD_HL_LIB=ON -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
  -DHDF5_BUILD_EXAMPLES=OFF \
  -DMPI_C_COMPILER="$MPICH/bin/mpicc" \
  -DMPI_CXX_COMPILER="$MPICH/bin/mpicxx" \
  -DCMAKE_PREFIX_PATH="$AEC;$MPICH"
cmake --build . -j8 && cmake --install .
```

Then:

```bash
python Atakan/run_pqera_v1.py --install --build --mpi
```
