# FFTW built against MPICH

Homebrew’s `fftw` bottle links its MPI symbols to **OpenMPI**. CosmoLattice’s
MPI binary must use the same MPI as `mpirun`. On macOS where OpenMPI 5’s
`mpirun` segfaults, use **MPICH** and this tree.

Rebuild (after `brew link mpich`):

```bash
SRC=$PWD/../fftw_mpich_src
PREFIX=$PWD
mkdir -p "$SRC" && cd "$SRC"
curl -L -o fftw-3.3.11.tar.gz https://fftw.org/fftw-3.3.11.tar.gz
tar xf fftw-3.3.11.tar.gz && cd fftw-3.3.11
export CC=mpicc CXX=mpicxx MPICC=mpicc
./configure --prefix="$PREFIX" --enable-shared --enable-threads --enable-mpi --disable-fortran
make -j && make install
```

Then:

```bash
python simulation/run_cosmolattice.py --build --mpi --dry_run
```
