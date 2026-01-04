# Comparison of a tiled Cholesky decomposition using HPX and OpenMP

This reposity contains different implementations of a tiled Cholesky decomposition using the two framework OpenMP and HPX.

For OpenMP, four variants are implemented:
    - Naive parallel for loop
    - Collapsed parallel for loop
    - Synchronous tasking (OpenMP 3.1)
    - Asynchronous tasking (OpenMP 4.0/5.0)

For HPX, nine variants are implemented:
    - Naive parallel for loop
    - Collapsed parallel for loop
    - Synchronous tasking (futures, references, values)
    - Asynchronous tasking (futures, references, values, mutable)

## How to compile

In the `hpx` and `openmp` directory run `./compile.sh`.

## How to run

For HPX: `./build/cholesky_hpx --hpx:threads=128 --loop=1 --size_start=1024 --size_stop=65536 --tiles_start=64 --tiles_stop=64`

For OpenMP: `OMP_NUM_THREADS=128 OMP_PROC_BIND=close OMP_PLACES=cores ./build/cholesky_openmp --loop 1 --size_start 1024 --size_stop 65536 --tiles_start 64 --tiles_stop 64`
