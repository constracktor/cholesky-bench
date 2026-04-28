# Cholesky-Bench

Cholesky-Bench benchmarks right-looking tiled Cholesky factorization from fork-join to asynchronous tasks across several parallelism models, currently comparing OpenMP and HPX implementations side by side.

## Variants

### OpenMP (`openmp/`)

| Mode | Description |
|------|-------------|
| `for_naive` | Naive parallel for loop with dynamic schedule for trailing-update |
| `for_collapse` | Collapsed parallel for loop (optional dynamic schedule for trailing-update with LLVM) |
| `task_naive` | OpenMP tasking without dependencies (OpenMP 3.0) |
| `task_depend` | OpenMP tasking with `depend` clauses (OpenMP 4.0) |
| `task_prio` | OpenMP tasking with `priority` clauses (OpenMP 4.5) (requires `OMP_MAX_TASK_PRIORITY` to be set at runtime) |

### HPX (`hpx/`)

| Mode | Description |
|------|-------------|
| `async_future` | Fully asynchronous tasking with dataflow using `hpx::shared_future<vector>` |
| `sync_future` | Manually synchronized tasking with dataflow using `hpx::shared_future<vector>` |
| `loop_one` | Naive fork-join with dynamic schedule for trailing-update |
| `loop_two` | Collapsed fork-join with dynamic schedule for trailing-update |
| `async_void` |  Fully asynchronous tasking with dataflow using `hpx::shared_future<void>` |

## Dependencies

Both implementations share the same sequential BLAS backend and are built with CMake (≥ 3.23) and C++20.

| Dependency | OpenMP | HPX |
|---|---|---|
| OpenBLAS 0.3.28 | ✓ (default) | ✓ (default) |
| Intel oneMKL | optional (`ENABLE_MKL=ON`) | optional (`ENABLE_MKL=ON`) |
| HPX 1.11.0 + jemalloc | — | ✓ |
| GCC 14.2.0 | ✓ | ✓ |
| LLVM/Clang 22.1.2 | optional | — |

Dependencies are managed via [Spack](https://spack.io/). The compile scripts auto-detect the host system and load the correct Spack environment.

## Build

From within the `openmp/` or `hpx/` directory, run:

```bash
./compile.sh [gcc|llvm]   # OpenMP: gcc (default) or llvm
./compile.sh              # HPX: always gcc
```

The script clears and recreates the `build/` directory, then runs CMake in Release mode followed by a parallel make.

### CMake options

These can be set as environment variables before calling `compile.sh`:

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_VALIDATION` | `OFF` | After each factorization, compute the relative residual ‖A − LL^T‖_F / ‖A‖_F and warn if it exceeds 1e-10. Mutually exclusive with `DISABLE_COMPUTATION`. |
| `DISABLE_COMPUTATION` | `OFF` | Replace all BLAS/tile-generation calls with no-ops. The task graph and loops remain intact, so scheduling overhead can be measured in isolation. |
| `ENABLE_DYNAMIC_SCHEDULE` | `OFF` | *(OpenMP only)* Use `schedule(dynamic,1)` on the trailing-update worksharing loops in `for_collapse`. Requires the LLVM toolchain; rejected at compile time with GCC. |
| `ENABLE_MKL` | `OFF` | Link against Intel oneMKL instead of OpenBLAS. |

**Examples:**

```bash
# OpenMP: GCC build with validation enabled
ENABLE_VALIDATION=ON ./compile.sh gcc

# OpenMP: LLVM build with dynamic scheduling
ENABLE_DYNAMIC_SCHEDULE=ON ./compile.sh llvm

# HPX: measure pure scheduling overhead
DISABLE_COMPUTATION=ON ./compile.sh
```

## Run

### Directly

```bash
# OpenMP
OMP_NUM_THREADS=128 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./build/cholesky_openmp \
  --loop 1 --size_start 1024 --size_stop 65536 \
  --tiles_start 64 --tiles_stop 64

# HPX
./build/cholesky_hpx \
  --hpx:threads=128 \
  --loop=1 --size_start=1024 --size_stop=65536 \
  --tiles_start=64 --tiles_stop=64
```

### Via SLURM

Both directories contain a `run.sh` that is a ready-to-submit SLURM batch script (128 CPUs, exclusive node, 144-hour wall time):

```bash
sbatch openmp/run.sh          # gcc runtime (default)
sbatch openmp/run.sh llvm     # llvm runtime
sbatch hpx/run.sh
```

### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--loop` / `--loop=` | 1 | Number of timed repetitions per configuration |
| `--size_start` / `--size_stop` | 32 / 128 | Problem size range (doubled each step) |
| `--tiles_start` / `--tiles_stop` | 16 / 32 | Tile count range (doubled each step) |

## Output

Results are appended to a text file in the working directory:

```
runtimes_openmp_cholesky_<suffix>.txt
runtimes_hpx_cholesky_<suffix>.txt
```

The suffix encodes which dimension is swept: `tile_` if tiles vary, `size_` if size varies, followed by the loop count. The file uses `;`-separated columns:

```
threads;problem_size;tile_size;n_tiles;for_collapse;for_naive;task_naive;task_depend
128;65536;1024;64;3.14;3.21;2.98;2.87
```

The same lines are also printed to stdout.

## Repository structure

```
.
├── openmp/
│   ├── CMakeLists.txt
│   ├── CMakePresets.json
│   ├── compile.sh          # build script (gcc or llvm)
│   ├── run.sh              # SLURM job script
│   ├── main.cpp
│   └── core/
│       ├── include/
│       │   ├── cholesky_factor.hpp
│       │   ├── functions.hpp
│       │   ├── tile_generation.hpp
│       │   ├── validate.hpp
│       │   └── adapter_cblas_fp64.hpp
│       └── src/
│           ├── cholesky_factor.cpp
│           ├── functions.cpp
│           ├── tile_generation.cpp
│           ├── validate.cpp
│           └── adapter_cblas_fp64.cpp
└── hpx/
    ├── CMakeLists.txt
    ├── CMakePresets.json
    ├── compile.sh          # build script (gcc only)
    ├── run.sh              # SLURM job script
    ├── main.cpp
    └── core/
        ├── include/
        │   ├── cholesky_factor.hpp
        │   ├── functions.hpp
        │   ├── tile_generation.hpp
        │   ├── validate.hpp
        │   └── adapter_cblas_fp64.hpp
        └── src/
            ├── cholesky_factor.cpp
            ├── functions.cpp
            ├── tile_generation.cpp
            ├── validate.cpp
            └── adapter_cblas_fp64.cpp
```

## Contributing

We would be happy to expand Cholesky-Bench to additional asynchronous many-task (AMT) runtimes. If you have an implementation you would like to add, feel free to open a pull request.

## How to cite

If you use Cholesky-Bench in your research, please cite:

```bibtex
@misc{coming_soon}
```
