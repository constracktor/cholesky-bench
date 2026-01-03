#include "cholesky_factor.hpp"

#include "adapter_cblas_fp64.hpp"
#include <cmath>
#include <iostream>

namespace cpu
{

void right_looking_cholesky_tiled(Variant variant, Tiled_vector_matrix &tiles)
{
    // Parameters
    int N = std::sqrt(tiles[0].size());
    std::size_t n_tiles = std::sqrt(tiles.size());
    // Variants
    switch (variant)
    {
        case Variant::for_collapse:
            for (std::size_t k = 0; k < n_tiles; ++k)
            {
                // POTRF: Compute Cholesky factor L
                potrf(tiles[k * n_tiles + k], N);

                // TRSM over the panel below k
#pragma omp parallel for schedule(static)
                for (std::size_t m = k + 1; m < n_tiles; ++m)
                {
                    trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                }

                // Trailing matrix update
#pragma omp parallel for collapse(2)
                for (std::size_t m = k + 1; m < n_tiles; ++m)
                {
                    for (std::size_t n = k + 1; n < m + 1; ++n)
                    {
                        if (n == m)
                        {
                            // SYRK: A = A - B * B^T
                            syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                        }
                        else
                        {
                            // GEMM: C = C - A * B^T
                            gemm(tiles[m * n_tiles + k],
                                 tiles[n * n_tiles + k],
                                 tiles[m * n_tiles + n],
                                 N,
                                 N,
                                 N,
                                 Blas_no_trans,
                                 Blas_trans);
                        }
                    }
                }
            }
            break;
            break;
        case Variant::for_naive:
            for (std::size_t k = 0; k < n_tiles; ++k)
            {
                // POTRF: Compute Cholesky factor L
                potrf(tiles[k * n_tiles + k], N);

// TRSM over the panel below k
#pragma omp parallel for schedule(static)
                for (std::size_t m = k + 1; m < n_tiles; ++m)
                {
                    trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                }

// Trailing matrix update
#pragma omp parallel for schedule(static)
                for (std::size_t m = k + 1; m < n_tiles; ++m)
                {
                    // SYRK: A = A - B * B^T
                    syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);

                    for (std::size_t n = k + 1; n < m + 1; ++n)
                    {
                        // GEMM: C = C - A * B^T
                        gemm(tiles[m * n_tiles + k],
                             tiles[n * n_tiles + k],
                             tiles[m * n_tiles + n],
                             N,
                             N,
                             N,
                             Blas_no_trans,
                             Blas_trans);
                    }
                }
            }
            break;
        case Variant::task_naive:
#pragma omp parallel
            {
#pragma omp single
                {
                    for (std::size_t k = 0; k < n_tiles; ++k)
                    {
                        // POTRF: Cholesky on diagonal tile
                        potrf(tiles[k * n_tiles + k], N);

                        // TRSM: Solve tiles below the diagonal
                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
#pragma omp task firstprivate(m)
                            {
                                trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                            }
                        }
#pragma omp taskwait
                        // Trailing matrix update
                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
#pragma omp task firstprivate(m)
                            {
                                // SYRK: Update diagonal tile
                                syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                            }
                            for (std::size_t n = k + 1; n <= m; ++n)
                            {
#pragma omp task firstprivate(m, n)
                                {
                                    // GEMM: Update off-diagonal tile
                                    gemm(tiles[m * n_tiles + k],
                                         tiles[n * n_tiles + k],
                                         tiles[m * n_tiles + n],
                                         N,
                                         N,
                                         N,
                                         Blas_no_trans,
                                         Blas_trans);
                                }
                            }
                        }
#pragma omp taskwait
                    }
                }
            }
            break;
        case Variant::task_ref:
#pragma omp parallel
            {
#pragma omp single
                {
                    for (std::size_t k = 0; k < n_tiles; ++k)
                    {
                        // -------------------------
                        // POTRF on diagonal
                        // -------------------------
                        auto &Akk = tiles[k * n_tiles + k];
#pragma omp task depend(inout : Akk)
                        {
                            potrf(tiles[k * n_tiles + k], N);
                        }

                        // -------------------------
                        // TRSM on panel below diagonal
                        // -------------------------
                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
                            auto &Amk = tiles[m * n_tiles + k];

#pragma omp task depend(in : Akk) depend(inout : Amk)
                            {
                                trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                            }
                        }

                        // -------------------------
                        // Trailing matrix update
                        // -------------------------
                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
                            auto &Amk = tiles[m * n_tiles + k];

                            // SYRK diagonal update
                            auto &Amm = tiles[m * n_tiles + m];
#pragma omp task depend(in : Amk) depend(inout : Amm)
                            {
                                // SYRK: Update diagonal tile
                                syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                            }

                            // GEMM off-diagonal updates
                            for (std::size_t n = k + 1; n < m; ++n)
                            {
                                auto &Ank = tiles[n * n_tiles + k];
                                auto &Amn = tiles[m * n_tiles + n];

#pragma omp task depend(in : Amk, Ank) depend(inout : Amn)
                                {
                                    // GEMM: Update off-diagonal tile
                                    gemm(tiles[m * n_tiles + k],
                                         tiles[n * n_tiles + k],
                                         tiles[m * n_tiles + n],
                                         N,
                                         N,
                                         N,
                                         Blas_no_trans,
                                         Blas_trans);
                                }
                            }
                        }
                    }
#pragma omp taskwait
                }
            }
            break;
        case Variant::task_pointer:
#pragma omp parallel
            {
#pragma omp single nowait  // nowait is safe and recommended here
                {
                    for (std::size_t k = 0; k < n_tiles; ++k)
                    {
                        // Diagonal tile: POTRF
                        const double *Akk_ptr = tiles[k * n_tiles + k].data();

#pragma omp task depend(inout : *Akk_ptr)
                        {
                            // POTRF: Compute Cholesky factor L
                            potrf(tiles[k * n_tiles + k], N);
                        }

                        // Panel below diagonal: TRSM
                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
                            const double *Amk_ptr = tiles[m * n_tiles + k].data();

#pragma omp task depend(in : *Akk_ptr) depend(inout : *Amk_ptr)
                            {
                                trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                            }
                        }

                        // Trailing submatrix update
                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
                            const double *Amk_ptr = tiles[m * n_tiles + k].data();
                            const double *Amm_ptr = tiles[m * n_tiles + m].data();

// Diagonal update: SYRK
#pragma omp task depend(in : *Amk_ptr) depend(inout : *Amm_ptr)
                            {
                                // SYRK: Update diagonal tile
                                syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                            }

                            // Off-diagonal updates: GEMM
                            for (std::size_t n = k + 1; n < m; ++n)
                            {
                                const double *Ank_ptr = tiles[n * n_tiles + k].data();
                                const double *Amn_ptr = tiles[m * n_tiles + n].data();

#pragma omp task depend(in : *Amk_ptr, *Ank_ptr) depend(inout : *Amn_ptr)
                                {
                                    // GEMM: Update off-diagonal tile
                                    gemm(tiles[m * n_tiles + k],
                                         tiles[n * n_tiles + k],
                                         tiles[m * n_tiles + n],
                                         N,
                                         N,
                                         N,
                                         Blas_no_trans,
                                         Blas_trans);
                                }
                            }
                        }
                    }

#pragma omp taskwait
                }
            }
            break;
        default: std::cout << "Variant not supported.\n"; break;
    }
}

}  // end of namespace cpu
