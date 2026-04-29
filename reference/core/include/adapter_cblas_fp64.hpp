#ifndef CPU_ADAPTER_CBLAS_FP64_H
#define CPU_ADAPTER_CBLAS_FP64_H

#pragma once

#include <vector>

using vector = std::vector<double>;

// LAPACK level 3 operations

/**
 * @brief FP64 In-place Cholesky decomposition of A using a single, threaded
 *        LAPACKE_dpotrf call (no tiling). This is the parallel-BLAS reference
 *        implementation that the OpenMP and HPX tiled variants are compared
 *        against.
 *
 * @param A row-major matrix of size N*N to be factorised in place
 * @param N matrix dimension
 */
void potrf(vector &A, const int N);

#endif  // end of CPU_ADAPTER_CBLAS_FP64_H
