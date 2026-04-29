#ifndef CPU_CHOLESKY_FACTOR_H
#define CPU_CHOLESKY_FACTOR_H

#pragma once

#include <vector>

namespace cpu
{

/**
 * @brief Run a single, threaded LAPACKE_dpotrf on the full N x N row-major
 *        matrix @p A. This is the reference (non-tiled) parallel BLAS
 *        Cholesky factorisation that the OpenMP / HPX tiled variants are
 *        benchmarked against.
 */
void parallel_blas_cholesky(std::vector<double> &A, int N);

}  // end of namespace cpu
#endif  // end of CPU_CHOLESKY_FACTOR_H
