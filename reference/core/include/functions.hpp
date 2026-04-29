#ifndef CPU_FUNCTIONS_H
#define CPU_FUNCTIONS_H

#pragma once

#include <cstddef>
#include <vector>

namespace cpu
{

/**
 * @brief Time a single threaded LAPACKE_dpotrf call on the @p A buffer
 *        (row-major, N x N). The buffer is factorised in place.
 *
 * @param A row-major matrix; on return contains the lower-triangular factor L
 * @param N matrix dimension
 * @return wall-clock elapsed time in seconds
 */
double cholesky(std::vector<double> &A, std::size_t N);

}  // namespace cpu
#endif  // end of CPU_FUNCTIONS_H
