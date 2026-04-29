#ifndef CPU_PLASMA_FACTOR_H
#define CPU_PLASMA_FACTOR_H

#pragma once

#include <vector>

namespace cpu
{

/**
 * @brief PLASMA tiled Cholesky on a row-major N x N buffer.
 *
 * PLASMA's high-level API is column-major, so we ask for @c PlasmaUpper:
 * the upper triangle in PLASMA's column-major view aliases the lower
 * triangle in our row-major view, which is the layout the validation
 * routine expects (and which matches the LAPACKE_dpotrf2 reference).
 *
 * Caller is responsible for having invoked plasma_init() at startup; that
 * cost is intentionally amortised over all timed calls and stays out of the
 * timed region.
 */
void plasma_cholesky(std::vector<double> &A, int N);

}  // end of namespace cpu
#endif  // end of CPU_PLASMA_FACTOR_H
