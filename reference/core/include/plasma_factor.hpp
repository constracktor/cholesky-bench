#ifndef CPU_PLASMA_FACTOR_H
#define CPU_PLASMA_FACTOR_H

#pragma once

#include <vector>

namespace cpu
{

/**
 * @brief PLASMA tiled Cholesky on a row-major N x N buffer using the
 *        high-level synchronous API (plasma_dpotrf).
 *
 * PLASMA's high-level API is column-major, so we ask for @c PlasmaUpper:
 * the upper triangle in PLASMA's column-major view aliases the lower
 * triangle in our row-major view, which is the layout the validation
 * routine expects (and which matches the LAPACKE_dpotrf2 reference).
 *
 * Caller is responsible for having invoked plasma_init() at startup; that
 * cost is intentionally amortised over all timed calls and stays out of the
 * timed region.
 *
 * Throws @c std::runtime_error before calling PLASMA when the descriptor
 * size computation inside plasma_desc_*_create() would overflow int32
 * (PLASMA 24.8.7 still does this multiplication in @c int). With the
 * default @c nb=256 the boundary is at @c N=65280; main.cpp transparently
 * clamps any iteration size in @c (65280, 65536] down to 65280, so this
 * guard fires only for @c N>65536 (which then becomes a @c nan cell).
 */
void plasma_cholesky(std::vector<double> &A, int N);

}  // end of namespace cpu
#endif  // end of CPU_PLASMA_FACTOR_H
