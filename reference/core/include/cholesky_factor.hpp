#ifndef CPU_CHOLESKY_FACTOR_H
#define CPU_CHOLESKY_FACTOR_H

#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace cpu
{

/**
 * @brief Reference Cholesky variants.
 *
 *   - reference : single threaded LAPACKE_dpotrf2 call (no tiling; parallelism
 *                 lives entirely inside the threaded BLAS).
 *   - plasma    : single plasma_dpotrf call (PLASMA's own tiled parallel
 *                 Cholesky over the OpenMP runtime).
 */
enum class Variant { reference, plasma };

inline Variant to_variant(const std::string &s)
{
    if (s == "reference")
    {
        return Variant::reference;
    }
    if (s == "plasma")
    {
        return Variant::plasma;
    }
    throw std::invalid_argument("Unknown Variant: " + s);
}

/**
 * @brief Run the requested reference variant on the full row-major N x N
 *        matrix @p A. Factorisation is in place; @p A holds the lower
 *        triangular factor L on return.
 */
void parallel_blas_cholesky(Variant variant, std::vector<double> &A, int N);

}  // end of namespace cpu
#endif  // end of CPU_CHOLESKY_FACTOR_H
