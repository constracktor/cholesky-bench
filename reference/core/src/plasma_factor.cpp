#include "plasma_factor.hpp"

#include <plasma.h>

#include <stdexcept>
#include <string>

namespace cpu
{

void plasma_cholesky(std::vector<double> &A, int N)
{
    // PLASMA 24.8.7's plasma_desc_*_create routines compute their tile-storage
    // size as int*int and then cast to size_t, so the malloc gets a
    // sign-extended-negative argument and fails for any padded total
    // >= INT32_MAX. With the default nb=256 the triangular padded element
    // count first crosses INT32_MAX at N=65281 (mt=256), so any N>65280 hits
    // the bug. Guard before invoking PLASMA so the multi-line PLASMA ERROR
    // diagnostic does not reach stderr.
    //
    // main.cpp transparently clamps iteration sizes in (65280, 65536] down to
    // 65280, so in practice this guard only fires for N>65536 -- which then
    // becomes a nan cell via main.cpp's per-mode catch handler.
    constexpr int kPlasmaMaxN = 65280;
    if (N > kPlasmaMaxN)
    {
        throw std::runtime_error(
            "plasma_dpotrf: skipped to avoid PLASMA descriptor int32 overflow at N=" + std::to_string(N)
            + " (max supported with default nb=256: " + std::to_string(kPlasmaMaxN) + ")");
    }

    // PLASMA is column-major. Our buffer is row-major and the matrix is
    // symmetric, so we can pass it through unchanged and ask PLASMA to write
    // its result into the upper triangle of its column-major view -- that
    // upper triangle aliases the lower triangle of our row-major view, which
    // is the layout the validator (and the LAPACKE reference path) expects.
    const int info = plasma_dpotrf(PlasmaUpper, N, A.data(), N);
    if (info != 0)
    {
        throw std::runtime_error("plasma_dpotrf failed with info=" + std::to_string(info));
    }
}

}  // end of namespace cpu
