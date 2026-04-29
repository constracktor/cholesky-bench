#include "plasma_factor.hpp"

#include <plasma.h>

#include <stdexcept>
#include <string>

namespace cpu
{

void plasma_cholesky(std::vector<double> &A, int N)
{
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
