#include "cholesky_factor.hpp"

#include "adapter_cblas_fp64.hpp"

namespace cpu
{

void parallel_blas_cholesky(std::vector<double> &A, int N)
{
    // The whole factorisation is one threaded LAPACKE call; the BLAS library
    // takes care of dispatching work across the available threads.
    potrf(A, N);
}

}  // end of namespace cpu
