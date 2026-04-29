#include "adapter_cblas_fp64.hpp"

#ifdef ENABLE_MKL
// MKL CBLAS / LAPACKE
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

void potrf(vector &A, const int N)
{
    // Single threaded LAPACKE call on the full matrix. dpotrf2 is the
    // recursive variant, which is what the OpenMP / HPX variants use on
    // their diagonal tiles, so picking it here keeps the underlying kernel
    // identical and isolates the parallelism source as the only difference.
    LAPACKE_dpotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
}
