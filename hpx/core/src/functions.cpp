#include "functions.hpp"

#include "cholesky_factor.hpp"


//#include <hpx/algorithm.hpp>
#include <hpx/future.hpp>



namespace cpu
{

double cholesky_future(
    Tiled_matrix &tiled_matrix,
    std::string variant)
{
    auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(to_variant(variant), tiled_matrix);
    // Synchronize
    hpx::wait_all(tiled_matrix);
    ///////////////////////////////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();
    return (stop - start).count() / 1e9;
}

double cholesky_loop(std::vector<std::vector<double>> &tiled_matrix,
    std::string variant)
{
     auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled_loop(to_variant(variant), tiled_matrix);
    ///////////////////////////////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();
    return (stop - start).count() / 1e9;
}

double cholesky_mutable(std::vector<hpx::shared_future<mutable_tile_data<double>>> &mutable_tiled_matrix)
{
    auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled_mutable(mutable_tiled_matrix);
    // Synchronize
    hpx::wait_all(mutable_tiled_matrix);
    ///////////////////////////////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();
    return (stop - start).count() / 1e9;
}

}  // end of namespace cpu
