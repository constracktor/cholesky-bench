#include "functions.hpp"
//#include "tile_generation.hpp"
#include <hpx/hpx_main.hpp>
#include <iostream>
#include <vector>

bool are_identical(const std::vector<std::vector<double>> &A,
                   const std::vector<std::vector<double>> &B,
                   double tol = 1e-14)
{
    if (A.size() != B.size())
    {
        std::cout << "Size mismatch: rows " << A.size() << " vs " << B.size() << std::endl;
        return false;
    }

    for (std::size_t i = 0; i < A.size(); ++i)
    {
        if (A[i].size() != B[i].size())
        {
            std::cout << "Size mismatch at row " << i << ": cols " << A[i].size() << " vs " << B[i].size() << std::endl;
            return false;
        }

        for (std::size_t j = 0; j < A[i].size(); ++j)
        {
            double diff = std::abs(A[i][j] - B[i][j]);
            if (diff > tol)
            {
                std::cout << "Mismatch at (" << i << "," << j << ")  " << "cpu=" << B[i][j] << " gpu=" << A[i][j]
                          << " diff=" << diff << std::endl;
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char *argv[])
{
    /////////////////////
    /////// configuration
    std::size_t START = 256;
    std::size_t END = 65'536;
    std::size_t STEP = 2;
    std::size_t LOOP = 1;

    const std::size_t N_CORES = 128;
    const std::size_t n_tiles = 64;

    for (std::size_t core = 128; core <= N_CORES; core = core * 2)
    {
        for (std::size_t size = START; size <= END; size = size * STEP)
        {
            std::cout << "\n\nProblem size: " << size << std::endl;
            // Compute tile sizes and number of predict tiles
            std::size_t tile_size = size / n_tiles;
            std::cout << "Tile size: " << tile_size << std::endl;

            for (std::size_t l = 0; l < LOOP; l++)
            {
                // async
                auto mut_tiled_matrix = gen_mutable_tiled_matrix(size, n_tiles);
                auto cholesky_cpu = cpu::cholesky_mutable(mut_tiled_matrix);
                std::cout << "cpu mut future: " << cholesky_cpu << std::endl;

                // async
                auto f_tiled_matrix = gen_futurized_tiled_matrix(size, n_tiles);
                cholesky_cpu = cpu::cholesky_future(f_tiled_matrix, "async_future");
                std::cout << "cpu async future: " << cholesky_cpu << std::endl;

                f_tiled_matrix = gen_futurized_tiled_matrix(size, n_tiles);
                cholesky_cpu = cpu::cholesky_future(f_tiled_matrix, "async_ref");
                std::cout << "cpu async ref: " << cholesky_cpu << std::endl;

                f_tiled_matrix = gen_futurized_tiled_matrix(size, n_tiles);
                cholesky_cpu = cpu::cholesky_future(f_tiled_matrix, "async_val");
                std::cout << "cpu async val: " << cholesky_cpu << std::endl;

                // sync
                f_tiled_matrix = gen_futurized_tiled_matrix(size, n_tiles);
                cholesky_cpu = cpu::cholesky_future(f_tiled_matrix, "sync_future");
                std::cout << "cpu sync future: " << cholesky_cpu << std::endl;

                f_tiled_matrix = gen_futurized_tiled_matrix(size, n_tiles);
                cholesky_cpu = cpu::cholesky_future(f_tiled_matrix, "sync_ref");
                std::cout << "cpu sync ref: " << cholesky_cpu << std::endl;

                f_tiled_matrix = gen_futurized_tiled_matrix(size, n_tiles);
                cholesky_cpu = cpu::cholesky_future(f_tiled_matrix, "sync_val");
                std::cout << "cpu sync val: " << cholesky_cpu << std::endl;

                // // loop
                // auto tiled_matrix = gen_tiled_matrix(size, n_tiles);
                // cholesky_cpu = cpu::cholesky_loop(tiled_matrix, "loop_one");
                // std::cout << "cpu loop one: " << cholesky_cpu << std::endl;
                //
                // tiled_matrix = gen_tiled_matrix(size, n_tiles);
                // cholesky_cpu = cpu::cholesky_loop(tiled_matrix, "loop_two");
                // std::cout << "cpu loop two: " << cholesky_cpu << std::endl;
            }
        }
    }

    return 0;
}
