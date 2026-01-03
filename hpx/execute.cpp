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
    std::size_t LOOP = 10;

    const std::size_t N_CORES = 128;
    const std::size_t n_tiles = 32;

    bool HEADER_FLAG = true;
    for (std::size_t core = 128; core <= N_CORES; core = core * 2)
    {
        for (std::size_t size = START; size <= END; size = size * STEP)
        {
            for (std::size_t l = 0; l < LOOP; l++)
            {
                std::size_t tile_size = size / n_tiles;
                // header for output file
                std::string header = "threads;problem_size;tile_size;n_tiles;";
                // runtime config and values
                std::string values = std::to_string(hpx::get_num_worker_threads());
                values += ";" + std::to_string(size);
                values += ";" + std::to_string(size / n_tiles);
                values += ";" + std::to_string(n_tiles);
                ///////////////////////////////////////////////////////////////////////////
                // futurized
                std::vector<std::string> f_modes = {
                    "async_future",
                    "async_ref",
                    "async_val",
                    "sync_future",
                    "sync_ref",
                    "sync_val"
                };
                for (const auto& mode : f_modes) {
                    auto f_tiled_matrix = gen_futurized_tiled_matrix(size, n_tiles);
                    auto cholesky_cpu   = cpu::cholesky_future(f_tiled_matrix, mode);

                    header += ";" + mode;
                    values += ";" + std::to_string(cholesky_cpu);
                }
                ///////////////////////////////////////////////////////////////////////////
                // mutable
                std::string mode = "async_mut";
                auto mut_tiled_matrix = gen_mutable_tiled_matrix(size, n_tiles);
                auto cholesky_cpu = cpu::cholesky_mutable(mut_tiled_matrix);

                header += ";" + mode;
                values += ";" + std::to_string(cholesky_cpu);
                ///////////////////////////////////////////////////////////////////////////
                // loop
                std::vector<std::string> loop_modes = {
                    "loop_one",
                    "loop_two"
                };
                for (const auto& mode : loop_modes) {
                    auto tiled_matrix = gen_tiled_matrix(size, n_tiles);
                    auto cholesky_cpu = cpu::cholesky_loop(tiled_matrix, mode);

                    header += ";" + mode;
                    values += ";" + std::to_string(cholesky_cpu);
                }
                ///////////////////////////////////////////////////////////////////////////
                // write header once
                if (HEADER_FLAG){
                    HEADER_FLAG = false;
                    std::cout << header << std::endl;
                }
                std::cout << values << std::endl;
            }
        }
    }

    return 0;
}
