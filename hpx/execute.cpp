#include "functions.hpp"
#include "tile_generation.hpp"

#include <hpx/hpx_main.hpp>
#include <iostream>
#include <vector>

// bool are_identical(const std::vector<std::vector<double>> &A,
//                    const std::vector<std::vector<double>> &B,
//                    double tol = 1e-14)
// {
//     if (A.size() != B.size())
//     {
//         std::cout << "Size mismatch: rows " << A.size() << " vs " << B.size() << std::endl;
//         return false;
//     }
//
//     for (std::size_t i = 0; i < A.size(); ++i)
//     {
//         if (A[i].size() != B[i].size())
//         {
//             std::cout << "Size mismatch at row " << i << ": cols " << A[i].size() << " vs " << B[i].size() << std::endl;
//             return false;
//         }
//
//         for (std::size_t j = 0; j < A[i].size(); ++j)
//         {
//             double diff = std::abs(A[i][j] - B[i][j]);
//             if (diff > tol)
//             {
//                 std::cout << "Mismatch at (" << i << "," << j << ")  " << "cpu=" << B[i][j] << " gpu=" << A[i][j]
//                           << " diff=" << diff << std::endl;
//                 return false;
//             }
//         }
//     }
//
//     return true;
// }

int main(int argc, char *argv[])
{
    ///////////////////////////////////////////////////////////////////////////
    // cmdline arguments
    using namespace hpx::program_options;
    options_description opts;
    opts.add_options()
        ("loop", value<std::size_t>()->default_value(1), "Number of repititions")
        ("size_start", value<std::size_t>()->default_value(32), "Start problem size")
        ("size_stop", value<std::size_t>()->default_value(128), "Stop problem size")
        ("tiles_start", value<std::size_t>()->default_value(16), "Start tiles per dimension")
        ("tiles_stop", value<std::size_t>()->default_value(32), "Stop tiles per dimension");
    variables_map vm;
    store(parse_command_line(argc, argv, opts), vm);
    notify(vm);
    ///////////////////////////////////////////////////////////////////////////
    // configuration
    const std::size_t LOOP = vm["loop"].as<std::size_t>();

    const std::size_t START_SIZE = vm["size_start"].as<std::size_t>();
    const std::size_t STOP_SIZE = vm["size_stop"].as<std::size_t>();
    const std::size_t STEP_SIZE = 2;

    const std::size_t START_TILES = vm["tiles_start"].as<std::size_t>();
    const std::size_t STOP_TILES = vm["tiles_stop"].as<std::size_t>();
    const std::size_t STEP_TILES = 2;

    bool HEADER_FLAG = true;
    for (std::size_t n_tiles = START_TILES; n_tiles <= STOP_TILES; n_tiles = n_tiles * STEP_TILES)
    {
        for (std::size_t size = START_SIZE; size <= STOP_SIZE; size = size * STEP_SIZE)
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
