#include "functions.hpp"
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
    std::size_t START = 1024;
    std::size_t END = 65'536;
    std::size_t STEP = 2;
    std::size_t LOOP = 1;

    int n_test = 1024;
    const std::size_t N_CORES = 128;
    const std::size_t n_tiles = 32;
    const std::size_t n_reg = 8;

    std::string train_path = "../../../data/data_19/training_input_19.txt";

    for (std::size_t core = 128; core <= N_CORES; core = core * 2)
    {
        for (std::size_t size = START; size <= END; size = size * STEP)
        {
            std::cout << "\n\nProblem size: " << size << std::endl;
            // Compute tile sizes and number of predict tiles
            std::size_t tile_size = size / n_tiles;
            std::cout << "\n\nTile size: " << tile_size << std::endl;

            for (std::size_t l = 0; l < LOOP; l++)
            {


                std::chrono::duration<double> cholesky_async_time;
                std::chrono::duration<double> cholesky_sync_time;
                std::chrono::duration<double> cholesky_ref_time;
                std::chrono::duration<double> cholesky_val_time;
                std::chrono::duration<double> cholesky_mut_time;

                std::string target;

                auto start = std::chrono::high_resolution_clock::now();
                auto end = std::chrono::high_resolution_clock::now();

                    /////////////////////
                    ///// GP

                    // start = std::chrono::high_resolution_clock::now();
                    // std::vector<std::vector<double>> cholesky_cpu_async = gp_cpu.cholesky_async("async_future");
                    // end = std::chrono::high_resolution_clock::now();
                    // cholesky_async_time = end - start;
                    // std::cout << "cpu async future cholesky time: " << cholesky_async_time.count() << std::endl;
                    //
                    // start = std::chrono::high_resolution_clock::now();
                    // cholesky_cpu_async = gp_cpu.cholesky_async("async_ref");
                    // end = std::chrono::high_resolution_clock::now();
                    // cholesky_async_time = end - start;
                    // std::cout << "cpu async ref cholesky time: " << cholesky_async_time.count() << std::endl;
                    //
                    // start = std::chrono::high_resolution_clock::now();
                    // cholesky_cpu_async = gp_cpu.cholesky_async("async_val");
                    // end = std::chrono::high_resolution_clock::now();
                    // cholesky_async_time = end - start;
                    // std::cout << "cpu async val cholesky time: " << cholesky_async_time.count() << std::endl;

                    ////

                    // start = std::chrono::high_resolution_clock::now();
                    // std::vector<std::vector<double>> cholesky_cpu_sync = gp_cpu.cholesky_sync("sync_future");
                    // end = std::chrono::high_resolution_clock::now();
                    // cholesky_sync_time = end - start;
                    // std::cout << "cpu sync future cholesky time: " << cholesky_sync_time.count() << std::endl;
                    //
                    // start = std::chrono::high_resolution_clock::now();
                    // cholesky_cpu_sync = gp_cpu.cholesky_sync("sync_ref");
                    // end = std::chrono::high_resolution_clock::now();
                    // cholesky_sync_time = end - start;
                    // std::cout << "cpu sync ref cholesky time: " << cholesky_sync_time.count() << std::endl;
                    //
                    start = std::chrono::high_resolution_clock::now();
                    auto cholesky_cpu_sync = cpu::cholesky_synchronous("sync_val", 16, 512);
                    end = std::chrono::high_resolution_clock::now();
                    cholesky_sync_time = end - start;
                    std::cout << "cpu sync val cholesky time: " << cholesky_sync_time.count() << std::endl;

                    ////

                    start = std::chrono::high_resolution_clock::now();
                    //std::vector<std::vector<double>> cholesky_cpu_ref = cpu::cholesky_loop("loop_two", 2, size);
                    end = std::chrono::high_resolution_clock::now();
                    cholesky_ref_time = end - start;
                    std::cout << "cpu ref cholesky time: " << cholesky_ref_time.count() << std::endl;

                    // start = std::chrono::high_resolution_clock::now();
                    // std::vector<std::vector<double>> cholesky_cpu_val = cpu::cholesky_loop("loop_two");
                    // end = std::chrono::high_resolution_clock::now();
                    // cholesky_val_time = end - start;
                    // std::cout << "cpu val cholesky time: " << cholesky_val_time.count() << std::endl;

                    // start = std::chrono::high_resolution_clock::now();
                    // std::vector<std::vector<double>> cholesky_cpu_mut = gp_cpu.cholesky_mutable();
                    // end = std::chrono::high_resolution_clock::now();
                    // cholesky_mut_time = end - start;
                    // std::cout << "cpu mut cholesky time: " << cholesky_mut_time.count() << std::endl;
                    // bool ok_sync = are_identical(cholesky_cpu_async, cholesky_cpu_sync);
                    // bool ok_ref = are_identical(cholesky_cpu_async, cholesky_cpu_ref);
                    // bool ok_val = are_identical(cholesky_cpu_async, cholesky_cpu_val);
                    // if (ok_sync && ok_ref && ok_val)
                    //     std::cout << "Cholesky results are IDENTICAL (within tolerance)\n";
                    // else
                    //       std::cout << "Cholesky results differ!\n";
            }
        }
    }

    return 0;
}
