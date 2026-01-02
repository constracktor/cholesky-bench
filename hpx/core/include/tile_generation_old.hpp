#pragma once 

#include "tile_data.hpp"

#include <hpx/algorithm.hpp>
#include <random>
#include <vector>

std::vector<double> gen_tile(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_tiles)
{
    std::size_t i_global, j_global;
    double random_value;
    // Create random generator
    size_t seed = row * col;
    std::mt19937 generator ( seed );
    std::uniform_real_distribution<double> distribute( 0, 1 );
    // Preallocate required memory
    std::vector<double> tile;
    tile.resize(N * N);
    // Compute entries
    // Check for diagonal tile
    if( row == col )
    {
        for (std::size_t i = 0; i < N; i++)
        {
            i_global = N * row + i;
            for (std::size_t j = 0; j <= i; j++)
            {
                j_global = N * col + j;
                // compute covariance function
                random_value = distribute(generator)  / 1000000;
;
                if (i_global == j_global)
                {
                    // noise variance on diagonal
                    random_value += N * n_tiles;
                }
                tile[i * N + j] = random_value;
                tile[j * N + i] = random_value;
            }
        }
    }
    else
    {
        for (std::size_t i = 0; i < N; i++)
        {
            for (std::size_t j = 0; j < N; j++)
            {
                random_value =distribute(generator) / 1000000;
                tile[i * N + j] = random_value;
            }
        }
    }
    // print tile
    std::cout << "(" << row << "," << col << ")\n";
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            std::cout << tile[i*N +j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";


    return tile;
}

std::vector<std::vector<double>> gen_tiled_matrix(
    std::size_t N,
    std::size_t n_tiles)
{
    // Tiled data structure
    std::vector<std::vector<double>> tiled_matrix;
    // Preallocate memory
    tiled_matrix.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure

    ///////////////////////////////////////////////////////////////////////////
    // Launch synchronous assembly
    hpx::experimental::for_loop(
        hpx::execution::par,
        std::size_t{ 0 },
        std::size_t(n_tiles),
        [&](std::size_t i)
        {
            hpx::experimental::for_loop(
                hpx::execution::par,
                std::size_t{ 0 },
                i + 1,
                [&](std::size_t j)
                {
                                    });
        });

    return tiled_matrix;
}

std::vector<hpx::shared_future<std::vector<double>>> gen_futurized_tiled_matrix(
    std::size_t N,
    std::size_t n_tiles)
{
    // Tiled data structure
    std::vector<hpx::shared_future<std::vector<double>>> tiled_matrix;
    // Preallocate memory
    tiled_matrix.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure

    ///////////////////////////////////////////////////////////////////////////
    // Launch synchronous assembly
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            tiled_matrix[i * n_tiles + j] = hpx::async(&gen_tile, i, j, N, n_tiles);
        }
    }
    // Synchronize
    hpx::wait_all(tiled_matrix);

    return tiled_matrix;
}
