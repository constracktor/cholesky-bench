#pragma once 

#include "tile_data.hpp"

#include <cmath>
#include <vector>
#include <hpx/algorithm.hpp>

// Tile generation

double compute_covariance_function(std::size_t i_global,
                                   std::size_t j_global,
                                   std::size_t n_regressors,
                                   const std::vector<double> &input)
{
    // SEK params
    double lengthscale = 1.0;
    double vertical_lengthsale = 1.0;
    double noise_variance = (i_global == j_global ? 0.1 : 0.0);
    // k(z_i,z_j) = vertical_lengthscale * exp(-0.5 / lengthscale^2 * (z_i - z_j)^2)
    double distance = 0.0;
    double z_ik_minus_z_jk;

    for (std::size_t k = 0; k < n_regressors; k++)
    {
        z_ik_minus_z_jk = input[i_global + k] - input[j_global + k];
        distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
    }

    return lengthscale * exp(-0.5 / (lengthscale * lengthscale) * distance) + noise_variance;
}

std::vector<double> gen_tile(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const std::vector<double> &input)
{
    std::size_t i_global, j_global;
    // Preallocate required memory
    std::vector<double> tile;
    tile.reserve(N * N);
    // Compute entries
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            j_global = N * col + j;
            // compute covariance function
            tile.push_back(compute_covariance_function(i_global, j_global, n_regressors, input));
        }
    }
    return tile;
}

std::vector<double> load_data(const std::string &file_path, int n_samples, int offset)
{
    std::vector<double> _data;
    _data.resize(static_cast<std::size_t>(n_samples + offset), 0.0);

    FILE *input_file = fopen(file_path.c_str(), "r");
    if (input_file == NULL)
    {
        throw std::runtime_error("Error: File not found: " + file_path);
    }

    // load data
    int scanned_elements = 0;
    for (int i = 0; i < n_samples; i++)
    {
        scanned_elements +=
            fscanf(input_file, "%lf", &_data[static_cast<std::size_t>(i + offset)]);  // scanned_elements++;
    }

    fclose(input_file);

    if (scanned_elements != n_samples)
    {
        throw std::runtime_error("Error: Data not correctly read. Expected " + std::to_string(n_samples)
                                 + " elements, but read " + std::to_string(scanned_elements));
    }
    return _data;
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
    std::size_t n_regressors = 8;
    std::vector<double> input = load_data("data/input_19.txt", N * n_tiles, n_regressors);
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
            tiled_matrix[i * n_tiles + j] = hpx::async(&gen_tile, i, j, N, n_regressors, input);
        }
    }
    // Synchronize
    hpx::wait_all(tiled_matrix);

    return tiled_matrix;
}
