#ifndef CPU_FUNCTIONS_H
#define CPU_FUNCTIONS_H

#pragma once

#include "tile_data.hpp"
#include "tile_generation.hpp"

#include <string>
#include <vector>
#include <hpx/future.hpp>
using Tiled_matrix = std::vector<hpx::shared_future<std::vector<double>>>;
namespace cpu
{

double cholesky_future(
    Tiled_matrix &tiled_matrix,
    std::string variant);

double cholesky_loop(
        std::vector<std::vector<double>> &tiled_matrix, std::string variant);

double cholesky_mutable(std::vector<hpx::shared_future<mutable_tile_data<double>>> &mutable_tiled_matrix);

}  // namespace cpu
#endif  // end of CPU_FUNCTIONS_H
