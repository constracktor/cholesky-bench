#ifndef CPU_CHOLESKY_FACTOR_H
#define CPU_CHOLESKY_FACTOR_H

#pragma once

#include <stdexcept>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;

namespace cpu
{
enum class Variant { for_collapse, for_naive, task_naive, task_pointer, task_ref };

inline Variant to_variant(const std::string &s)
{
    if (s == "for_collapse")
    {
        return Variant::for_collapse;
    }
    if (s == "for_naive")
    {
        return Variant::for_naive;
    }

    if (s == "task_naive")
    {
        return Variant::task_naive;
    }
    if (s == "task_pointer")
    {
        return Variant::task_pointer;
    }
    if (s == "task_ref")
    {
        return Variant::task_ref;
    }

    throw std::invalid_argument("Unknown Variant: " + std::string(s));
}

void right_looking_cholesky_tiled(Variant variant, Tiled_vector_matrix &tiles);

}  // end of namespace cpu
#endif  // end of CPU_CHOLESKY_FACTOR_H
