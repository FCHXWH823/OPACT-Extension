#pragma once
#ifndef __WIRE_CONNECT_H__
#define __WIRE_CONNECT_H__

#include <vector>
#include <string>
#include "CompressorLib.h"

/** Legacy structure retained for backward compatibility. */
struct SolsLegacy {
    std::vector<std::vector<int>> V, F, H, AC42, AC32;
    SolsLegacy() = default;
    SolsLegacy(const std::vector<std::vector<int>>& V_,
               const std::vector<std::vector<int>>& F_,
               const std::vector<std::vector<int>>& H_,
               const std::vector<std::vector<int>>& AC42_,
               const std::vector<std::vector<int>>& AC32_)
        : V(V_), F(F_), H(H_), AC42(AC42_), AC32(AC32_) {}
};

/** Generic solution for connection ordering and serialization. */
struct SolsGeneric {
    // [stage][column][type] counts
    std::vector<std::vector<std::vector<int>>> compCounts;
    // V_bits[stage][column]
    std::vector<std::vector<int>> V_bits;
};

#endif
