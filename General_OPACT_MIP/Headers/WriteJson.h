#pragma once
#include <string>
#include <vector>
#include "ConnectionOrderOpt.h"  // for column_sol_t, LeftBit, Port

// Legacy (family-based) writer kept for backward compatibility.
void WriteToJsonLegacy(
    const std::string& path,
    const std::vector<std::vector<int>>& V_out,
    const std::vector<std::vector<int>>& F,
    const std::vector<std::vector<int>>& H,
    const std::vector<std::vector<int>>& AC42,
    const std::vector<std::vector<int>>& AC32);

// Generic Pass #1 (all compressors counted by type).
// compCounts[stage][col][type], V_out[stage][col]
void WriteToJsonGeneric(
    const std::string& path,
    const std::vector<std::vector<std::vector<int>>>& compCounts,
    const std::vector<std::vector<int>>& V_out);

// Connection-Order mapping writer (Pass #2).
// Dumps, for each (stage,col): left_bits, right_ports, and right_to_left mapping.
// void WriteToJsonCO(
//     const std::string& path,
//     const std::vector<std::vector<column_sol_t>>& allSols);

// NEW: overload that appends a synthetic “stageS” and a convenience array “V_last”.
void WriteToJsonCO(
    const std::string& path,
    const std::vector<std::vector<column_sol_t>>& allSols,
    const std::vector<int>& last_stage_V);
