#pragma once
#ifndef __COMPRESSOR_LIB_H__
#define __COMPRESSOR_LIB_H__

#include <string>
#include <vector>
#include <unordered_map>

struct CompressorSpec {
    std::string name;      // key in JSON (e.g. "AC42_uw1", "EC22")
    int width;             // number of input bits the compressor consumes from a column
    double area;           // area cost
    double err_val;        // MED at p = 0.25 (signed)
    bool same_col;         // true if name contains "ew" (both outputs stay in same column)
};

extern std::vector<CompressorSpec> COMPRESSOR_LIB;
extern std::unordered_map<std::string,int> COMPRESSOR_INDEX;

/** Load ac_lib.json (throws std::runtime_error on failure). */
void load_compressor_lib(const std::string& jsonPath);

/** Convenience helpers used by the optimization code. */
double stage_scale(const CompressorSpec& c, int stage);   // width/2 ^ stage

#endif
