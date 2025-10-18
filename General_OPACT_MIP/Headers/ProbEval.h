#pragma once
#ifndef __PROB_EVAL_H__
#define __PROB_EVAL_H__

#include <vector>
#include "CompressorLib.h"

/** Optional truth tables for a compressor (read from JSON if present). */
struct TruthTables {
    // For width w, vectors have length 2^w, index = binary pattern of inputs (x_{w-1}..x0)
    std::vector<int> s;  // S(x)
    std::vector<int> c;  // C(x)
    bool valid() const { return !s.empty() && s.size() == c.size(); }
};

/** Per-type truth tables, parallel to COMPRESSOR_LIB (empty if not provided). */
extern std::vector<TruthTables> COMP_TT;

/** Load truth tables from JSON if fields exist: "truth_s", "truth_c". */
void load_truth_tables_from_json(const std::string& jsonPath);

/** E[S], E[C] for exact compressors (EC22/EC32) with independent inputs. */
void exact_E_SC_EC22(double p1, double p2, double& ES, double& EC);
void exact_E_SC_EC32(double p1, double p2, double p3, double& ES, double& EC);

/** Generic E[S], E[C] using truth tables.  inputs.size() == width. */
void tt_E_SC(const TruthTables& tt, const std::vector<double>& inputs, double& ES, double& EC);

#endif
