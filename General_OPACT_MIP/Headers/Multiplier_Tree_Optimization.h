#pragma once
#ifndef MULTIPLIER_TREE
#define MULTIPLIER_TREE

#include <iostream>
#include <vector>
#include <sstream>
#include "gurobi_c++.h"
#include "CompressorLib.h"

/* Global parameters (unchanged) */
extern int MULT_SIZE;
extern int stages_num;
extern std::string filename;
extern std::vector<int> input_patterns;

/* Build the initial per-column bit population for a WIDTH×WIDTH array multiplier. */
void generate_input_patterns();

/** Decision variables
 *  compVars[stage][column][type] : integer # of compressors of library type 'type'
 *  V[stage][column]              : integer # of live bits entering stage 'stage' (stage 0 = initial)
 */
void generate_variables_of_multiplier(
    std::vector<std::vector<std::vector<GRBVar>>>& compVars,
    std::vector<std::vector<GRBVar>>& V,
    GRBModel& model);

/** Stage-0 constraints: populate V[0][*] with the raw partial-product counts. */
void Initial_constraints_of_V(const std::vector<std::vector<GRBVar>>& V, GRBModel& model);

/** Capacity + propagation + global (signed+absolute) error modeling.
 *  Creates TotalErrorSigned and AbsError (|TotalErrorSigned|) variables.
 */
void generate_constraints_of_compressors(
    const std::vector<std::vector<std::vector<GRBVar>>>& compVars,
    std::vector<std::vector<GRBVar>>& V,
    GRBVar& TotalErrorSigned,
    GRBVar& AbsError,
    GRBModel& model);

/** Final-stage restriction: every column except the LSB must finish with <=2 bits.
 *  (Matches original behaviour.)
 */
void generate_constraints_of_last_stage(const std::vector<std::vector<GRBVar>>& V, GRBModel& model);

/** Area expression and legacy (FA/HA/AC32/AC42) counters for backward compatibility.
 *  Once connection-order is fully generic you can delete this and call build_area_expr_generic().
 */
void build_area_and_legacy_counters(
    const std::vector<std::vector<std::vector<GRBVar>>>& compVars,
    GRBLinExpr& areaExpr,
    GRBVar& fa_num, GRBVar& ha_num, GRBVar& ac32_num, GRBVar& ac42_num,
    GRBModel& model);

/** Generic area expression (no legacy counters). */
GRBLinExpr build_area_expr_generic(const std::vector<std::vector<std::vector<GRBVar>>>& compVars);

/** ===== Extraction APIs =====
 *  1. extract_legacy_solution(): returns grouped matrices F/H/AC32/AC42 (old pipeline).
 *  2. extract_solution_generic(): returns full per-type counts for connection-order optimisation.
 */
void extract_legacy_solution(
    const std::vector<std::vector<std::vector<GRBVar>>>& compVars,
    const std::vector<std::vector<GRBVar>>& V,
    std::vector<std::vector<int>>& F,
    std::vector<std::vector<int>>& H,
    std::vector<std::vector<int>>& AC32,
    std::vector<std::vector<int>>& AC42,
    std::vector<std::vector<int>>& V_out);

/* New: expose every type.  compCounts[stage][column][type] */
void extract_solution_generic(
    const std::vector<std::vector<std::vector<GRBVar>>>& compVars,
    const std::vector<std::vector<GRBVar>>& V,
    std::vector<std::vector<std::vector<int>>>& compCounts,
    std::vector<std::vector<int>>& V_out);

#endif // !MULTIPLIER_TREE
