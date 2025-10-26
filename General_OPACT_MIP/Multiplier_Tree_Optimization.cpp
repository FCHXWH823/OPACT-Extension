#include "gurobi_c++.h"
#include "Multiplier_Tree_Optimization.h"

/* Raw per‑column partial‑product counts (MSB->LSB indexing). */
std::vector<int> input_patterns;

void generate_input_patterns() {
    input_patterns.clear();
    for (int i = 1; i <= 2 * MULT_SIZE - 1; ++i)
        input_patterns.push_back(MULT_SIZE - std::abs(i - MULT_SIZE));
}

/* ---------------- Variable creation ---------------- */

void generate_variables_of_multiplier(
    std::vector<std::vector<std::vector<GRBVar>>>& compVars,
    std::vector<std::vector<GRBVar>>& V,
    GRBModel& model)
{
    const int nCols = (int)input_patterns.size() + stages_num;

    /* V[stage][col] for stage 0..stages_num */
    V.resize(stages_num + 1);
    for (int s = 0; s <= stages_num; ++s) {
        V[s].reserve(nCols);
        for (int j = 0; j < nCols; ++j) {
            std::stringstream ss; ss << "V_" << s << "_" << j;
            V[s].push_back(model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER, ss.str()));
        }
    }

    /* Compressor counts for stages 0..stages_num-1 */
    compVars.resize(stages_num);
    for (int s = 0; s < stages_num; ++s) {
        compVars[s].resize(nCols);
        for (int j = 0; j < nCols; ++j) {
            compVars[s][j].resize(COMPRESSOR_LIB.size());
            for (int t = 0; t < (int)COMPRESSOR_LIB.size(); ++t) {
                std::stringstream ss; ss << "x_" << COMPRESSOR_LIB[t].name
                                         << "_" << s << "_" << j;
                compVars[s][j][t] = model.addVar(0, GRB_INFINITY, 0,
                                                 GRB_INTEGER, ss.str());
            }
        }
    }
}

/* ---------------- Stage 0 initialisation ---------------- */

void Initial_constraints_of_V(const std::vector<std::vector<GRBVar>>& V, GRBModel& model) {
    const int nCols = (int)input_patterns.size() + stages_num;
    for (int j = 0; j < nCols; ++j) {
        if (j < stages_num)
            model.addConstr(V[0][j] == 0);
        else
            model.addConstr(V[0][j] == input_patterns[j - stages_num]);
    }
}

/* ---------------- Capacity / propagation / error ---------------- */

void generate_constraints_of_compressors(
    const std::vector<std::vector<std::vector<GRBVar>>>& compVars,
    std::vector<std::vector<GRBVar>>& V,
    GRBVar& TotalErrorSigned,
    GRBVar& AbsError,
    GRBModel& model)
{
    const int nCols = (int)input_patterns.size() + stages_num;
    std::vector<double> bitW(nCols);
    for (int jj = 0; jj < nCols; ++jj)
        bitW[jj] = std::pow(2.0, nCols - jj - 1);
    /* Global signed + absolute error */
    TotalErrorSigned = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0,
                                    GRB_CONTINUOUS, "TotalErrorSigned");
    AbsError         = model.addVar(0, GRB_INFINITY, 0,
                                    GRB_CONTINUOUS, "AbsError");
    GRBLinExpr signedErrExpr = 0;

    for (int s = 0; s < stages_num; ++s) {
        for (int j = 0; j < nCols; ++j) {

            /* Capacity: sum(width * count) <= V[s][j] */
            GRBLinExpr capacity = 0;
            for (int t = 0; t < (int)COMPRESSOR_LIB.size(); ++t)
                capacity += COMPRESSOR_LIB[t].width * compVars[s][j][t];
            model.addConstr(capacity <= V[s][j]);

            /* Propagation bookkeeping */
            GRBLinExpr reduction = 0;
            GRBLinExpr incoming  = 0;

            for (int t = 0; t < (int)COMPRESSOR_LIB.size(); ++t) {
                const auto& spec = COMPRESSOR_LIB[t];

                /* Reduction: 'ew' keeps both outputs locally => width-2.
                   Otherwise one carry shifts to column j+1 => width-1.
                 */
                if (spec.same_col)
                    reduction += (spec.width - 2) * compVars[s][j][t];
                else {
                    reduction += (spec.width - 1) * compVars[s][j][t];
                    if (j + 1 < nCols)      // carry enters less‑significant column
                        incoming += compVars[s][j + 1][t];
                }

                /* Signed error contribution (stage scaling * err_val). */
                if (spec.err_val != 0.0)
                    signedErrExpr += bitW[j] * spec.err_val * compVars[s][j][t];
            }

            if (j == nCols - 1) {
                /* LSB: force exactly one bit to remain and disallow compressors. */
                model.addConstr(V[s + 1][j] == 1);
                for (int t = 0; t < (int)COMPRESSOR_LIB.size(); ++t)
                    model.addConstr(compVars[s][j][t] == 0);
            } else {
                model.addConstr(V[s + 1][j] == V[s][j] - reduction + incoming);
            }
        }
    }

    model.addConstr(TotalErrorSigned == signedErrExpr, "TotalErrorDef");
    model.addConstr(AbsError >=  TotalErrorSigned, "Abs_ge_pos");
    model.addConstr(AbsError >= -TotalErrorSigned, "Abs_ge_neg");
}

/* ---------------- Last stage constraints ---------------- */

void generate_constraints_of_last_stage(const std::vector<std::vector<GRBVar>>& V, GRBModel& model) {
    const int nCols = (int)input_patterns.size() + stages_num;
    for (int j = 0; j < nCols - 1; ++j) {
        model.addConstr(V[stages_num][j] <= 2);
        if (j >= stages_num)  // replicate original lower bound (>=1) on “real” columns
            model.addConstr(V[stages_num][j] >= 1);
    }
}

/* ---------------- Area (legacy & generic) ---------------- */

void build_area_and_legacy_counters(
    const std::vector<std::vector<std::vector<GRBVar>>>& compVars,
    GRBLinExpr& areaExpr,
    GRBVar& fa_num, GRBVar& ha_num, GRBVar& ac32_num, GRBVar& ac42_num,
    GRBModel& model)
{
    areaExpr = 0;
    GRBLinExpr fa_sum = 0, ha_sum = 0, ac32_sum = 0, ac42_sum = 0;

    for (int s = 0; s < stages_num; ++s)
        for (int j = 0; j < (int)compVars[s].size(); ++j)
            for (int t = 0; t < (int)COMPRESSOR_LIB.size(); ++t) {
                const auto& spec = COMPRESSOR_LIB[t];
                areaExpr += spec.area * compVars[s][j][t];
                if (spec.name == "EC32") fa_sum += compVars[s][j][t];
                else if (spec.name == "EC22") ha_sum += compVars[s][j][t];
                else if (spec.name.find("AC32") != std::string::npos) ac32_sum += compVars[s][j][t];
                else if (spec.name.find("AC42") != std::string::npos) ac42_sum += compVars[s][j][t];
            }

    fa_num   = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "fa_num");
    ha_num   = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "ha_num");
    ac32_num = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "ac32_num");
    ac42_num = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "ac42_num");
    model.addConstr(fa_num   == fa_sum);
    model.addConstr(ha_num   == ha_sum);
    model.addConstr(ac32_num == ac32_sum);
    model.addConstr(ac42_num == ac42_sum);
}

GRBLinExpr build_area_expr_generic(const std::vector<std::vector<std::vector<GRBVar>>>& compVars) {
    GRBLinExpr area = 0;
    for (int s = 0; s < stages_num; ++s)
        for (int j = 0; j < (int)compVars[s].size(); ++j)
            for (int t = 0; t < (int)COMPRESSOR_LIB.size(); ++t)
                area += COMPRESSOR_LIB[t].area * compVars[s][j][t];
    return area;
}

/* ---------------- Extraction (legacy + generic) ---------------- */

void extract_legacy_solution(
    const std::vector<std::vector<std::vector<GRBVar>>>& compVars,
    const std::vector<std::vector<GRBVar>>& V,
    std::vector<std::vector<int>>& F,
    std::vector<std::vector<int>>& H,
    std::vector<std::vector<int>>& AC32,
    std::vector<std::vector<int>>& AC42,
    std::vector<std::vector<int>>& V_out)
{
    const int nCols = (int)input_patterns.size() + stages_num;
    F.assign(stages_num,  std::vector<int>(nCols,0));
    H.assign(stages_num,  std::vector<int>(nCols,0));
    AC32.assign(stages_num,std::vector<int>(nCols,0));
    AC42.assign(stages_num,std::vector<int>(nCols,0));
    V_out.assign(stages_num,std::vector<int>(nCols,0));

    for (int s = 0; s < stages_num; ++s)
        for (int j = 0; j < nCols; ++j) {
            V_out[s][j] = int(std::round(V[s+1][j].get(GRB_DoubleAttr_X)));
            for (int t = 0; t < (int)COMPRESSOR_LIB.size(); ++t) {
                int val = int(std::round(compVars[s][j][t].get(GRB_DoubleAttr_X)));
                const auto& spec = COMPRESSOR_LIB[t];
                if (spec.name == "EC32") F[s][j] += val;
                else if (spec.name == "EC22") H[s][j] += val;
                else if (spec.name.find("AC32") != std::string::npos) AC32[s][j] += val;
                else if (spec.name.find("AC42") != std::string::npos) AC42[s][j] += val;
            }
        }
}

/* New: return full per-type counts for generic connection ordering. */
void extract_solution_generic(
    const std::vector<std::vector<std::vector<GRBVar>>>& compVars,
    const std::vector<std::vector<GRBVar>>& V,
    std::vector<std::vector<std::vector<int>>>& compCounts,
    std::vector<std::vector<int>>& V_out)
{
    const int nCols = (int)input_patterns.size() + stages_num;
    compCounts.assign(stages_num,
                      std::vector<std::vector<int>>(nCols,
                        std::vector<int>(COMPRESSOR_LIB.size(),0)));
    V_out.assign(stages_num,std::vector<int>(nCols,0));

    for (int s = 0; s < stages_num; ++s)
        for (int j = 0; j < nCols; ++j) {
            V_out[s][j] = int(std::round(V[s+1][j].get(GRB_DoubleAttr_X)));
            for (int t = 0; t < (int)COMPRESSOR_LIB.size(); ++t)
                compCounts[s][j][t] =
                    int(std::round(compVars[s][j][t].get(GRB_DoubleAttr_X)));
        }
}
