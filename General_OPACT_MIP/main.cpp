#include "gurobi_c++.h"
#include "CompressorLib.h"
#include "ProbEval.h"
#include "Multiplier_Tree_Optimization.h"
#include "ConnectionOrderOpt.h"
#include "WriteJson.h"
#include <iostream>
#include <vector>
#include <string>

int PARAMETER_W, PARAMETER_L, MULT_SIZE, stages_num;
int Opt_Mode;              // 0: minimize area with error bound; 1: minimize |error| with area bound; else: weighted sum
float MED_bound, AREA_bound, Weight;
std::string filename;


int main() {
    try {
        /* ----- Library + truth tables ----- */
        const std::string lib_path = "/Users/fch/Python/OPACT-Extension/ACs/ac_lib.json";
        load_compressor_lib(lib_path);
        load_truth_tables_from_json(lib_path);

        /* ----- Parameters (adapt to your CLI/config) ----- */
        MULT_SIZE = 16;
        stages_num = 4;
        Opt_Mode = 2;       // minimize |error|
        MED_bound = 100.0f; // used if Opt_Mode==0
        AREA_bound = 640.0f;
        Weight = 0.01;
        filename = "gurobi";

        /* ----- Pass #1: Sizing ----- */
        GRBEnv env = GRBEnv();
		GRBModel model = GRBModel(env);
        model.set(GRB_INT_PAR_PRESOLVE, "2");
        model.set(GRB_DBL_PAR_TIMELIMIT, "20");

        generate_input_patterns();

        // test input patterns
        std::cout << "Input patterns: ";
        for (const auto& pat : input_patterns) {
            std::cout << pat << " ";
        }
        std::cout << std::endl;
        std::vector<std::vector<std::vector<GRBVar>>> compVars;
        std::vector<std::vector<GRBVar>> V;
        generate_variables_of_multiplier(compVars, V, model);
        Initial_constraints_of_V(V, model);

        GRBVar TotalErrorSigned, AbsError;
        generate_constraints_of_compressors(compVars, V, TotalErrorSigned, AbsError, model);
        generate_constraints_of_last_stage(V, model);

        // Objective for pass #1
        GRBLinExpr areaExpr = build_area_expr_generic(compVars);
        GRBLinExpr MED = AbsError;
        if (Opt_Mode == 0) { model.addConstr(AbsError <= MED_bound); model.setObjective(areaExpr, GRB_MINIMIZE); }
        else if (Opt_Mode == 1) { model.addConstr(areaExpr <= AREA_bound); model.setObjective(MED, GRB_MINIMIZE); }
        else { model.setObjective(areaExpr + Weight * AbsError, GRB_MINIMIZE); }

        model.write(filename + "_p1.lp");
        model.optimize();
        model.write(filename + "_p1.sol");

        // print final area and absolute error
        // get the value of areaExpr
        std::cout << "Final area: " << areaExpr.getValue() << std::endl;
        // std::cout << "Final area: " << model.getObjective().getValue() << std::endl;
        std::cout << "Final absolute error: " << AbsError.get(GRB_DoubleAttr_X) << std::endl;

        // Read solution values into compact integer arrays
        std::vector<std::vector<std::vector<int>>> compCounts;  // [stage][col][type]
        std::vector<std::vector<int>> V_out_solution;           // [stage][col]
        extract_solution_generic(compVars, V, compCounts, V_out_solution);

        // Optionally persist the sizing (Pass #1) result
        WriteToJsonGeneric(filename + "_generic_p1.json", compCounts, V_out_solution);

        std::cout << "Pass#1: |Error| = " << AbsError.get(GRB_DoubleAttr_X)
                << "  Area(sum) = " << model.getObjective().getValue() << std::endl;

        // ----- Pass #2: Connection order (all compressor types) -----
        /* ------------------------------------------------------------------ *
         * 0)  **Append a virtual FINAL stage**: no compressors, same V-out
         * ------------------------------------------------------------------ */
        {
            const int C = static_cast<int>(compCounts[0].size());
            const int T = static_cast<int>(COMPRESSOR_LIB.size());

            // a) compCounts ← push_back  zeros  [C][T]
            compCounts.push_back(std::vector<std::vector<int>>(C,
                                    std::vector<int>(T, 0)));

            // b) V_out_solution ← push_back copy of last real row
            V_out_solution.push_back(V_out_solution.back());
        }


        // 1) Initial PP population per column for an N×N multiplier
        const int C = static_cast<int>(compCounts[0].size());
        input_patterns.insert(input_patterns.begin(), stages_num, 0); // add 0 for stage 0

        // 2) Bit significance (MSB-first): bitW[0] = 2^(C-1), ..., bitW[C-1] = 2^0.
        std::vector<double> bitW(C);
        for (int j = 0; j < C; ++j) {
            bitW[j] = std::ldexp(1.0, C - 1 - j);  // == pow(2.0, C-1-j)
        }


        // 3) Run connection‑order optimization (creates its own Gurobi model)
        auto co_res = optimize_connection_order(
            compCounts,           // [stage][col][type] counts from Pass #1
            V_out_solution,       // [stage][col] bit counts after each stage
            input_patterns,       // initial PP bits at s==0
            bitW,                 // bit significance per column
            filename + "_co"      // run tag for LP/SOL
        );

        // for (int i = 0; i < co_res.allSols.size(); i++)
		// 	reverse(co_res.allSols[i].begin(), co_res.allSols[i].end());

        // 4) Report final CO result
        std::cout << "CO: signed error = " << co_res.signed_error
                << " | |error| = " << co_res.abs_error << std::endl;

        // 5) Persist the connection-order mapping (allSols)
        //    This writes per (stage,col): left_bits, right_ports, and right_to_left mapping.
        WriteToJsonCO(filename + "_co_map.json", co_res.allSols, V_out_solution[stages_num - 1]);

        std::cout << "Done." << std::endl;


    } catch (GRBException& e) {
        std::cerr << "Gurobi exception: " << e.getMessage() << std::endl;
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
