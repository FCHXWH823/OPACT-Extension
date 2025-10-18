/*********************************************************************
 *  Requires JsonCpp (https://github.com/open-source-parsers/jsoncpp)
 *  g++  -std=c++17  parse_allocation.cpp  -ljsoncpp
 *********************************************************************/
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <json/json.h>
#include "gurobi_c++.h"
#include "Multiplier_Tree_Optimization.h"
#include "ConnectionOrderOpt.h"
#include "wireConnect.h"
#include <iostream>
#include <vector>
#include "Simulator.h"
#include <sstream>
#include <fstream>
#include <string>
#include "WriteJson.h"
#include <filesystem>
#include <regex>
#include <utility>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace std;
using std::string;
using std::vector;
namespace fs = std::filesystem;



using Point = std::pair<double,double>;   // (f1 , f2)

/**
 * 2-objective non-dominated sorting (minimisation).
 *
 * @param objs   population of points (f1,f2)
 * @return       fronts[i] = vector of points in Pareto front i   (0 = best)
 *
 * Complexity    O(N log N)   – sort by f1, then sweep on f2.
 */
std::vector<std::vector<Point>>
nondominatedSort(const std::vector<Point>& objs)
{
    const std::size_t N = objs.size();
    std::vector<std::size_t> perm(N);                  // permutation of indices
    std::iota(perm.begin(), perm.end(), 0);

    /* 1) sort indices by ascending f1, then ascending f2 */
    std::sort(perm.begin(), perm.end(),
              [&objs](std::size_t a, std::size_t b)
              {
                  if (objs[a].first != objs[b].first)
                      return objs[a].first < objs[b].first;
                  return objs[a].second < objs[b].second;
              });

    std::vector<std::vector<Point>> fronts;            // result
    std::vector<double> minF2;                         // best f2 per front

    /* 2) sweep */
    for (std::size_t id : perm)
    {
        const double f2 = objs[id].second;

        /* first front whose current *best* f2 is < f2
           – duplicates (same f1 & f2) join the same front           */
        std::size_t pos = 0;
        while (pos < minF2.size() && minF2[pos] < f2)
            ++pos;

        if (pos == fronts.size()) {                    // new front needed
            fronts.emplace_back();                     // create front
            minF2.push_back(std::numeric_limits<double>::infinity());
        }

        fronts[pos].push_back(objs[id]);               // store the point
        minF2[pos] = std::min(minF2[pos], f2);         // update best f2
    }
    return fronts;
}

std::vector<std::pair<double,double>>
readFirstFrontier(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open file: " + path);

    std::string line;
    const std::regex pair_re(
        R"(\(\s*([+-]?(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?)\s*\))");

    while (std::getline(in, line))
    {
        if (line.rfind("Front 0:", 0) == 0)          // starts with "Front 0:"
        {
            std::vector<std::pair<double,double>> frontier;
            std::sregex_iterator it(line.begin(), line.end(), pair_re);
            std::sregex_iterator end;

            for (; it != end; ++it)
            {
                double x = std::stod((*it)[1]);
                double y = std::stod((*it)[2]);
                frontier.emplace_back(x, y);
            }
            return frontier;                         // done
        }
    }
    throw std::runtime_error("No \"Front 0:\" line found in " + path);
}

double read_med(const std::string& path)
{
    std::ifstream in(path);
    // if (!in) throw std::runtime_error("cannot open " + path);

    std::regex  re(R"(Mean error distance.*?:\s*([0-9]*\.?[0-9]+))");
    std::smatch m;
    std::string line;

    while (std::getline(in, line))
        if (std::regex_search(line, m, re))
            return std::stod(m[1]);       // matched group #1 is the number

    // throw std::runtime_error("MED line not found");
}

/**
 * Parse <file>.json   →   populate F, H, AC32, AC42, V
 *
 *  JSON layout expected (one column object shown):
 *    "stage0": [
 *      {
 *        "col_idx": 0,
 *        "bits_before": 17,
 *        "bits_after":  13,
 *        "alloc": {
 *          "ex-3:2":  3,
 *          "ex-2:2":  1,
 *          "ap-3:2":  2,
 *          "ap-4:2":  0,
 *          "dummy":   1
 *        }
 *      }, ...
 *    ],
 *    "stage1": [ ... ], ...
 *
 *  `stages_num`  – how many pipeline stages you want to import
 *  `MULT_SIZE`   – operand width ⇒ physical columns = 2*M + stages_num - 1
 */
bool loadAllocationJSON(const string& file,
                        int stages_num,
                        int MULT_SIZE,
                        vector<vector<int>>& F,
                        vector<vector<int>>& H,
                        vector<vector<int>>& AC32,
                        vector<vector<int>>& AC42,
                        vector<vector<int>>& V)
{
    // ---------------- JsonCpp: read file ----------------------------------
    Json::Value root;
    std::ifstream ifs(file, std::ifstream::binary);
    if (!ifs) { std::cerr << "Cannot open " << file << '\n'; return false; }

    ifs >> root;

    const int C = 2 * MULT_SIZE + stages_num - 1;     // total columns incl. padding

    // quick sanity check — allow missing padding columns (they'll stay 0)
    for (int s = 0; s < stages_num + 1; ++s) {
        string key = "stage" + std::to_string(s);
        if (!root.isMember(key)) {
            std::cerr << "Missing " << key << " in JSON\n";
            return false;
        }
        if (!root[key].isArray()) {
            std::cerr << key << " is not an array\n";
            return false;
        }
    }

    // ---------------- populate the output matrices ------------------------
    for (int s = 0; s < stages_num + 1; ++s) {
        const auto& stageArr = root["stage" + std::to_string(s)];
        for (Json::ArrayIndex j = 0; j < stageArr.size() && j < C; ++j) {
            const auto& col = stageArr[j];

            // ---- compressor counts --------------------------------------
            if (s < stages_num) {
                const auto& alloc = col["alloc"];
                F   [s][j] = alloc["exact_3to2"].asInt();
                H   [s][j] = alloc["exact_2to2"].asInt();
                AC32[s][j] = alloc["approx_3to2"].asInt();
                AC42[s][j] = alloc["approx_4to2"].asInt();
            }
            // dummy counts are ignored here; add if you need them

            // ---- bit counts (take bits_before; use bits_after if preferred)
            if (s)
                V[s-1][j] = col["bits"].asInt();
        }
    }
    return true;
}
int stages_num = 4;
int MULT_SIZE = 16;
vector<vector<double>> allcoefs;
string filename1;
string filename;
vector<int> input_patterns;
void CAOE(pair<double, double>& solution, double Area_bound) {
    vector<vector<int>> F(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
    vector<vector<int>> H(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
    vector<vector<int>> AC32(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
    vector<vector<int>> AC42(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
    vector<vector<double>> E(stages_num, vector<double>(2 * MULT_SIZE + stages_num - 1, 0.0));
    vector<vector<int>> V(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
    //optimization process
    try {
        filename = "gurobi";
        /*******Set up environment******/
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        // model.set(GRB_StringParam_LogFile, filename + ".log");
        //model.set(GRB_INT_PAR_MIPFOCUS, "1");
        model.set(GRB_INT_PAR_PRESOLVE, "2");
        /*model.set(GRB_DBL_PAR_TIMELIMIT, to_string(Time_Bound_s));*/
        //model.set(GRB_DBL_PAR_HEURISTICS, "0.5");
        //model.set(GRB_DBL_PAR_MIPGAP, "0.005");
        /*******generate input patterns******/
        input_patterns = {};
        generate_input_patterns();

        /*******Creat e variables******/
        vector<double> MED_ac42(stages_num, double(14) / 256);
        vector<double> MED_ac32(stages_num, double(1) / 64);
        /*for (int i = 0; i < stages_num; i++)
        {
            MED_ac32[i] = pow(1.5, i) * MED_ac32[i];
            MED_ac42[i] = pow(2, i) * MED_ac42[i];
        }*/
        //generate_MEDs(MED_ac42, MED_ac32);

        /*******Create variables******/
        //Compressor tree
        vector<vector<GRBVar>> variables_fa, variables_ha;
        vector<vector<GRBVar>> variables_V;
        vector<vector<GRBVar>> variables_ac32, variables_ac42;
        vector<vector<GRBVar>> variables_error;
        GRBVar fa_num, ha_num, ac32_num, ac42_num, E;
        generate_variables_of_multiplier(variables_fa, variables_ha, variables_V, variables_ac32, variables_ac42, variables_error, model);

        /*Add general constraints*/
        //Compressor Tree
        Initial_constraints_of_V(variables_V, model);
        generate_constraints_of_h_f_V(variables_fa, variables_ha, variables_V, variables_ac32, variables_ac42, variables_error, MED_ac42, MED_ac32, model);
        generate_constraints_of_last_stage(variables_V, model);

        // Set objective
        GRBLinExpr obj = 0;
        generate_cost_of_adders(variables_fa, variables_ha, fa_num, ha_num, variables_ac32, variables_ac42, ac32_num, ac42_num, model);
        generate_total_error(variables_error, E, model);
    
        model.addConstr((5.05 * fa_num + 2.66 * ha_num + 2.66 * ac32_num + 4.78 * ac42_num) <= Area_bound);
        obj += E;
        
        model.setObjective(obj, GRB_MINIMIZE);

        // Save problem
        model.write(filename + ".lp");

        // Optimize model
        model.update();

        model.optimize();

        //Save solution

        model.write(filename + ".sol");
        int nFA, nHA, nAC42, nAC32;
        nFA = fa_num.get(GRB_DoubleAttr_X);
        nHA = ha_num.get(GRB_DoubleAttr_X);
        nAC32 = ac32_num.get(GRB_DoubleAttr_X);
        nAC42 = int(ac42_num.get(GRB_DoubleAttr_X));
        solution.first = (5.05 * nFA + 2.66 * nHA + 2.66 * nAC32 + 4.78 * nAC42);

        for (int s = 0; s < stages_num; s++) {
            for (int j = 0; j < variables_V[s].size(); j++) {
                V[s][j] = int(variables_V[s+1][j].get(GRB_DoubleAttr_X));
                F[s][j] = int(variables_fa[s][j].get(GRB_DoubleAttr_X));
                H[s][j] = int(variables_ha[s][j].get(GRB_DoubleAttr_X));
                AC32[s][j] = int(variables_ac32[s][j].get(GRB_DoubleAttr_X));
                AC42[s][j] = int(variables_ac42[s][j].get(GRB_DoubleAttr_X));
            }
        }

    }
    catch (GRBException e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
    }
    catch (...) {
        cout << "Exception during optimization" << endl;
    }

    int Time_Bound_s = 10;
    int EP_Approx = 0;

    string simulation_path = "/Users/fch/Python/ApproximateMult/ApproxMULT_MyHDL/ApproxMULT_MyHDL/simulation/"; //default simulator path
    int NTEST = (2 * MULT_SIZE > 20) ? (1 << 20) : (1 << (2 * MULT_SIZE)); //default number of tests for simulation
    // optimize
    try {
        filename1 = "gurobi_CO";
        /*******Set up environment******/
        GRBEnv env = GRBEnv();
        GRBModel model_CO = GRBModel(env);
        model_CO.set(GRB_StringParam_LogFile, filename1 + ".log");
        model_CO.set(GRB_INT_PAR_MIPFOCUS, "1");
        model_CO.set(GRB_INT_PAR_NUMERICFOCUS, "1");
        //model_CO.set(GRB_INT_PAR_CUTS, "2");
        //model_CO.set(GRB_INT_PAR_MIQCPMETHOD, "0");
        //model_CO.set(GRB_INT_PAR_NODEMETHOD, "1");
        //model_CO.set(GRB_INT_PAR_PREQLINEARIZE, "1");
        model_CO.set(GRB_INT_PAR_PRESOLVE, "2");
        model_CO.set(GRB_DBL_PAR_TIMELIMIT, to_string(Time_Bound_s));
        model_CO.set(GRB_DBL_PAR_HEURISTICS, "0.4");
        //model_CO.set(GRB_DBL_PAR_MIPGAP, "0.005");
        model_CO.set(GRB_INT_PAR_NONCONVEX, "2");

        // ConstructCT(F, H, AC32, AC42, V);
        vector<vector<Vars_Column>> allVars;
        vector<vector<GRBVar>> allMEDs;
        vector<int> initial_stage = input_patterns;
        initial_stage.insert(initial_stage.begin(), stages_num, 0);
        V.insert(V.begin(), initial_stage);
        F.insert(F.end(), vector<int>(F[0].size(), 0));
        H.insert(H.end(), vector<int>(H[0].size(), 0));
        AC32.insert(AC32.end(), vector<int>(AC32[0].size(), 0));
        AC42.insert(AC42.end(), vector<int>(AC42[0].size(), 0));
        Sols sol(V, F, H, AC42, AC32);

        generate_variables_of_ConnectOrderOpt(allVars, sol, model_CO);
        Initial_probs_of_V(allVars[0], model_CO);
        allMEDs = generate_constraints_of_ConnectOrderOpt(allVars, EP_Approx, model_CO);

		// vector<GRBVar> Errors = generate_constraints_of_ConnectOrderOpt_last_stage(probs_last_stage, connect_last_stage, model_CO);
        GRBLinExpr MED = 0;
        for (int i = 0; i < allMEDs.size(); i++)
            for (int j = 0; j < allMEDs[i].size(); j++)
                MED += allMEDs[i][j] * pow(2, allMEDs[i].size() - j - 1);
        model_CO.setObjective(MED, GRB_MINIMIZE);

        // Save problem
        model_CO.write(filename1 + ".lp");

        // Optimize model

        model_CO.optimize();

        //Save solution

        model_CO.write(filename1 + ".sol");

        // post verification
        vector<vector<column_sol_t>> allSols;
        allSols = Init_Sols(allVars);

        for (int i = 0; i < allSols.size(); i++)
            reverse(allSols[i].begin(), allSols[i].end());

        string path = "/Users/fch/Python/ApproximateMult/ApproxMULT_MyHDL/ApproxMULT_MyHDL/";
        string file = "CompressorTree.json";
        WriteToJson(allSols, path + file);

        system(("python \"" + path + "ApproxMULT_MyHDL.py\" \"" + path + file + "\" ILP_ApproxMult " + to_string(MULT_SIZE) + " " + simulation_path).c_str());
        system(("cd " + simulation_path + "; ./run.sh -w " + to_string(MULT_SIZE) + " -n " + to_string(NTEST) + " -m approx_mult 1>/dev/null").c_str());
        system(("cd " + simulation_path + "; ./sim > output.txt").c_str());
        solution.second = read_med(simulation_path + "output.txt");;
        
    }
    catch (GRBException e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
    }
    catch (...) {
        cout << "Exception during optimization" << endl;
    }

}

int postProcess(pair<double, double>& solution, string file, int errorCompensate = 0)
{
    cout << "Post-processing allocation from " << file << "...\n";
    // string file = "/Users/fch/Python/OPACT-Extension/Training_log/AC_Allocation_43000.json";
    // string file = "/Users/fch/Python/OPACT-Extension/Training_log/AC_Allocation_8320.json";
    // string file = "/Users/fch/Python/OPACT-Extension/Training_log/AC_Allocation_99990.json";
    // string file = "/Users/fch/Python/OPACT-Extension/Training_log_0.001/AC_Allocation_3440.json";
    
    vector<vector<int>> F(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
    vector<vector<int>> H(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
    vector<vector<int>> AC32(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
    vector<vector<int>> AC42(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
    vector<vector<int>> V(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));

    if (loadAllocationJSON(file, stages_num, MULT_SIZE, F, H, AC32, AC42, V)) {
        std::cout << "Allocation data loaded successfully.\n";
        // reverse each vector
        for (int s = 0; s < stages_num; ++s) {
            std::reverse(F[s].begin(), F[s].end());
            std::reverse(H[s].begin(), H[s].end());
            std::reverse(AC32[s].begin(), AC32[s].end());
            std::reverse(AC42[s].begin(), AC42[s].end());
            std::reverse(V[s].begin(), V[s].end());
        }
        // std::reverse(V[stages_num].begin(), V[stages_num].end());
    } else {
        std::cerr << "Failed to load allocation data.\n";
        return 1;
    }
    // Output the loaded data for verification
    for (int s = 0; s < stages_num; ++s) {
        std::cout << "Stage " << s << ":\n";
        for (int j = 0; j < F[s].size(); ++j)
        {
            std::cout << "Col " << j << ": "
                      << "F=" << F[s][j] << ", "
                      << "H=" << H[s][j] << ", "
                      << "AC32=" << AC32[s][j] << ", "
                      << "AC42=" << AC42[s][j] << ", "
                      << "V=" << V[s][j] << "\n";
        }
    }
    // output the total number of F, H, AC32, AC42
    int total_F = 0, total_H = 0, total_AC32 = 0, total_AC42 = 0;
    for(int s = 0; s < stages_num; s++)
    {
        for (int j = 0; j < F[s].size(); j++)
        {
            total_F += F[s][j];
            total_H += H[s][j];
            total_AC32 += AC32[s][j];
            total_AC42 += AC42[s][j];
        }
    }
    std::cout << "Total F=" << total_F << ", "
    << "Total H=" << total_H << ", "
    << "Total AC32=" << total_AC32 << ", "
    << "Total AC42=" << total_AC42 << "\n";

    // compensate compressors
    try
    {
        filename = "gurobi";
        /*******Set up environment******/
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.set(GRB_StringParam_LogFile, filename + ".log");
        model.set(GRB_INT_PAR_MIPFOCUS, "1");
        model.set(GRB_INT_PAR_NUMERICFOCUS, "1");
        //model_CO.set(GRB_INT_PAR_CUTS, "2");
        //model_CO.set(GRB_INT_PAR_MIQCPMETHOD, "0");
        //model_CO.set(GRB_INT_PAR_NODEMETHOD, "1");
        //model_CO.set(GRB_INT_PAR_PREQLINEARIZE, "1");
        model.set(GRB_INT_PAR_PRESOLVE, "2");
        model.set(GRB_DBL_PAR_TIMELIMIT, "100"); 
        model.set(GRB_DBL_PAR_HEURISTICS, "0.4");
        //model_CO.set(GRB_DBL_PAR_MIPGAP, "0.005");
        model.set(GRB_INT_PAR_NONCONVEX, "2");

        /*******Create variables******/
        //Compressor tree
        vector<vector<GRBVar>> variables_fa, variables_ha;
        vector<vector<GRBVar>> variables_V;
        vector<vector<GRBVar>> variables_ac32, variables_ac42;
        GRBVar fa_num, ha_num, ac32_num, ac42_num, E;
        generate_variables_of_multiplier(variables_fa, variables_ha, variables_V, variables_ac32, variables_ac42, model);

        /*Add general constraints*/
        //Compressor Tree
        Initial_constraints_of_V(variables_V, model);
        generate_constraints_of_h_f_V(variables_fa, variables_ha, variables_V, variables_ac32, variables_ac42, model);
        generate_constraints_of_last_stage(variables_V, model);

        if (errorCompensate == 1)
            generate_constraints_error_compensate1(AC32, AC42, variables_ac32, variables_ac42, model);
        else if (errorCompensate == 0)   
            generate_constraints_error_compensate0(AC32, AC42, variables_ac32, variables_ac42, model);
        else
            generate_constraints_error_compensate2(AC32, AC42, variables_ac32, variables_ac42, model);

        // Set objective
        GRBLinExpr obj = 0;
        generate_cost_of_adders(variables_fa, variables_ha, fa_num, ha_num, variables_ac32, variables_ac42, ac32_num, ac42_num, model);

        obj += (5.05 * fa_num + 2.66 * ha_num + 2.66 * ac32_num + 4.78 * ac42_num);
        model.setObjective(obj, GRB_MINIMIZE);

        model.write("gurobi.lp");

        model.optimize();

        //Save solution
        model.write(filename + ".sol");

        cout << fa_num.get(GRB_StringAttr_VarName) << " "
				<< fa_num.get(GRB_DoubleAttr_X) << endl;
        cout << ha_num.get(GRB_StringAttr_VarName) << " "
            << ha_num.get(GRB_DoubleAttr_X) << endl;
        cout << ac32_num.get(GRB_StringAttr_VarName) << " "
            << ac32_num.get(GRB_DoubleAttr_X) << endl;
        cout << ac42_num.get(GRB_StringAttr_VarName) << " "
            << ac42_num.get(GRB_DoubleAttr_X) << endl;

        solution.first = model.get(GRB_DoubleAttr_ObjVal);
        // update V, F, H, AC42, AC32
        for (int s = 0; s < stages_num; s++) {
            for (int j = 0; j < variables_V[s].size(); j++) {
                V[s][j] = int(round(variables_V[s+1][j].get(GRB_DoubleAttr_X)));
                F[s][j] = int(round(variables_fa[s][j].get(GRB_DoubleAttr_X)));
                H[s][j] = int(round(variables_ha[s][j].get(GRB_DoubleAttr_X)));
                AC32[s][j] = int(round(variables_ac32[s][j].get(GRB_DoubleAttr_X)));
                AC42[s][j] = int(round(variables_ac42[s][j].get(GRB_DoubleAttr_X)));
            }
        }

        // // show the last stage of variables_V
        // vector<GRBVar> last_stage_V = variables_V[variables_V.size() - 1];
        // cout << "Last stage variables_V:" << endl;
        // for (int j = 0; j < last_stage_V.size(); j++)
        //     cout << last_stage_V[j].get(GRB_DoubleAttr_X) << " ";
        // cout << endl;
    }
    catch (GRBException e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
        return 0;
    }
    catch (...) {
        cout << "Exception during optimization" << endl;
        return 0;
    }
    

    // optimize
    filename1 = "gurobi_CO";
    int Time_Bound_s = 10;
    int EP_Approx = 0;
	
    string simulation_path = "/Users/fch/Python/ApproximateMult/ApproxMULT_MyHDL/ApproxMULT_MyHDL/simulation/"; //default simulator path
    int NTEST = (2 * MULT_SIZE > 20) ? (1 << 20) : (1 << (2 * MULT_SIZE)); //default number of tests for simulation
    try {
        /*******Set up environment******/
        GRBEnv env = GRBEnv();
        GRBModel model_CO = GRBModel(env);
        model_CO.set(GRB_StringParam_LogFile, filename1 + ".log");
        model_CO.set(GRB_INT_PAR_MIPFOCUS, "1");
        model_CO.set(GRB_INT_PAR_NUMERICFOCUS, "1");
        //model_CO.set(GRB_INT_PAR_CUTS, "2");
        //model_CO.set(GRB_INT_PAR_MIQCPMETHOD, "0");
        //model_CO.set(GRB_INT_PAR_NODEMETHOD, "1");
        //model_CO.set(GRB_INT_PAR_PREQLINEARIZE, "1");
        model_CO.set(GRB_INT_PAR_PRESOLVE, "2");
        model_CO.set(GRB_DBL_PAR_TIMELIMIT, to_string(Time_Bound_s));
        model_CO.set(GRB_DBL_PAR_HEURISTICS, "0.4");
        //model_CO.set(GRB_DBL_PAR_MIPGAP, "0.005");
        model_CO.set(GRB_INT_PAR_NONCONVEX, "2");

        // ConstructCT(F, H, AC32, AC42, V);
        vector<vector<Vars_Column>> allVars;
        vector<vector<GRBVar>> allMEDs;
        vector<int> initial_stage = input_patterns;
        initial_stage.insert(initial_stage.begin(), stages_num, 0);
        V.insert(V.begin(), initial_stage);
        F.insert(F.end(), vector<int>(F[0].size(), 0));
        H.insert(H.end(), vector<int>(H[0].size(), 0));
        AC32.insert(AC32.end(), vector<int>(AC32[0].size(), 0));
        AC42.insert(AC42.end(), vector<int>(AC42[0].size(), 0));
        Sols sol(V, F, H, AC42, AC32);

        generate_variables_of_ConnectOrderOpt(allVars, sol, model_CO);
        Initial_probs_of_V(allVars[0], model_CO);
        allMEDs = generate_constraints_of_ConnectOrderOpt(allVars, EP_Approx, model_CO);

		// vector<GRBVar> Errors = generate_constraints_of_ConnectOrderOpt_last_stage(probs_last_stage, connect_last_stage, model_CO);
        GRBLinExpr MED = 0;
        for (int i = 0; i < allMEDs.size(); i++)
            for (int j = 0; j < allMEDs[i].size(); j++)
                MED += allMEDs[i][j] * pow(2, allMEDs[i].size() - j - 1);
        model_CO.setObjective(MED, GRB_MINIMIZE);

        // Save problem
        model_CO.write(filename1 + ".lp");

        // Optimize model

        model_CO.optimize();

        //Save solution

        model_CO.write(filename1 + ".sol");

        // post verification
        vector<vector<column_sol_t>> allSols;
        allSols = Init_Sols(allVars);
        for (int i = 0; i < allSols.size(); i++)
            reverse(allSols[i].begin(), allSols[i].end());
        string path = "/Users/fch/Python/ApproximateMult/ApproxMULT_MyHDL/ApproxMULT_MyHDL/";
        string file = "CompressorTree.json";
        WriteToJson(allSols, path + file);

        system(("python \"" + path + "ApproxMULT_MyHDL.py\" \"" + path + file + "\" ILP_ApproxMult " + to_string(MULT_SIZE) + " " + simulation_path).c_str());
        system(("cd " + simulation_path + "; ./run.sh -w " + to_string(MULT_SIZE) + " -n " + to_string(NTEST) + " -m approx_mult 1>/dev/null").c_str());
        system(("cd " + simulation_path + "; ./sim > output.txt").c_str());
        solution.second = read_med(simulation_path + "output.txt");;
        return 1;
    }
    catch (GRBException e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
        return 0;
    }
    catch (...) {
        cout << "Exception during optimization" << endl;
        return 0;
    }
}

int postProcess_CAOE(pair<double, double>& solution, double Area_bound)
{
    
    vector<vector<int>> F(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
    vector<vector<int>> H(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
    vector<vector<int>> AC32(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
    vector<vector<int>> AC42(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
    vector<vector<int>> V(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));

    // compensate compressors
    try
    {
        filename = "gurobi";
        /*******Set up environment******/
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        // model.set(GRB_StringParam_LogFile, filename + ".log");
        //model.set(GRB_INT_PAR_MIPFOCUS, "1");
        model.set(GRB_INT_PAR_PRESOLVE, "2");
        /*model.set(GRB_DBL_PAR_TIMELIMIT, to_string(Time_Bound_s));*/
        //model.set(GRB_DBL_PAR_HEURISTICS, "0.5");
        //model.set(GRB_DBL_PAR_MIPGAP, "0.005");

        /*******Create variables******/
        //Compressor tree
        vector<vector<GRBVar>> variables_fa, variables_ha;
        vector<vector<GRBVar>> variables_V;
        vector<vector<GRBVar>> variables_ac32, variables_ac42;
        vector<vector<GRBVar>> variables_error;
        GRBVar fa_num, ha_num, ac32_num, ac42_num, E;

        vector<double> MED_ac42(stages_num, double(14) / 256);
        vector<double> MED_ac32(stages_num, double(1) / 64);

        generate_variables_of_multiplier(variables_fa, variables_ha, variables_V, variables_ac32, variables_ac42, variables_error, model);

        /*Add general constraints*/
        //Compressor Tree
        Initial_constraints_of_V(variables_V, model);
        generate_constraints_of_h_f_V(variables_fa, variables_ha, variables_V, variables_ac32, variables_ac42, variables_error, MED_ac42, MED_ac32, model);
        generate_constraints_of_last_stage(variables_V, model);

        // Set objective
        GRBLinExpr obj = 0;
        generate_cost_of_adders(variables_fa, variables_ha, fa_num, ha_num, variables_ac32, variables_ac42, ac32_num, ac42_num, model);
        generate_total_error(variables_error, E, model);
    
        model.addConstr((5.05 * fa_num + 2.66 * ha_num + 2.66 * ac32_num + 4.78 * ac42_num) <= Area_bound);
        obj += E;

        model.setObjective(obj, GRB_MINIMIZE);

        model.write("gurobi.lp");

        model.optimize();

        //Save solution
        model.write(filename + ".sol");

        int nFA, nHA, nAC42, nAC32;
        nFA = fa_num.get(GRB_DoubleAttr_X);
        nHA = ha_num.get(GRB_DoubleAttr_X);
        nAC32 = ac32_num.get(GRB_DoubleAttr_X);
        nAC42 = int(ac42_num.get(GRB_DoubleAttr_X));
        solution.first = (5.05 * nFA + 2.66 * nHA + 2.66 * nAC32 + 4.78 * nAC42);
        cout << fa_num.get(GRB_StringAttr_VarName) << " "
				<< fa_num.get(GRB_DoubleAttr_X) << endl;
        // nFA = fa_num.get(GRB_DoubleAttr_X);
        cout << ha_num.get(GRB_StringAttr_VarName) << " "
            << ha_num.get(GRB_DoubleAttr_X) << endl;
        // nHA = ha_num.get(GRB_DoubleAttr_X);
        cout << ac32_num.get(GRB_StringAttr_VarName) << " "
            << ac32_num.get(GRB_DoubleAttr_X) << endl;
        // nAC32 = ac32_num.get(GRB_DoubleAttr_X);
        cout << ac42_num.get(GRB_StringAttr_VarName) << " "
            << ac42_num.get(GRB_DoubleAttr_X) << endl;
        // nAC42 = int(ac42_num.get(GRB_DoubleAttr_X));
        // cout << E.get(GRB_StringAttr_VarName) << " "
        //     << E.get(GRB_DoubleAttr_X) << endl;
        // update V, F, H, AC42, AC32
        for (int s = 0; s < stages_num; s++) {
            for (int j = 0; j < variables_V[s].size(); j++) {
                V[s][j] = int(round(variables_V[s+1][j].get(GRB_DoubleAttr_X)));
                F[s][j] = int(round(variables_fa[s][j].get(GRB_DoubleAttr_X)));
                H[s][j] = int(round(variables_ha[s][j].get(GRB_DoubleAttr_X)));
                AC32[s][j] = int(round(variables_ac32[s][j].get(GRB_DoubleAttr_X)));
                AC42[s][j] = int(round(variables_ac42[s][j].get(GRB_DoubleAttr_X)));
            }
        }

        // // show the last stage of variables_V
        // vector<GRBVar> last_stage_V = variables_V[variables_V.size() - 1];
        // cout << "Last stage variables_V:" << endl;
        // for (int j = 0; j < last_stage_V.size(); j++)
        //     cout << last_stage_V[j].get(GRB_DoubleAttr_X) << " ";
        // cout << endl;
    }
    catch (GRBException e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
        return 0;
    }
    catch (...) {
        cout << "Exception during optimization" << endl;
        return 0;
    }
    

    // optimize
    filename1 = "gurobi_CO";
    int Time_Bound_s = 10;
    int EP_Approx = 0;
	
    string simulation_path = "/Users/fch/Python/ApproximateMult/ApproxMULT_MyHDL/ApproxMULT_MyHDL/simulation/"; //default simulator path
    int NTEST = (2 * MULT_SIZE > 20) ? (1 << 20) : (1 << (2 * MULT_SIZE)); //default number of tests for simulation
    try {
        /*******Set up environment******/
        GRBEnv env = GRBEnv();
        GRBModel model_CO = GRBModel(env);
        model_CO.set(GRB_StringParam_LogFile, filename + ".log");
        model_CO.set(GRB_INT_PAR_MIPFOCUS, "1");
        model_CO.set(GRB_INT_PAR_NUMERICFOCUS, "1");
        //model_CO.set(GRB_INT_PAR_CUTS, "2");
        //model_CO.set(GRB_INT_PAR_MIQCPMETHOD, "0");
        //model_CO.set(GRB_INT_PAR_NODEMETHOD, "1");
        //model_CO.set(GRB_INT_PAR_PREQLINEARIZE, "1");
        model_CO.set(GRB_INT_PAR_PRESOLVE, "2");
        model_CO.set(GRB_DBL_PAR_TIMELIMIT, to_string(Time_Bound_s));
        model_CO.set(GRB_DBL_PAR_HEURISTICS, "0.4");
        //model_CO.set(GRB_DBL_PAR_MIPGAP, "0.005");
        model_CO.set(GRB_INT_PAR_NONCONVEX, "2");

        vector<vector<Vars_Column>> allVars;
        vector<vector<GRBVar>> allMEDs;
        vector<int> initial_stage = input_patterns;
        // vector<vector<int>> F(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
        // vector<vector<int>> H(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
        // vector<vector<int>> AC32(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
        // vector<vector<int>> AC42(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
        // vector<vector<double>> E(stages_num, vector<double>(2 * MULT_SIZE + stages_num - 1, 0.0));
        // vector<vector<int>> V(stages_num, vector<int>(2 * MULT_SIZE + stages_num - 1, 0));
        // ConstructCT(F, H, AC32, AC42, E, V);
        initial_stage.insert(initial_stage.begin(), stages_num, 0);
        V.insert(V.begin(), initial_stage);
        F.insert(F.end(), vector<int>(F[0].size(), 0));
        H.insert(H.end(), vector<int>(H[0].size(), 0));
        AC32.insert(AC32.end(), vector<int>(AC32[0].size(), 0));
        AC42.insert(AC42.end(), vector<int>(AC42[0].size(), 0));
        Sols sol(V, F, H, AC42, AC32);

        generate_variables_of_ConnectOrderOpt(allVars, sol, model_CO);
        Initial_probs_of_V(allVars[0], model_CO);
        allMEDs = generate_constraints_of_ConnectOrderOpt(allVars, EP_Approx, model_CO);
        // vector<vector<GRBVar>> probs_last_stage;
        // vector<vector<GRBVar>> connect_last_stage(allVars[allVars.size() - 1].size());
        // for (int j = 0; j < allVars[allVars.size() - 1].size(); j++)
        // {
        // 	probs_last_stage.push_back(allVars[allVars.size() - 1][j].Probs);
        // 	for (int k = 0; k < allVars[allVars.size() - 1][j].Probs.size(); k++){
        // 		stringstream ss;
        // 		string s;
        // 		ss << "connect_" << j << "_" << k;
        // 		ss >> s; ss.clear(); ss.str("");
        // 		GRBVar connect_var = model_CO.addVar(0, 1, 0, GRB_BINARY, s);
        // 		connect_last_stage[j].push_back(connect_var);
        // 	}
        // }

        // vector<GRBVar> Errors = generate_constraints_of_ConnectOrderOpt_last_stage(probs_last_stage, connect_last_stage, model_CO);
        GRBLinExpr MED = 0;
        for (int i = 0; i < allMEDs.size(); i++)
            for (int j = 0; j < allMEDs[i].size(); j++)
                MED += allMEDs[i][j] * pow(2, allMEDs[i].size() - j - 1);
        // for (int j = 0; j < Errors.size(); j++)
        // 	MED += Errors[j] * pow(2, Errors.size() - j - 1);
        model_CO.setObjective(MED, GRB_MINIMIZE);

        // Save problem
        model_CO.write(filename1 + ".lp");

        // Optimize model

        model_CO.optimize();

        //Save solution

        model_CO.write(filename1 + ".sol");

        // post verification
        vector<vector<column_sol_t>> allSols;
        allSols = Init_Sols(allVars);
        for (int i = 0; i < allSols.size(); i++)
            reverse(allSols[i].begin(), allSols[i].end());
        string path = "/Users/fch/Python/ApproximateMult/ApproxMULT_MyHDL/ApproxMULT_MyHDL/";
        string file = "CompressorTree.json";
        WriteToJson(allSols, path + file);

        system(("python \"" + path + "ApproxMULT_MyHDL.py\" \"" + path + file + "\" ILP_ApproxMult " + to_string(MULT_SIZE) + " " + simulation_path).c_str());
        system(("cd " + simulation_path + "; ./run.sh -w " + to_string(MULT_SIZE) + " -n " + to_string(NTEST) + " -m approx_mult 1>/dev/null").c_str());
        system(("cd " + simulation_path + "; ./sim > output.txt").c_str());
        solution.second = read_med(simulation_path + "output.txt");;
        return 1;
    }
    catch (GRBException e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
        return 0;
    }
    catch (...) {
        cout << "Exception during optimization" << endl;
        return 0;
    }
}


int main(int argc, char* argv[])
{
    for (int i = 1; i <= 2 * MULT_SIZE - 1; i++)
		input_patterns.push_back(MULT_SIZE - abs(i - MULT_SIZE));
    // 1) Pick the directory to scan
    // pair<double, double> solution;
    // postProcess_CAOE(solution, 600);

    // auto front0 = readFirstFrontier("/Users/fch/Python/OPACT-Extension/Training_log_0.0001_ste_counts/non-dominated-sorting.txt");
    // vector<pair<double,double>> CAOE_solutions;
    // for (auto [f1,f2] : front0) {
    //     pair<double, double> solution;
    //     postProcess_CAOE(solution, f1);
    //     CAOE_solutions.push_back(solution);
    // }

    // cout << "CAOE solutions:\n";
    // for (const auto& sol : CAOE_solutions) {
    //     cout << "(" << sol.first << ", " << sol.second << "), ";
    // }
    // cout << endl;

    fs::path dir = "/Users/fch/Python/OPACT-Extension/Training_log_0.0001_ste_counts"; // Change to your directory
    
    string file = "/Users/fch/Python/OPACT-Extension/Training_log_0.0001_ste_counts/AC_Allocation_6000.json";
    pair<double, double> solution_test, solution_ILP_test;
    vector<pair<double, double>> solutions_test, solutions_ILP_test;
    postProcess_CAOE(solution_ILP_test, 655.65);
    cout<< "ILP solution: " << solution_ILP_test.first << ", " << solution_ILP_test.second << endl;
    if(postProcess(solution_test, file, 0)){
        solutions_test.push_back(solution_test);
        postProcess_CAOE(solution_ILP_test, solution_test.first);
        cout<< "ILP solution: " << solution_ILP_test.first << ", " << solution_ILP_test.second << endl;
    }    
    if(postProcess(solution_test, file, 1)){
        solutions_test.push_back(solution_test);
        CAOE(solution_ILP_test, solution_test.first);
        cout<< "ILP solution: " << solution_ILP_test.first << ", " << solution_ILP_test.second << endl;
    }
    if(postProcess(solution_test, file, 2)){
        solutions_test.push_back(solution_test);
        CAOE(solution_ILP_test, solution_test.first);
        cout<< "ILP solution: " << solution_ILP_test.first << ", " << solution_ILP_test.second << endl;
    }

    vector<pair<double, double>> solutions;
    // 2) Iterate over *all* directory entries (files + sub-dirs)
    for (const fs::directory_entry& entry : fs::directory_iterator{dir})
    {
        if (entry.is_regular_file())        // skip dirs/symlinks if you want files only
        {
            // only process .json files
            if (entry.path().extension() != ".json")
                continue;
            
            pair<double, double> solution;
            
            if(postProcess(solution,entry.path(),0))
                solutions.push_back(solution);
            if(postProcess(solution,entry.path(),1))
                solutions.push_back(solution);
            if(postProcess(solution,entry.path(),2))
                solutions.push_back(solution);
            pair<double, double> solution_ILP;
            CAOE(solution_ILP, solution.first);
            cout<< "ILP solution: " << solution_ILP.first << ", " << solution_ILP.second << endl;
        }
    }

    vector<vector<Point>> fronts = nondominatedSort(solutions);

    std::cout << "Pareto fronts:\n";
    for (std::size_t i = 0; i < fronts.size(); ++i)
    {
        std::cout << "Front " << i << ": ";
        for (const auto& p : fronts[i])
            std::cout << "(" << p.first << ", " << p.second << ") ";
        std::cout << '\n';
    }
    return 0;
}