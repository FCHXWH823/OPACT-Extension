#include "ProbEval.h"
#include <json/json.h>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include "CompressorLib.h"


// std::vector<CompressorSpec> COMPRESSOR_LIB;
// std::unordered_map<std::string,int> COMPRESSOR_INDEX;
std::vector<TruthTables> COMP_TT;

void load_truth_tables_from_json(const std::string& jsonPath) {
    COMP_TT.assign(COMPRESSOR_LIB.size(), TruthTables{});
    Json::Value root;
    std::ifstream ifs(jsonPath);
    if (!ifs.good()) return;  // library was already loaded; skip silently
    ifs >> root;
    for (auto it = root.begin(); it != root.end(); ++it) {
        const std::string name = it.key().asString();
        auto f = COMPRESSOR_INDEX.find(name);
        if (f == COMPRESSOR_INDEX.end()) continue;
        const int t = f->second;
        if ((*it).isMember("truth_s") && (*it).isMember("truth_c")) {
            const auto& js = (*it)["truth_s"];
            const auto& jc = (*it)["truth_c"];
            TruthTables tt;
            tt.s.reserve(js.size());
            tt.c.reserve(jc.size());
            for (auto &v : js) tt.s.push_back(v.asInt());
            for (auto &v : jc) tt.c.push_back(v.asInt());
            COMP_TT[t] = std::move(tt);
        }
    }
}

static double prob_of_pattern(const std::vector<double>& p, int mask) {
    // p[k] is Pr[x_k=1] for input position k (0..w-1)
    // mask bit k = x_k
    double r = 1.0;
    for (int k = 0; k < (int)p.size(); ++k) {
        const bool bit = (mask >> k) & 1;
        r *= bit ? p[k] : (1.0 - p[k]);
    }
    return r;
}

void tt_E_SC(const TruthTables& tt, const std::vector<double>& inputs, double& ES, double& EC) {
    const int W = (int)inputs.size();
    if (!tt.valid() || (int)tt.s.size() != (1<<W))
        throw std::runtime_error("Truth table size does not match compressor width.");
    ES = 0.0; EC = 0.0;
    for (int mask = 0; mask < (1<<W); ++mask) {
        const double pm = prob_of_pattern(inputs, mask);
        ES += tt.s[mask] * pm;
        EC += tt.c[mask] * pm;
    }
}

/* --- Exact compressors --- */

void exact_E_SC_EC22(double p1, double p2, double& ES, double& EC) {
    // HA: S = x1 XOR x2, C = x1 & x2
    EC = p1 * p2;
    ES = p1 + p2 - 2.0 * p1 * p2;
}

void exact_E_SC_EC32(double p1, double p2, double p3, double& ES, double& EC) {
    // FA: C = majority(x1,x2,x3) = sum p_i p_j - 2 p1 p2 p3
    const double s2 = p1*p2 + p1*p3 + p2*p3;
    const double s3 = p1*p2*p3;
    EC = s2 - 2.0 * s3;
    // S = parity(x1,x2,x3) = sum p_i - 2 sum p_i p_j + 4 p1 p2 p3
    const double s1 = p1 + p2 + p3;
    ES = s1 - 2.0*s2 + 4.0*s3;
}

// int main() {
//     try {
//         load_compressor_lib("/Users/fch/Python/OPACT-Extension/ACs/ac_lib.json");
//         load_truth_tables_from_json("/Users/fch/Python/OPACT-Extension/ACs/ac_lib.json");
//         for (const auto& tt : COMP_TT) {
//             if (tt.valid()) {
//                 // Print the name of the compressor and its truth table
//                 int index = &tt - &COMP_TT[0];  // Get index of the truth table
//                 std::cout << "Compressor name: " << COMPRESSOR_LIB[index].name << " ";
//                 std::cout << "Truth table: S = [";
//                 for (int s : tt.s) std::cout << s << " ";
//                 std::cout << "], C = [";
//                 for (int c : tt.c) std::cout << c << " ";
//                 std::cout << "]\n";
//             }
//         }
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }
//     return 0;
// }
