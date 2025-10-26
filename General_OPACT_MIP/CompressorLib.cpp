#include "CompressorLib.h"
#include <iostream>
#include <json/json.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

std::vector<CompressorSpec> COMPRESSOR_LIB;
std::unordered_map<std::string,int> COMPRESSOR_INDEX;

static double to_double(const std::string& s) {
    std::stringstream ss(s);
    double v; ss >> v;
    return v;
}

void load_compressor_lib(const std::string& jsonPath) {
    COMPRESSOR_LIB.clear();
    COMPRESSOR_INDEX.clear();

    Json::Value root;
    std::ifstream ifs(jsonPath);
    if (!ifs.good())
        throw std::runtime_error("Cannot open compressor library JSON: " + jsonPath);
    ifs >> root;

    for (auto it = root.begin(); it != root.end(); ++it) {
        CompressorSpec spec;
        spec.name = it.key().asString();
        spec.width = (*it)["width"].asInt();
        spec.area  = (*it)["area"].asDouble();
        // err_val stored as string in JSON
        spec.err_val = to_double((*it)["err_val"].asString());
        spec.same_col = (spec.name.find("ew") != std::string::npos);
        COMPRESSOR_INDEX[spec.name] = (int)COMPRESSOR_LIB.size();
        COMPRESSOR_LIB.push_back(spec);
    }
    if (COMPRESSOR_LIB.empty())
        throw std::runtime_error("Empty compressor library.");
}

double stage_scale(const CompressorSpec& c, int stage) {
    // matches your previous scaling: width=3 -> 1.5^i, width=4 -> 2^i, width=2 -> 1^i
    return std::pow(double(c.width)/2.0, stage);
}


// int main() {
//     try {
//         load_compressor_lib("/Users/fch/Python/OPACT-Extension/ACs/ac_lib.json");
//         for (const auto& spec : COMPRESSOR_LIB) {
//             std::cout << "Compressor: " << spec.name
//                       << ", Width: " << spec.width
//                       << ", Area: " << spec.area
//                       << ", Error Value: " << spec.err_val
//                       << ", Same Column: " << (spec.same_col ? "Yes" : "No") << std::endl;
//         }
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }
//     return 0;
// }