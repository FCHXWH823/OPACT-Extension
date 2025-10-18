#include "Simulator.h"

void simulator_ES_EC_for_type(int type_index, const std::vector<double>& p_inputs, double& ES, double& EC) {
    const auto& spec = COMPRESSOR_LIB[type_index];
    if (spec.name == "EC22") {
        exact_E_SC_EC22(p_inputs[0], p_inputs[1], ES, EC);
        return;
    }
    if (spec.name == "EC32") {
        exact_E_SC_EC32(p_inputs[0], p_inputs[1], p_inputs[2], ES, EC);
        return;
    }
    const auto& tt = COMP_TT[type_index];
    if (tt.valid()) { tt_E_SC(tt, p_inputs, ES, EC); return; }
    // fallback: constants
    ES = spec.err_val; EC = 0.0;
}
