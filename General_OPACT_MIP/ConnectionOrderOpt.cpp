#include "ConnectionOrderOpt.h"
#include "CompressorLib.h"   // COMPRESSOR_LIB, COMPRESSOR_INDEX
#include "ProbEval.h"        // COMP_TT, TruthTables
#include "gurobi_c++.h"

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <numeric>

using std::vector;
using std::string;

/* ------------------------------------------------------------------------- */
/*                         Left / Right layouts per (s,j)                    */
/* ------------------------------------------------------------------------- */

/* Each left-side bit feeding (stage s, column j). We keep provenance. */
// struct LeftBit {
//     int   type   = -1;   // -1 for remain
//     int   inst   = -1;   // instance index within that type at source stage/column
//     int   out    = -1;   // 0 or 1 for compressor outputs; -1 for remain
//     int   src_stage = -1; // stage where this bit was produced (typically s-1)
//     int   src_col   = -1; // column where this bit was produced (j or j-1 for carry-in)
// };

struct LeftLayout {
    int n_slots = 0;
    int n_remain = 0;
    vector<LeftBit> bits;        // every left-side bit in order
};

/* Build the left-side list for (stage s, column j):
 * Order = Remain(j) → Local outputs from (s-1, j) → Incoming carries from (s-1, j-1)
 * Types are iterated in ORIGINAL library order (index order).
 */
static LeftLayout build_left_layout(
    const vector<vector<vector<int>>>& compCounts, // [stage][col][type]
    const vector<vector<int>>& V_out,              // [stage][col] := bits AFTER stage
    const vector<int>& init_bits_col,              // initial PP bits per column (stage 0), MSB-first
    int s, int j, int nCols)
{
    const int T = (int)COMPRESSOR_LIB.size();

    // Count local outputs at (s-1, j) and incoming carries from (s-1, j+1)
    int locals_sum = 0, carries_sum = 0;
    vector<int> local_cnt(T, 0), carry_cnt(T, 0);

    // Local outputs produced at (s-1, j)
    if (s - 1 >= 0) {
        for (int id = 0; id < T; ++id) {
            const auto& spec = COMPRESSOR_LIB[id];
            int n = compCounts[s-1][j][id];
            if (n <= 0) continue;
            if (spec.same_col) { local_cnt[id] = 2 * n; }   // both outputs stay at j
            else               { local_cnt[id] = 1 * n; }   // only sum stays at j
            locals_sum += local_cnt[id];
        }
    }

    // Incoming carries to column j come from column (j+1) of stage (s-1) for non-same_col types (MSB-first)
    if (s - 1 >= 0 && j + 1 < nCols) {
        for (int id = 0; id < T; ++id) {
            const auto& spec = COMPRESSOR_LIB[id];
            int n = compCounts[s-1][j+1][id];
            if (n <= 0 || spec.same_col) continue;          // carries only for non-same_col
            carry_cnt[id] = n;                              // one carry per instance arrives at j
            carries_sum += carry_cnt[id];
        }
    }

    // At s==0 use initial PP bits; otherwise use outputs of stage s-1
    int bits_prev = (s == 0 ? init_bits_col[j] : V_out[s-1][j]);

    int nRemain = bits_prev - (locals_sum + carries_sum);
    if (nRemain < 0) nRemain = 0;

    LeftLayout L;
    L.n_remain = nRemain;

    // 1) Remain first
    for (int r = 0; r < nRemain; ++r) {
        LeftBit lb; lb.type = -1; lb.inst = r; lb.out = -1;
        lb.src_stage = (s > 0 ? s-1 : -1);
        lb.src_col   = j;
        L.bits.push_back(lb);
    }

    // 2) Local outputs from (s-1, j) in original type order
    if (s - 1 >= 0) {
        for (int id = 0; id < T; ++id) {
            const auto& spec = COMPRESSOR_LIB[id];
            int n = compCounts[s-1][j][id];
            if (n <= 0) continue;
            if (spec.same_col) {
                for (int k = 0; k < n; ++k) {
                    // two outputs: out=0 and out=1 both at column j
                    for (int out = 0; out < 2; ++out) {
                        LeftBit lb; lb.type = id; lb.inst = k; lb.out = out;
                        lb.src_stage = s-1; lb.src_col = j;
                        L.bits.push_back(lb);
                    }
                }
            } else {
                // only sum stays locally (out=0)
                for (int k = 0; k < n; ++k) {
                    LeftBit lb; lb.type = id; lb.inst = k; lb.out = 0;
                    lb.src_stage = s-1; lb.src_col = j;
                    L.bits.push_back(lb);
                }
            }
        }
    }

    // 3) Incoming carries from (s-1, j+1) for non-same_col types (MSB-first)
    if (s - 1 >= 0 && j + 1 < nCols) {
        for (int id = 0; id < T; ++id) {
            const auto& spec = COMPRESSOR_LIB[id];
            int n = compCounts[s-1][j+1][id];
            if (n <= 0 || spec.same_col) continue;
            for (int k = 0; k < n; ++k) {
                LeftBit lb; lb.type = id; lb.inst = k; lb.out = 1;  // carry output
                lb.src_stage = s-1; lb.src_col = j+1;               // arrived from less significant column
                L.bits.push_back(lb);
            }
        }
    }

    L.n_slots = (int)L.bits.size();
    return L;
}


struct RightLayout {
    int n_ports = 0;
    int n_remain_ports = 0;
    // struct Port { int type; int inst; int pin; }; // type id, instance index, pin index
    vector<Port> ports;
};

static RightLayout build_right_layout(
    const vector<vector<vector<int>>>& compCounts,
    int s, int j, int left_slots_count)
{
    RightLayout R;
    const int T = (int)COMPRESSOR_LIB.size();

    int needed_inputs = 0;
    for (int t = 0; t < T; ++t)
        needed_inputs += COMPRESSOR_LIB[t].width * compCounts[s][j][t];

    int nRemainPorts = left_slots_count - needed_inputs;
    if (nRemainPorts < 0) nRemainPorts = 0;

    R.n_remain_ports = nRemainPorts;
    R.ports.clear();

    // Remain ports first
    for (int r = 0; r < nRemainPorts; ++r)
        R.ports.push_back({-1, r, -1}); // pass-through

    // Per-type instance ports in ORIGINAL order
    for (int id = 0; id < T; ++id) {
        int cnt = compCounts[s][j][id];
        if (cnt <= 0) continue;
        int w = COMPRESSOR_LIB[id].width;
        for (int k = 0; k < cnt; ++k)
            for (int pin = 0; pin < w; ++pin)
                R.ports.push_back({id, k, pin});
    }
    R.n_ports = (int)R.ports.size();
    return R;
}

/* ------------------------------------------------------------------------- */
/*           Pair-term construction and per-pattern probability vars         */
/* ------------------------------------------------------------------------- */

// Pair terms for (p_i, p_j): z11 = p_i*p_j (bilinear); z10, z01, z00 are linear definitions.
struct PairTerms { GRBVar z11, z10, z01, z00; };

static PairTerms make_pair_terms(GRBModel& model, const GRBVar& pi, const GRBVar& pj,
                                 const string& base, int i, int j, int uid) {
    PairTerms pt;
    std::ostringstream nm11; nm11 << base << "_z11_" << uid << "_" << i << "_" << j;
    pt.z11 = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, nm11.str());
    model.addQConstr(pt.z11 == pi * pj, nm11.str() + "_eq"); // requires NonConvex=2

    std::ostringstream nm10; nm10 << base << "_z10_" << uid << "_" << i << "_" << j;
    std::ostringstream nm01; nm01 << base << "_z01_" << uid << "_" << i << "_" << j;
    std::ostringstream nm00; nm00 << base << "_z00_" << uid << "_" << i << "_" << j;
    pt.z10 = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, nm10.str());
    pt.z01 = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, nm01.str());
    pt.z00 = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, nm00.str());
    model.addConstr(pt.z10 == pi - pt.z11, nm10.str() + "_def");
    model.addConstr(pt.z01 == pj - pt.z11, nm01.str() + "_def");
    model.addConstr(pt.z00 == 1.0 - pi - pj + pt.z11, nm00.str() + "_def");
    return pt;
}

static GRBVar select_pair_term(const PairTerms& pt, int b0, int b1) {
    if (b0==0 && b1==0) return pt.z00;
    if (b0==0 && b1==1) return pt.z01;
    if (b0==1 && b1==0) return pt.z10;
    return pt.z11; // 1,1
}

/* Build exact E[f] by summing per-pattern probabilities:
 *  - W=2: P(pattern) is the selected pair term (no extra product).
 *  - W=3: P(b0b1b2) = term01(b0,b1) * (b2 ? p2 : 1-p2)   → one bilinear equality per pattern.
 *  - W=4: P(b0b1b2b3) = term01(b0,b1) * term23(b2,b3)   → one bilinear equality per pattern.
 */
static GRBQuadExpr expectation_from_patterns_exact(
    const vector<int>& f,                 // 2^W truth table (0/1) for one output bit
    const vector<GRBVar>& p,              // input probabilities
    GRBModel& model,
    const string& base, int uid)
{
    const int W = (int)p.size();
    GRBQuadExpr E = 0;

    if (W == 2) {
        PairTerms pt01 = make_pair_terms(model, p[0], p[1], base, 0, 1, uid);
        for (int b0 = 0; b0 <= 1; ++b0)
            for (int b1 = 0; b1 <= 1; ++b1) {
                int m = (b0<<0) | (b1<<1);
                if (!f[m]) continue;
                GRBVar term = select_pair_term(pt01, b0, b1);
                E += term;
            }
        return E;
    }

    if (W == 3) {
        PairTerms pt01 = make_pair_terms(model, p[0], p[1], base, 0, 1, uid);
        for (int b0 = 0; b0 <= 1; ++b0)
            for (int b1 = 0; b1 <= 1; ++b1)
                for (int b2 = 0; b2 <= 1; ++b2) {
                    int m = (b0<<0) | (b1<<1) | (b2<<2);
                    if (!f[m]) continue;

                    GRBVar t12 = select_pair_term(pt01, b0, b1);
                    GRBLinExpr t3 = b2 ? p[2] : (1.0 - p[2]);

                    std::ostringstream nm; nm << base << "_pat_" << uid << "_" << b0 << b1 << b2;
                    GRBVar mb = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, nm.str());
                    model.addQConstr(mb == t12 * t3, nm.str() + "_eq");
                    E += mb;
                }
        return E;
    }

    if (W == 4) {
        PairTerms pt01 = make_pair_terms(model, p[0], p[1], base, 0, 1, uid);
        PairTerms pt23 = make_pair_terms(model, p[2], p[3], base, 2, 3, uid);

        for (int b0 = 0; b0 <= 1; ++b0)
            for (int b1 = 0; b1 <= 1; ++b1)
                for (int b2 = 0; b2 <= 1; ++b2)
                    for (int b3 = 0; b3 <= 1; ++b3) {
                        int m = (b0<<0) | (b1<<1) | (b2<<2) | (b3<<3);
                        if (!f[m]) continue;

                        GRBVar t01 = select_pair_term(pt01, b0, b1);
                        GRBVar t23 = select_pair_term(pt23, b2, b3);

                        std::ostringstream nm; nm << base << "_pat_" << uid
                                                  << "_" << b0 << b1 << b2 << b3;
                        GRBVar mb = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, nm.str());
                        model.addQConstr(mb == t01 * t23, nm.str() + "_eq");
                        E += mb;
                    }
        return E;
    }

    // Unexpected widths: no contribution
    return 0;
}

/* ------------------------------------------------------------------------- */
/*           Per-type evaluation: E[S], E[C], MED (signed, with same_col)    */
/* ------------------------------------------------------------------------- */

struct TTResult {
    GRBQuadExpr out0;   // E[S]
    GRBQuadExpr out1;   // E[C]
    GRBQuadExpr med;    // (E[S] + (same_col?1:2) * E[C]) - Σ p_i
};

static TTResult eval_outputs_and_med_from_lib(const string& type_name,
                                              const vector<GRBVar>& in_probs,
                                              const bool same_col,
                                              GRBModel& model,
                                              const string& tag = "tt")
{
    TTResult r; r.out0 = 0; r.out1 = 0; r.med = 0;

    auto it = COMPRESSOR_INDEX.find(type_name);
    if (it == COMPRESSOR_INDEX.end())
        throw std::runtime_error("Unknown compressor type: " + type_name);
    int tid = it->second;
    const TruthTables& tt = COMP_TT[tid];

    const int W = (int)in_probs.size();
    const int need = 1 << W;
    if (!tt.valid() || (int)tt.s.size() != need || (int)tt.c.size() != need) {
        std::ostringstream oss;
        oss << "Truth table size mismatch for " << type_name << " (W=" << W << ")";
        throw std::runtime_error(oss.str());
    }

    static int g_uid = 0; int uid = g_uid++;

    // Exact E[S], E[C] via per-pattern summation (order ≤ 2 with our pair-term scheme)
    r.out0 = expectation_from_patterns_exact(tt.s, in_probs, model, tag + "_S_" + type_name, uid);
    r.out1 = expectation_from_patterns_exact(tt.c, in_probs, model, tag + "_C_" + type_name, uid);

    // signed MED depends on same_col (carry weight 1 vs 2)
    const double carry_weight = same_col ? 1.0 : 2.0;
    GRBLinExpr ideal_sum = 0;
    for (const auto& pi : in_probs) ideal_sum += pi;
    r.med = ideal_sum - (r.out0 + carry_weight * r.out1);

    return r;
}

/* ------------------------------------------------------------------------- */
/*                            Main CO MILP                                   */
/* ------------------------------------------------------------------------- */

ConnectionOrderResult optimize_connection_order(
    const vector<vector<vector<int>>>& compCounts, // [stage][col][type]
    const vector<vector<int>>& V_out,              // [stage][col] AFTER stage
    const vector<int>& init_bits_col,              // initial PP bits per column (stage 0), MSB-first
    const vector<double>& bitW,                    // MSB-first weights per column
    const string& run_tag)
{
    ConnectionOrderResult res{};
    const int S = (int)compCounts.size();
    if (S == 0) return res;
    const int C = (int)compCounts[0].size();
    const int T = (int)COMPRESSOR_LIB.size();

    try {
        GRBEnv env;
        GRBModel model(env);
        model.set(GRB_IntParam_Presolve, 2);
        model.set(GRB_IntParam_NonConvex, 2);   // allow z == a*b constraints
        model.set(GRB_DBL_PAR_TIMELIMIT, "20");
        GRBQuadExpr total_signed_err = 0;       // Σ bitW[j] * MED_instance

        // --- Stage-to-stage probability propagation containers (MSB-first) ---
        // prev_* hold outputs produced at stage s-1, to feed as prob_prev at stage s.
        // Indexing is MSB→LSB: column 0 = MSB.
        vector<vector<GRBVar>> prev_pass(C), prev_local(C), prev_carry_in(C);

         // Keep what we need to reconstruct mapping post-solve
        struct CellVars {
        LeftLayout L;
        RightLayout R;
        vector<vector<GRBVar>> sel; // [left][right]
        };
        vector<vector<CellVars>> cells(S, vector<CellVars>(C));

        for (int s = 0; s < S; ++s) {

            // Will collect outputs of stage s for feeding stage s+1
            vector<vector<GRBVar>> cur_pass(C), cur_local(C), cur_carry_to_col(C);

            for (int j = 0; j < C; ++j) {

                // 1) Layouts (must enumerate carries for (s,j) from (s-1, j+1))
                LeftLayout L = build_left_layout(compCounts, V_out, init_bits_col, s, j, C);
                RightLayout R = build_right_layout(compCounts, s, j, L.n_slots);

                // 2) Mapping vars: sel[u][p] ∈ {0,1}
                vector<vector<GRBVar>> sel(L.n_slots, vector<GRBVar>(R.n_ports));
                for (int u = 0; u < L.n_slots; ++u)
                    for (int p = 0; p < R.n_ports; ++p) {
                        std::ostringstream nm;
                        nm << "sel_s" << s << "_j" << j << "_u" << u << "_p" << p;
                        sel[u][p] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, nm.str());
                    }
                
                // Save for reconstruction
                cells[s][j].L = L;
                cells[s][j].R = R;
                cells[s][j].sel = sel;

                // 3) Last-stage output probabilities at this (s,j) — one per left bit
                vector<GRBVar> prob_prev(L.n_slots);
                for (int u = 0; u < L.n_slots; ++u) {
                    std::ostringstream nm; nm << "p_prev_s" << s << "_j" << j << "_u" << u;
                    prob_prev[u] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, nm.str());
                }

                // 3a) Assign prob_prev from previous stage outputs (or seed at s==0)
                if (s == 0) {
                    // Seed all left bits to 0.25 at stage 0
                    for (int u = 0; u < L.n_slots; ++u)
                        model.addConstr(prob_prev[u] == 0.25,
                                        "seed_s0_j"+std::to_string(j)+"_u"+std::to_string(u));
                } else {
                    int rem_i = 0, loc_i = 0, car_i = 0;
                    for (int u = 0; u < L.n_slots; ++u) {
                        const LeftBit& lb = L.bits[u];
                        if (lb.type == -1) {
                            // Remain bit from previous stage pass-throughs at (s-1, j)
                            if (rem_i >= (int)prev_pass[j].size())
                                throw std::runtime_error("prev_pass size mismatch at s="+std::to_string(s)+" j="+std::to_string(j));
                            model.addConstr(prob_prev[u] == prev_pass[j][rem_i++],
                                            "prev_rem_s"+std::to_string(s)+"_j"+std::to_string(j)+"_u"+std::to_string(u));
                        } else if (lb.src_col == j) {
                            // Local output from (s-1, j)
                            if (loc_i >= (int)prev_local[j].size())
                                throw std::runtime_error("prev_local size mismatch at s="+std::to_string(s)+" j="+std::to_string(j));
                            model.addConstr(prob_prev[u] == prev_local[j][loc_i++],
                                            "prev_loc_s"+std::to_string(s)+"_j"+std::to_string(j)+"_u"+std::to_string(u));
                        } else if (lb.src_col == j + 1) {
                            // Incoming carry from (s-1, j+1) landing at j (MSB-first indexing)
                            if (car_i >= (int)prev_carry_in[j].size())
                                throw std::runtime_error("prev_carry size mismatch at s="+std::to_string(s)+" j="+std::to_string(j));
                            model.addConstr(prob_prev[u] == prev_carry_in[j][car_i++],
                                            "prev_car_s"+std::to_string(s)+"_j"+std::to_string(j)+"_u"+std::to_string(u));
                        } else {
                            throw std::runtime_error("Left bit src_col not in {j, j+1}.");
                        }
                    }
                }

                // 4) Mapping constraints
                for (int u = 0; u < L.n_slots; ++u) {
                    GRBLinExpr sum = 0;
                    for (int p = 0; p < R.n_ports; ++p) sum += sel[u][p];
                    model.addConstr(sum == 1, "one_port_s"+std::to_string(s)+"_j"+std::to_string(j)+"_u"+std::to_string(u));
                }
                for (int p = 0; p < R.n_ports; ++p) {
                    GRBLinExpr sum = 0;
                    for (int u = 0; u < L.n_slots; ++u) sum += sel[u][p];
                    model.addConstr(sum == 1, "port_cap_s"+std::to_string(s)+"_j"+std::to_string(j)+"_p"+std::to_string(p)); // <= 1
                }

                // 5) Pass-through ports (remain ports) → collect as outputs for next stage
                for (int p = 0; p < R.n_remain_ports; ++p) {
                    GRBVar pass = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS,
                                               "pt_s"+std::to_string(s)+"_j"+std::to_string(j)+"_p"+std::to_string(p));
                    GRBQuadExpr rhs = 0;
                    for (int u = 0; u < L.n_slots; ++u) rhs += sel[u][p] * prob_prev[u]; // bilinear
                    model.addQConstr(pass == rhs, "pt_def_s"+std::to_string(s)+"_j"+std::to_string(j)+"_p"+std::to_string(p));
                    cur_pass[j].push_back(pass);   // remains at same column j for next stage
                }

                // 6) Per-instance wiring and truth-table evaluation (ORIGINAL type order)
                int p_idx = R.n_remain_ports;
                for (int id = 0; id < T; ++id) {
                    int cnt = compCounts[s][j][id];
                    if (cnt <= 0) continue;
                    const auto& spec = COMPRESSOR_LIB[id];
                    const int w = spec.width;

                    for (int k = 0; k < cnt; ++k) {
                        // Build input probabilities as convex combos of left bits
                        vector<GRBVar> in_probs(w);
                        for (int pin = 0; pin < w; ++pin) {
                            std::ostringstream nm;
                            nm << "p_in_s" << s << "_j" << j << "_t" << id << "_k" << k << "_pin" << pin;
                            in_probs[pin] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, nm.str());

                            GRBQuadExpr rhs = 0;
                            for (int u = 0; u < L.n_slots; ++u)
                                rhs += sel[u][p_idx + pin] * prob_prev[u]; // bilinear
                            model.addQConstr(in_probs[pin] == rhs,
                                "in_def_s"+std::to_string(s)+"_j"+std::to_string(j)+"_t"+std::to_string(id)+"_k"+std::to_string(k)+"_pin"+std::to_string(pin));
                        }
                        p_idx += w;

                        // Evaluate outputs & MED
                        TTResult tt = eval_outputs_and_med_from_lib(spec.name, in_probs, spec.same_col,
                                        model, "s"+std::to_string(s)+"_j"+std::to_string(j));

                        // Bit-weighted signed error (bitW is MSB-first)
                        total_signed_err += bitW[j] * tt.med;

                        // --- Materialize outputs for next-stage chaining ---
                        // Sum output stays at column j
                        GRBVar p_sum = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS,
                                                    "p_sum_s"+std::to_string(s)+"_j"+std::to_string(j)+"_t"+std::to_string(id)+"_k"+std::to_string(k));
                        model.addQConstr(p_sum == tt.out0, "p_sum_def_s"+std::to_string(s)+"_j"+std::to_string(j)+"_t"+std::to_string(id)+"_k"+std::to_string(k));
                        cur_local[j].push_back(p_sum);

                        if (spec.same_col) {
                            // Carry also remains in the same column j
                            GRBVar p_car = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS,
                                                        "p_carry_s"+std::to_string(s)+"_j"+std::to_string(j)+"_t"+std::to_string(id)+"_k"+std::to_string(k));
                            model.addQConstr(p_car == tt.out1, "p_carry_def_s"+std::to_string(s)+"_j"+std::to_string(j)+"_t"+std::to_string(id)+"_k"+std::to_string(k));
                            cur_local[j].push_back(p_car);
                        } else {
                            // **MSB-first**: carry goes to the *more significant* column j-1
                            if (j - 1 >= 0) {
                                GRBVar p_car = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS,
                                                            "p_carry_to_"+std::to_string(j-1)+"_from_s"+std::to_string(s)+"_j"+std::to_string(j)+"_t"+std::to_string(id)+"_k"+std::to_string(k));
                                model.addQConstr(p_car == tt.out1, "p_carry_to_def_"+std::to_string(j-1)+"_from_s"+std::to_string(s)+"_j"+std::to_string(j)+"_t"+std::to_string(id)+"_k"+std::to_string(k));
                                cur_carry_to_col[j-1].push_back(p_car); // lands at col (j-1) in next stage
                            }
                        }
                    } // k
                } // id

            } // for j

            // --- Stage s finished: make these the previous outputs for stage s+1 ---
            prev_pass     = std::move(cur_pass);
            prev_local    = std::move(cur_local);
            prev_carry_in = std::move(cur_carry_to_col);
        } // for s

        // 7) Objective: minimize absolute value of global signed error
        GRBVar TotalErrorSigned = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "TotalErrorSigned");
        GRBVar AbsError         = model.addVar(0.0, GRB_INFINITY, 0, GRB_CONTINUOUS, "AbsError");
        model.addQConstr(TotalErrorSigned == total_signed_err, "TotalErrorDef");
        model.addConstr(AbsError >=  TotalErrorSigned, "Abs_ge_pos");
        model.addConstr(AbsError >= -TotalErrorSigned, "Abs_ge_neg");
        GRBLinExpr MED = AbsError;
        model.setObjective(MED, GRB_MINIMIZE);

        model.write(run_tag + ".lp");
        model.optimize();
        model.write(run_tag + ".sol");

        res.signed_error = TotalErrorSigned.get(GRB_DoubleAttr_X);
        res.abs_error    = AbsError.get(GRB_DoubleAttr_X);

        /* ---------------------------- Reconstruct mapping ---------------------------- */
        res.allSols.assign(S, std::vector<column_sol_t>(C));
        for (int s=0; s<S; ++s) {
            for (int j=0; j<C; ++j) {
                const auto& cell = cells[s][j];

                column_sol_t cs;
                cs.stage = s;
                cs.col   = j;
                cs.n_remain_left  = cell.L.n_remain;
                cs.n_remain_ports = cell.R.n_remain_ports;

                cs.left_bits   = cell.L.bits;  // preserve exact order used by MILP
                cs.right_ports = cell.R.ports;

                cs.right_to_left.assign(cell.R.n_ports, -1);
                for (int p=0; p<cell.R.n_ports; ++p) {
                int u_star = -1;
                for (int u=0; u<cell.L.n_slots; ++u) {
                    if (cell.sel[u][p].get(GRB_DoubleAttr_X) > 0.5) { u_star = u; break; }
                }
                if (u_star < 0) {
                    std::ostringstream oss;
                    oss << "CO reconstruction failed at (s=" << s << ", j=" << j << ", p=" << p << ")";
                    throw std::runtime_error(oss.str());
                }
                cs.right_to_left[p] = u_star;
                }

                res.allSols[s][j] = std::move(cs);
            }
        }

    } catch (GRBException& e) {
        std::cerr << "[CO] Gurobi exception: " << e.getMessage() << std::endl;
    } catch (std::exception& e) {
        std::cerr << "[CO] Exception: " << e.what() << std::endl;
    }

    return res;
}
