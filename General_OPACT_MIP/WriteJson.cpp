#include "WriteJson.h"
#include <fstream>
#include <json/json.h>  // JsonCPP

using namespace std;
using namespace Json;

/* ------------------------------- Legacy writer ------------------------------- */
/* This mirrors your former style. Keep if you still generate legacy structures. */
void WriteToJsonLegacy(
    const std::string& path,
    const std::vector<std::vector<int>>& V_out,
    const std::vector<std::vector<int>>& F,
    const std::vector<std::vector<int>>& H,
    const std::vector<std::vector<int>>& AC42,
    const std::vector<std::vector<int>>& AC32)
{
    StyledWriter writer;
    Value root;

    // V_out
    {
        Value arrVout; // array of arrays
        for (const auto& row : V_out) {
            Value line;
            for (int v : row) line.append(v);
            arrVout.append(line);
        }
        root["V_out"] = arrVout;
    }

    auto dump2D = [](const vector<vector<int>>& M) {
        Value A;
        for (const auto& row : M) {
            Value line;
            for (int v : row) line.append(v);
            A.append(line);
        }
        return A;
    };

    root["F"]    = dump2D(F);
    root["H"]    = dump2D(H);
    root["AC42"] = dump2D(AC42);
    root["AC32"] = dump2D(AC32);

    ofstream os(path.c_str());
    os << writer.write(root);
    os.close();
}

/* ------------------------------ Generic writer ------------------------------- */
/* Pass #1 generic (all compressors): counts per type per (stage,col) + V_out. */
void WriteToJsonGeneric(
    const std::string& path,
    const std::vector<std::vector<std::vector<int>>>& compCounts,
    const std::vector<std::vector<int>>& V_out)
{
    StyledWriter writer;
    Value root;

    const int S = (int)compCounts.size();
    const int C = S ? (int)compCounts[0].size() : 0;
    const int T = (S && C) ? (int)compCounts[0][0].size() : 0;

    root["stages"] = S;
    root["cols"]   = C;
    root["types"]  = T;

    // V_out
    {
        Value arrVout;
        for (const auto& row : V_out) {
            Value line;
            for (int v : row) line.append(v);
            arrVout.append(line);
        }
        root["V_out"] = arrVout;
    }

    // compCounts[s][c][t]
    {
        Value CC; // array of stages; each stage is array of columns; each column is array<int> per type
        for (int s = 0; s < S; ++s) {
            Value stageCols;
            for (int c = 0; c < C; ++c) {
                Value types;
                for (int t = 0; t < T; ++t) types.append(compCounts[s][c][t]);
                stageCols.append(types);
            }
            CC.append(stageCols);
        }
        root["compCounts"] = CC;
    }

    ofstream os(path.c_str());
    os << writer.write(root);
    os.close();
}

/* ------------------------- Helpers for CO writer ----------------------------- */

static Value LeftBitToJson(const LeftBit& lb) {
    Value v;
    v["type"] = lb.type;
    v["inst"] = lb.inst;
    v["out"]  = lb.out;
    v["src_stage"] = lb.src_stage;
    v["src_col"]   = lb.src_col;
    return v;
}

static Value PortToJson(const Port& p) {
    Value v;
    v["type"] = p.type;
    v["inst"] = p.inst;
    v["pin"]  = p.pin;
    return v;
}

/* ------------------------- Connection-Order writer --------------------------- */
/* Writes a tree shaped like your former JsonCPP file:
 * {
 *   "stage0": [ { ... per-column ... }, ... ],
 *   "stage1": [ ... ],
 *   ...
 * }
 * Each per-column object contains:
 *   - "n_remain_left", "n_remain_ports"
 *   - "left_bits":       array of {type,inst,out,src_stage,src_col}
 *   - "right_ports":     array of {type,inst,pin}
 *   - "right_to_left":   array<int> (port index -> left index)
 */
void WriteToJsonCO(
    const std::string& path,
    const std::vector<std::vector<column_sol_t>>& allSols,
    const std::vector<int>& last_stage_V)
{
    StyledWriter writer;
    Value CompressorTree; // match your former top-level naming

    const int S = (int)allSols.size();
    const int C = S ? static_cast<int>(allSols[0].size()) : 0;

    for (int s = 0; s < S; ++s) {
        Value Stage; // array of columns

        const int C = (int)allSols[s].size();
        for (int j = 0; j < C; ++j) {
            const auto& cs = allSols[s][j];

            Value Col;

            // Basic counts (similar to your former fields)
            Col["n_remain_left"]  = cs.n_remain_left;
            Col["n_remain_ports"] = cs.n_remain_ports;

            // Left bits array
            {
                Value L;
                L.resize(0u);
                for (const auto& lb : cs.left_bits) {
                    L.append(LeftBitToJson(lb));
                }
                Col["left_bits"] = L;
            }

            // Right ports array
            {
                Value R;
                R.resize(0u);
                for (const auto& rp : cs.right_ports) {
                    R.append(PortToJson(rp));
                }
                Col["right_ports"] = R;
            }

            // right_to_left mapping (one entry per right port)
            {
                Value M;
                M.resize(0u);
                for (int u : cs.right_to_left) M.append(u);
                Col["right_to_left"] = M;
            }

            Stage.append(Col);
        }

        // Key name "stageX" to mirror your former file
        std::string key = "stage" + std::to_string(s);
        CompressorTree[key] = Stage;
    }

    // // 2) Append synthetic final stage "stageS" using REMAIN ports (type=-1)
    // {
    //     Value StageS;
    //     for (int j = 0; j < C; ++j) {
    //         const int n = (j < static_cast<int>(last_stage_V.size())) ? last_stage_V[j] : 0;

    //         Value Col;
    //         Col["n_remain_left"]  = n;
    //         Col["n_remain_ports"] = n;   // one remain port per remaining bit

    //         // left_bits: one per remaining bit
    //         {
    //             Value L;
    //             L.resize(0u);
    //             for (int u = 0; u < n; ++u) {
    //                 Value o;
    //                 o["type"]      = -1;      // remain bit on the left side
    //                 o["inst"]      = u;       // stable index
    //                 o["out"]       = -1;
    //                 o["src_stage"] = S - 1;   // last real stage
    //                 o["src_col"]   = j;
    //                 L.append(o);
    //             }
    //             Col["left_bits"] = L;
    //         }

    //         // right_ports: remain ports, indexed to match left_bits[u]
    //         {
    //             Value R;
    //             R.resize(0u);
    //             for (int u = 0; u < n; ++u) {
    //                 Value o;
    //                 o["type"] = -1;   // remain port
    //                 o["inst"] = u;    // <-- changed here
    //                 o["pin"]  = -1;
    //                 R.append(o);
    //             }
    //             Col["right_ports"] = R;
    //         }

    //         // Identity mapping: port k consumes left_bit k
    //         {
    //             Value M;
    //             M.resize(0u);
    //             for (int u = 0; u < n; ++u) M.append(u);
    //             Col["right_to_left"] = M;
    //         }

    //         StageS.append(Col);
    //     }
    //     CompressorTree["stage" + std::to_string(S)] = StageS;
    // }

    ofstream os(path.c_str());
    os << writer.write(CompressorTree);
    os.close();
}
