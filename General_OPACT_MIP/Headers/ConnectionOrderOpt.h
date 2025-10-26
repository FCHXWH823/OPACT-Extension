#pragma once
#include <string>
#include <vector>

/* ------------- Basic tokens used on both sides of the mapping ----------------
 * We keep them here so they can be reused by other modules (e.g., JSON writer).
 * - LeftBit describes *a bit* that arrives to (stage s, column j), with provenance.
 * - Port    describes *a right-side input port* at (stage s, column j).
 */
struct LeftBit {
  int type = -1;      // -1 = remain (pass-through from previous stage)
  int inst = -1;      // compressor instance index at the source stage/column
  int out  = -1;      // 0=sum, 1=carry (for compressors); -1 for remain
  int src_stage = -1; // stage where this bit was produced (s-1 for s>0, else -1)
  int src_col   = -1; // column where this bit was produced (MSB-first indexing)
};

struct Port {
  int type = -1;  // -1 = remain port; otherwise compressor type id (index in COMPRESSOR_LIB)
  int inst = -1;  // instance index for that type at (s,j); -1 for remain
  int pin  = -1;  // input pin index [0..width-1] for compressor; -1 for remain
};

/* Column solution: exact objects and orders the MILP used.
 * right_to_left[p] gives the index u of the driving left bit for port p.
 */
struct column_sol_t {
  int stage = -1;
  int col   = -1;

  int n_remain_left  = 0;  // number of left "remain" bits (prefix of left_bits)
  int n_remain_ports = 0;  // number of right "remain" ports (prefix of right_ports)

  std::vector<LeftBit> left_bits;   // order: Remain(j) → Local(s-1,j) → Carries(s-1,j+1)
  std::vector<Port>    right_ports; // order: Remain ports → per-type(instance,pin) in lib order
  std::vector<int>     right_to_left; // size == right_ports.size(); values in [0..left_bits.size()-1]
};

/* Overall result of connection-order optimization. */
struct ConnectionOrderResult {
  double signed_error = 0.0;
  double abs_error    = 0.0;
  std::vector<std::vector<column_sol_t>> allSols; // [stage][col]
};

/* Run connection-order optimization over all compressor types.
 *
 * Conventions:
 * - Columns are MSB-first: index 0 is MSB, increasing index is toward LSB.
 * - Carries flow toward MSB (i.e., from column j to j-1 when !same_col).
 * - For (s,j), incoming carries come from (s-1, j+1).
 *
 * Parameters:
 * - compCounts[s][j][t] : number of instances of type t placed at (s,j)
 * - V_out[s][j]         : number of bits *after* stage s at column j (from Pass #1)
 * - init_bits_col[j]    : number of initial (stage 0) partial-product bits at column j (MSB-first)
 * - bitW[j]             : bit significance weight for column j (MSB-first, e.g., 2^(C-1-j))
 * - run_tag             : base name for emitted .lp/.sol files
 */
ConnectionOrderResult optimize_connection_order(
  const std::vector<std::vector<std::vector<int>>>& compCounts,
  const std::vector<std::vector<int>>& V_out,
  const std::vector<int>& init_bits_col,
  const std::vector<double>& bitW,
  const std::string& run_tag
);
