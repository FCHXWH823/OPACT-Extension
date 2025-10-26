#include "wireConnect.h"

// In this refactor the detailed wiring is produced directly by ConnectionOrderOpt.
// The wireConnect layer becomes a thin DTO; keep the legacy struct for old paths.
// If you had parsing of .sol files here, point it to read the new _co.sol or use Gurobi APIs.
