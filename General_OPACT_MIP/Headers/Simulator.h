#pragma once
#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

#include <vector>
#include "CompressorLib.h"
#include "ProbEval.h"

/** Compute E[S], E[C] for a compressor type with given per-bit independent probabilities. */
void simulator_ES_EC_for_type(int type_index, const std::vector<double>& p_inputs, double& ES, double& EC);

#endif
