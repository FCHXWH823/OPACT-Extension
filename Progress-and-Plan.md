# Work Record

## Idea

Given the optimized compressor allocation, we further minimize the MED of the approximate CT by optimizing the connection order of all the allocated compressors. In this case, the area and delay of the circuit is fixed, so we only focus on error.

Originally, we want to optimize this function:
$$ min_{s} \sum_{IP_i}^{2^N} \frac{1}{2^N} \mid \sum_{compressor_j} MED_{i, j}(s) \mid $$

where s is a discrete variable, representing the possible bijective connection between compressors. This problem is solved by MIP solver.
Through **continuous relaxation**, we can turn it into a continuous optimization problem, and use approaches like gradient descent to solve it more efficiently.

The objective function after relaxation is

$$ min_{p} \sum_{IP_i}^{2^N} \frac{1}{2^N}  \mid \sum_{compressor_j} MED_{i, j}(p) \mid $$

where p represents the probability of each permutation is chosen.

## Progress

## Plan

- [ ] Read DiffLogic code
- [ ] List places that needs to modify
- [ ] Finish compressor library (Weihua Xiao)
- [ ] Finish slides for the whole idea (Weihua Xiao)
- [ ] Read the slides (Mohan Huang)
- [ ] Run and debug DiffLogic
