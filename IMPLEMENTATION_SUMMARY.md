# Implementation Summary: Multi-Input Pairs File

## Overview
Created a new Python file `ACs/multi_input_pairs.py` that contains functions with multiple independent input parameters (p1, p2, p3, ...) instead of a single probability parameter p, as requested in the issue.

## Files Created

### 1. ACs/multi_input_pairs.py (158 lines)
- Contains functions for all 16 approximate compressors in ac_lib.json
- Each function has parameters matching the `input_ports` from ac_lib.json
- Three functions per compressor: `_S` (sum), `_C` (carry), `_err` (error)
- Example: `EC22_S(p1, p2)`, `AC32_ew1_S(p1, p2, p3)`, `AC42_uw1_S(p1, p2, p3, p4)`

### 2. ACs/generate_multi_input_pairs.py (176 lines)
- Script to automatically generate multi_input_pairs.py from ac_lib.json
- Reads truth tables and converts them to expectation expressions
- Handles torch device selection (cuda/mps/cpu)
- Can be re-run whenever ac_lib.json is updated

### 3. ACs/README.md
- Documentation explaining the difference between pairs.py and multi_input_pairs.py
- Usage instructions for both files
- Examples showing the relationship between the two approaches

### 4. ACs/test_multi_input_pairs.py
- Comprehensive test and documentation script
- Shows function signatures for all ACs
- Demonstrates usage examples
- Verifies mathematical equivalence with pairs.py

### 5. .gitignore
- Excludes __pycache__ and build artifacts from version control

## Key Differences: pairs.py vs multi_input_pairs.py

### pairs.py (existing)
- Single parameter: `def AC_NAME_S(p):`
- Assumes all inputs have the same probability p
- Example: `EC22_S(p) = -2*p**2 + 2*p`

### multi_input_pairs.py (new)
- Multiple parameters: `def AC_NAME_S(p1, p2, p3, ...):`
- Each input has independent probability
- Example: `EC22_S(p1, p2) = p1*(1 - p2) + (1 - p1)*p2`

## Mathematical Verification

The new functions are mathematically equivalent to the original when all input probabilities are equal:

**EC22 Example:**
- pairs.py: `-2*p**2 + 2*p`
- multi_input_pairs.py with p1=p2=p: `p*(1-p) + (1-p)*p = 2*p - 2*p**2` ✓

**AC32_ew1 Example:**
- pairs.py: `-p**2 + 2*p`
- multi_input_pairs.py with p1=p2=p3=p: `2*p*(1-p)^2 + 3*p^2*(1-p) + p^3 = -p^2 + 2*p` ✓

## How to Use

### Generate the file
```bash
cd ACs
python3 generate_multi_input_pairs.py
```

### Import and use
```python
from ACs.multi_input_pairs import EC22_S, AC32_ew1_S
import torch

# Example 1: Equal probabilities
p1 = p2 = torch.tensor([0.5])
result = EC22_S(p1, p2)

# Example 2: Independent probabilities
p1 = torch.tensor([0.3])
p2 = torch.tensor([0.7])
result = EC22_S(p1, p2)

# Example 3: 3-input compressor
p1, p2, p3 = torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([0.5])
result = AC32_ew1_S(p1, p2, p3)
```

## Function Signatures

All 16 ACs now have multi-input versions:

| AC Name    | Width | Signature |
|------------|-------|-----------|
| EC22       | 2     | `EC22_S(p1, p2)` |
| AC32_ew1   | 3     | `AC32_ew1_S(p1, p2, p3)` |
| EC32       | 3     | `EC32_S(p1, p2, p3)` |
| AC42_ew1   | 4     | `AC42_ew1_S(p1, p2, p3, p4)` |
| AC42_uw1-12| 4     | `AC42_uw*_S(p1, p2, p3, p4)` |

Each AC has three functions: `_S` (sum), `_C` (carry), and `_err` (error).

## Validation

✓ All function signatures match input_ports from ac_lib.json
✓ Python syntax is valid (checked with py_compile)
✓ Mathematical equivalence verified for sample ACs
✓ Can be imported without torch (import fails with proper error)
✓ Generated file has 158 lines covering all 16 ACs
✓ Function lists (F_S_LIST, F_C_LIST, F_ERR_LIST) created

## Impact

This change does NOT modify any existing files except for adding .gitignore. The new multi_input_pairs.py file provides an alternative approach for users who need independent input probabilities, while the original pairs.py continues to work as before.
