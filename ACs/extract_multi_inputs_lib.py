#!/usr/bin/env python
"""
Generate multi_input_pairs.py from ac_lib.json.
Each function has multiple input parameters (p1, p2, p3, ...) 
instead of a single parameter p.
"""
import json
from pathlib import Path
import sympy as sp

# Configuration
AC_LIB = Path(__file__).parent / "ac_lib.json"
OUT_FILE = Path(__file__).parent / "multi_input_pairs.py"

def generate_expectation_expr(truth_table, input_ports, use_torch=True):
    """
    Generate expectation expression from truth table.
    
    Args:
        truth_table: List of 0s and 1s for the output
        input_ports: List of input port names (e.g., ['x1', 'x2', 'x3'])
        use_torch: If True, generate torch expressions, else numpy
        
    Returns:
        String expression using p1, p2, p3, etc.
    """
    n_inputs = len(input_ports)
    terms = 0
    
    # Iterate through all possible input combinations
    for i in range(2 ** n_inputs):
        if truth_table[i] == 0:
            continue
            
        # Build term for this input combination
        term_factors = 1
        for bit_idx in range(n_inputs):
            bit_val = (i >> bit_idx) & 1
            param_name = f"p{bit_idx + 1}"
            p = sp.symbols(param_name)

            if bit_val == 1:
                term_factors *= p
            else:
                term_factors *= (1 - p)

        terms += term_factors
    if terms == 0:
        return "torch.zeros_like(p1, device=device)"
    elif terms == 1:
        return "torch.ones_like(p1, device=device)"
    else:
        return sp.expand(terms)

def generate_multi_input_pairs():
    """Generate multi_input_pairs.py from ac_lib.json"""
    
    # Load ac_lib.json
    with open(AC_LIB, 'r') as f:
        lib = json.load(f)
    
    # Start writing output file
    with open(OUT_FILE, 'w') as f:
        # Write header
        f.write("import torch\n\n")
        f.write("if torch.cuda.is_available():\n")
        f.write("    device = \"cuda\"\n\n")
        f.write("elif torch.backends.mps.is_available():\n")
        f.write("    device = \"mps\"\n\n")
        f.write("else:\n")
        f.write("    device = \"cpu\"\n\n")
        
        # Sort AC names for deterministic order
        ac_names = sorted(lib.keys())
        
        # Generate functions for each AC
        for ac_name in ac_names:
            ac = lib[ac_name]
            input_ports = ac['input_ports']
            n_inputs = len(input_ports)
            
            # Generate parameter list (p1, p2, p3, ...)
            param_list = ", ".join([f"p{i+1}" for i in range(n_inputs)])
            
            # Generate S function
            truth_s = ac['truth_s']
            expr_s = generate_expectation_expr(truth_s, input_ports, use_torch=True)
            f.write(f"def {ac_name}_S({param_list}):\n")
            f.write(f"    return {expr_s}\n\n")
            
            # Generate C function
            truth_c = ac['truth_c']
            expr_c = generate_expectation_expr(truth_c, input_ports, use_torch=True)
            f.write(f"def {ac_name}_C({param_list}):\n")
            f.write(f"    return {expr_c}\n\n")
            
            # Generate err function (using the MED polynomial from ac_lib)
            # We need to convert the error polynomial as well
            # For now, let's compute it from truth tables
            med_poly = ac.get('med', '0')
            if med_poly in ('0', '0.0'):
                expr_err = "torch.zeros_like(p1, device=device)"
            else:
                # Compute error from truth tables
                # error = exact - approximate = popcount - (S + 2*C)
                error_tt = []
                for i in range(2 ** n_inputs):
                    popcount = bin(i).count('1')
                    approx = truth_s[i] + 2 * truth_c[i]
                    error_tt.append(1 if (popcount - approx) > 0 else 0)
                
                # Actually, we need the signed error, so this is more complex
                # Let's compute the expectation of (popcount - approx)
                error_terms = []
                for i in range(2 ** n_inputs):
                    popcount = bin(i).count('1')
                    if "ew" in ac_name:
                        approx = truth_s[i] + truth_c[i]
                    else:
                        approx = truth_s[i] + 2 * truth_c[i]
                    err = popcount - approx
                    if err == 0:
                        continue
                    
                    # Build term for this input combination
                    term_factors = []
                    for bit_idx in range(n_inputs):
                        bit_val = (i >> bit_idx) & 1
                        param_name = f"p{bit_idx + 1}"
                        if bit_val == 1:
                            term_factors.append(param_name)
                        else:
                            term_factors.append(f"(1 - {param_name})")
                    
                    if term_factors:
                        term = "*".join(term_factors)
                        if err > 0:
                            error_terms.append(f"{err}*{term}")
                        else:
                            error_terms.append(f"({err})*{term}")
                
                if not error_terms:
                    expr_err = "torch.zeros_like(p1, device=device)"
                else:
                    expr_err = " + ".join(error_terms)
            
            f.write(f"def {ac_name}_err({param_list}):\n")
            f.write(f"    return {expr_err}\n\n")
        
        # Generate function lists
        fn_list_S = ", ".join([f"{ac}_S" for ac in ac_names] + ["lambda p1: p1"])
        fn_list_C = ", ".join([f"{ac}_C" for ac in ac_names] + ["lambda p1: 0*p1"])
        fn_list_err = ", ".join([f"{ac}_err" for ac in ac_names] + ["lambda p1: 0*p1"])
        
        f.write(f"F_S_LIST = [{fn_list_S}]\n")
        f.write(f"F_C_LIST = [{fn_list_C}]\n")
        f.write(f"F_ERR_LIST = [{fn_list_err}]\n")
    
    print(f"Generated {OUT_FILE}")

if __name__ == "__main__":
    generate_multi_input_pairs()