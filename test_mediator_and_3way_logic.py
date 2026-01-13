"""
Test script to verify mediator exclusion and 3-way interaction filtering logic

This test demonstrates:
1. Mediators are excluded from the interaction construction
2. 3-way interactions must include treatment + at least one two_way variable
"""

# Simulate the logic for 3-way interaction filtering

# Example variables after preprocessing
all_variables = [
    "Treatment", "Gender", "Age", "Education",  # Treatment + confounders
    "Income",  # Mediator (should be excluded!)
    "Region_A", "Region_B",  # Categorical variable
]

# User selections
two_way_interaction_variables = ["Gender", "Region_A", "Region_B"]
three_way_interaction_variables = ["Age", "Education"]
mediators = {"Income"}  # Mediator on causal path

print("="*80)
print("TEST: MEDIATOR EXCLUSION AND 3-WAY INTERACTION FILTERING")
print("="*80)

# Step 1: Filter out mediators from unique_values
unique_values = [v for v in all_variables if v not in mediators]
print(f"\nOriginal variables: {all_variables}")
print(f"Mediators (excluded): {mediators}")
print(f"Variables used for interactions: {unique_values}")

# Step 2: Generate all 3-way combinations
from itertools import combinations
all_3way = list(combinations(unique_values, 3))
print(f"\nTotal 3-way combinations: {len(all_3way)}")

# Step 3: Filter 3-way interactions for treatment variables (d_cols)
# Requirements:
# 1. Must include treatment
# 2. Must include at least one variable from two_way list
# 3. Third variable can be from either two_way or three_way list

treatment = "Treatment"
valid_3way_for_dcols = []
blocked_no_treatment = []
blocked_no_two_way_var = []

for combi in all_3way:
    parts = list(combi)

    # Check 1: Must include treatment
    has_treatment = treatment in parts
    if not has_treatment:
        blocked_no_treatment.append(combi)
        continue

    # Get other variables (not treatment)
    other_vars = [p for p in parts if p != treatment]

    # Check 2: Must have at least one variable from two_way list
    has_two_way_var = any(v in two_way_interaction_variables for v in other_vars)
    if not has_two_way_var:
        blocked_no_two_way_var.append(combi)
        continue

    # Check 3: Both other variables should be in selected lists
    all_selected = two_way_interaction_variables + three_way_interaction_variables
    matching_vars = [v for v in other_vars if v in all_selected]

    if len(matching_vars) >= 2:
        valid_3way_for_dcols.append(combi)

print("\n" + "="*80)
print("FILTERING RESULTS")
print("="*80)

print(f"\nValid 3-way interactions (will be in d_cols): {len(valid_3way_for_dcols)}")
for interaction in valid_3way_for_dcols:
    other_vars = [v for v in interaction if v != treatment]
    has_two_way = [v for v in other_vars if v in two_way_interaction_variables]
    print(f"  ✅ {':'.join(interaction)} - has two_way var: {has_two_way}")

print(f"\nBlocked - no treatment: {len(blocked_no_treatment)}")
if blocked_no_treatment:
    for ex in blocked_no_treatment[:5]:
        print(f"  ❌ {':'.join(ex)}")

print(f"\nBlocked - no two_way variable: {len(blocked_no_two_way_var)}")
if blocked_no_two_way_var:
    for ex in blocked_no_two_way_var[:5]:
        print(f"  ❌ {':'.join(ex)} - only has variables from three_way or not selected")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("""
1. ✅ Mediators (Income) are EXCLUDED from unique_values
   → No interactions with Income will be created

2. ✅ 3-way interactions for d_cols MUST include Treatment
   → Filters out interactions between non-treatment variables

3. ✅ 3-way interactions for d_cols MUST include at least one two_way variable
   → Ensures interactions build on the two_way structure
   → Example: If Gender is in two_way, then Treatment:Gender:Age is valid
   → But Treatment:Age:Education would be blocked (neither Age nor Education in two_way)

This matches the requirement:
"3-way interactions that are considered as treatment variables and passed to the
d_col parameter need to always include the main treatment variable and the
variable(s) for 2-way interaction"
""")
