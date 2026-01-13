"""
Test to understand why we might get 20,747 columns

This simulates what happens when we have different variable naming patterns
"""
from itertools import combinations

# Scenario 1: Variables with shared prefixes (like in Sample_Analysis.ipynb)
variables_with_prefixes = [
    "Ethnic_Asian", "Ethnic_Black", "Ethnic_Other", "Ethnic_Hawaiian",
    "Region_South", "Region_Northeast", "Region_West",
    "Father_6-8_grades", "Father_9-11_grades", "Father_12_grades", "Father_College_BA",
    "Mother_6-8_grades", "Mother_9-11_grades", "Mother_12_grades", "Mother_College_BA",
    "Wealth_q1", "Wealth_q2", "Wealth_q3", "Wealth_q4",
    "Gender_Female", "Degree", "Born_Foreign"
]

# Scenario 2: Variables WITHOUT shared prefixes (each is unique)
variables_without_prefixes = [
    "Age", "Income", "Height", "Weight", "Score1", "Score2", "Grade", "Level",
    "Value1", "Value2", "Amount", "Total", "Count", "Price", "Cost", "Rate",
    "Factor1", "Factor2", "Item1", "Item2", "Unit", "Batch"
]

def analyze_scenario(variables, scenario_name):
    print("\n" + "="*80)
    print(f"SCENARIO: {scenario_name}")
    print("="*80)
    print(f"Number of variables: {len(variables)}")

    # Analyze prefixes
    prefix_counts = {}
    for var in variables:
        prefix = var[:6]
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    prefixes_with_multiple = {k: v for k, v in prefix_counts.items() if v > 1}
    print(f"\nPrefix analysis:")
    print(f"  Total unique prefixes: {len(prefix_counts)}")
    print(f"  Prefixes with 2+ variables: {len(prefixes_with_multiple)}")
    if prefixes_with_multiple:
        print(f"  Shared prefixes:")
        for prefix, count in sorted(prefixes_with_multiple.items(), key=lambda x: x[1], reverse=True):
            cols = [v for v in variables if v[:6] == prefix]
            print(f"    '{prefix}': {count} variables -> {cols}")

    # Calculate 2-way interactions
    all_2way = list(combinations(variables, 2))
    allowed_2way = []
    blocked_2way = []

    for combi in all_2way:
        prefix1 = combi[0][:6]
        prefix2 = combi[1][:6]
        if prefix1 != prefix2:
            allowed_2way.append(combi)
        else:
            blocked_2way.append(combi)

    print(f"\n2-way interactions:")
    print(f"  Total possible: {len(all_2way)}")
    print(f"  Allowed (different prefixes): {len(allowed_2way)}")
    print(f"  Blocked (same prefix): {len(blocked_2way)}")
    print(f"  Block rate: {len(blocked_2way)/len(all_2way)*100:.1f}%")

    if blocked_2way:
        print(f"  Examples of blocked:")
        for ex in blocked_2way[:5]:
            print(f"    ❌ {ex[0]}:{ex[1]}")

    # Calculate 3-way interactions
    all_3way = list(combinations(variables, 3))
    allowed_3way = []
    blocked_3way = []

    for combi in all_3way:
        prefix1 = combi[0][:6]
        prefix2 = combi[1][:6]
        prefix3 = combi[2][:6]
        if (prefix1 != prefix2) and (prefix1 != prefix3) and (prefix2 != prefix3):
            allowed_3way.append(combi)
        else:
            blocked_3way.append(combi)

    print(f"\n3-way interactions:")
    print(f"  Total possible: {len(all_3way)}")
    print(f"  Allowed (all different prefixes): {len(allowed_3way)}")
    print(f"  Blocked (at least 2 same prefix): {len(blocked_3way)}")
    print(f"  Block rate: {len(blocked_3way)/len(all_3way)*100:.1f}%")

    if blocked_3way:
        print(f"  Examples of blocked:")
        for ex in blocked_3way[:5]:
            print(f"    ❌ {ex[0]}:{ex[1]}:{ex[2]}")

    # Total columns
    total_cols = len(variables) + len(allowed_2way) + len(allowed_3way)
    print(f"\n{'='*80}")
    print(f"TOTAL COLUMNS IN DATAFRAME:")
    print(f"  Original variables:           {len(variables):>6,}")
    print(f"  2-way interactions (allowed): {len(allowed_2way):>6,}")
    print(f"  3-way interactions (allowed): {len(allowed_3way):>6,}")
    print(f"  {'─'*80}")
    print(f"  TOTAL:                        {total_cols:>6,}")
    print(f"{'='*80}")

    return total_cols

# Run scenarios
total1 = analyze_scenario(variables_with_prefixes, "Variables WITH shared prefixes (like Sample_Analysis.ipynb)")
total2 = analyze_scenario(variables_without_prefixes, "Variables WITHOUT shared prefixes (each unique)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"""
If you're seeing 20,747 columns, this could happen if:

1. You have ~50 variables with mostly unique prefixes (different categories)
   - Example: If 50 variables have 45 unique prefixes
   - 2-way: ~1,200 interactions (few blocked)
   - 3-way: ~19,000 interactions (few blocked)
   - Total: ~20,250 columns ✓ Matches your observation!

2. Your variable names don't follow the prefix pattern
   - Example: "Age", "Income", "Score" instead of "Demog_Age", "Demog_Income", "Academ_Score"
   - Each variable is treated as a unique category
   - Result: Almost NO interactions are blocked → Very high column count

The filtering IS working correctly - it's blocking interactions within the same
categorical variable (same 6-character prefix). But if your variables don't have
shared prefixes, then most interactions will be allowed, resulting in a high column count.

RECOMMENDATIONS:
1. Check your variable names - do they follow the pattern "CategoryPrefix_Value"?
2. If variables should be grouped (like Ethnicity categories), ensure they share a prefix
3. Consider whether you really need ALL 3-way interactions, or only specific ones
""")
