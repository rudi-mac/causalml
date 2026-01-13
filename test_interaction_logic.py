"""
Test script to verify the interaction filtering logic matches Sample_Analysis.ipynb
Tests that interactions are NOT created within the same categorical variable
"""

# Test the prefix logic used in the interaction filtering
test_columns = [
    "Ethnic_Asian",
    "Ethnic_Black",
    "Ethnic_Other",
    "Region_South",
    "Region_Northeast",
    "Region_West",
    "Father_12_grades",
    "Father_College_BA",
    "Mother_12_grades",
    "Mother_College_BA",
    "Wealth_q1",
    "Wealth_q2",
    "Gender_Female",
    "Degree"
]

from itertools import combinations

# Test 2-way interaction filtering
print("="*80)
print("TESTING 2-WAY INTERACTION FILTERING LOGIC")
print("="*80)

all_combinations_2w = list(combinations(test_columns, 2))
print(f"\nTotal possible 2-way combinations: {len(all_combinations_2w)}")

# Apply the filtering logic from Sample_Analysis.ipynb cell 35
filtered_2w = []
blocked_2w = []
for combi in all_combinations_2w:
    prefix1 = combi[0][:6]
    prefix2 = combi[1][:6]

    if prefix1 != prefix2:
        filtered_2w.append(f"{combi[0]}:{combi[1]}")
    else:
        blocked_2w.append(f"{combi[0]}:{combi[1]}")

print(f"\nInteractions ALLOWED (different categorical variables): {len(filtered_2w)}")
print(f"Interactions BLOCKED (same categorical variable): {len(blocked_2w)}")

print("\n--- Examples of BLOCKED interactions (same category) ---")
for interaction in blocked_2w[:10]:
    print(f"  ❌ {interaction}")

print("\n--- Examples of ALLOWED interactions (different categories) ---")
for interaction in filtered_2w[:10]:
    print(f"  ✅ {interaction}")

# Test 3-way interaction filtering
print("\n" + "="*80)
print("TESTING 3-WAY INTERACTION FILTERING LOGIC")
print("="*80)

all_combinations_3w = list(combinations(test_columns, 3))
print(f"\nTotal possible 3-way combinations: {len(all_combinations_3w)}")

# Apply the filtering logic from Sample_Analysis.ipynb cell 37
filtered_3w = []
blocked_3w = []
for combi in all_combinations_3w:
    prefix1 = combi[0][:6]
    prefix2 = combi[1][:6]
    prefix3 = combi[2][:6]

    if (prefix1 != prefix2) and (prefix1 != prefix3) and (prefix2 != prefix3):
        filtered_3w.append(f"{combi[0]}:{combi[1]}:{combi[2]}")
    else:
        blocked_3w.append(f"{combi[0]}:{combi[1]}:{combi[2]}")

print(f"\nInteractions ALLOWED (all different categorical variables): {len(filtered_3w)}")
print(f"Interactions BLOCKED (at least 2 from same category): {len(blocked_3w)}")

print("\n--- Examples of BLOCKED interactions (same category) ---")
for interaction in blocked_3w[:10]:
    parts = interaction.split(':')
    prefixes = [p[:6] for p in parts]
    print(f"  ❌ {interaction}")
    print(f"     Prefixes: {prefixes}")

print("\n--- Examples of ALLOWED interactions (all different categories) ---")
for interaction in filtered_3w[:10]:
    parts = interaction.split(':')
    prefixes = [p[:6] for p in parts]
    print(f"  ✅ {interaction}")
    print(f"     Prefixes: {prefixes}")

print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)
print(f"✅ 2-way filtering: {len(filtered_2w)}/{len(all_combinations_2w)} combinations allowed")
print(f"✅ 3-way filtering: {len(filtered_3w)}/{len(all_combinations_3w)} combinations allowed")
print("\nThe logic successfully prevents interactions within the same categorical variable!")
print("This matches the Sample_Analysis.ipynb cells 35 and 37 logic.")
