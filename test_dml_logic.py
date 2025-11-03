"""
Test script to validate DML logic before running the full notebook
"""
import pandas as pd
import numpy as np
import re

# Create a small synthetic dataset to test the logic
np.random.seed(42)
n = 100

# Create synthetic data
data = {
    'Hourly_Salary_log': np.random.randn(n) + 3,
    'Bachelor_Degree': np.random.binomial(1, 0.3, n),
    'Female': np.random.binomial(1, 0.5, n),
    'Ethnic_Asian': np.random.binomial(1, 0.05, n),
    'Ethnic_Black': np.random.binomial(1, 0.15, n),
    'Wealth_q1': np.random.binomial(1, 0.25, n),
    'Wealth_q4': np.random.binomial(1, 0.25, n),
    'Region_South': np.random.binomial(1, 0.3, n),
    'Region_West': np.random.binomial(1, 0.2, n),
}

causal_df_3w = pd.DataFrame(data)

# Add 2-way interaction
causal_df_3w['Bachelor_Degree:Female'] = causal_df_3w['Bachelor_Degree'] * causal_df_3w['Female']

# Add some 3-way interactions
causal_df_3w['Bachelor_Degree:Female:Ethnic_Asian'] = (
    causal_df_3w['Bachelor_Degree'] * causal_df_3w['Female'] * causal_df_3w['Ethnic_Asian']
)
causal_df_3w['Bachelor_Degree:Female:Wealth_q4'] = (
    causal_df_3w['Bachelor_Degree'] * causal_df_3w['Female'] * causal_df_3w['Wealth_q4']
)
causal_df_3w['Bachelor_Degree:Female:Region_South'] = (
    causal_df_3w['Bachelor_Degree'] * causal_df_3w['Female'] * causal_df_3w['Region_South']
)

print("Synthetic dataframe created with shape:", causal_df_3w.shape)
print("\nColumns:", list(causal_df_3w.columns))

# Test the treatment variables construction logic
treatment_variables = []

# Add main effects
treatment_variables.append('Bachelor_Degree')
treatment_variables.append('Female')

# Add 2-way interaction Bachelor_Degree:Female
if 'Bachelor_Degree:Female' in causal_df_3w.columns:
    treatment_variables.append('Bachelor_Degree:Female')
elif 'Female:Bachelor_Degree' in causal_df_3w.columns:
    treatment_variables.append('Female:Bachelor_Degree')
else:
    # Create it if it doesn't exist
    causal_df_3w['Bachelor_Degree:Female'] = causal_df_3w['Bachelor_Degree'] * causal_df_3w['Female']
    treatment_variables.append('Bachelor_Degree:Female')

# Add all 3-way interactions: Bachelor_Degree:Female:X
pattern1 = r'.*Female.*'
pattern2 = r'.*Bachelor_Degree.*'
for col in causal_df_3w.columns:
    if re.search(pattern1, col, re.IGNORECASE) and re.search(pattern2, col, re.IGNORECASE):
        # Check if it's a 3-way interaction (has 2 colons)
        if col.count(':') == 2:
            # Make sure it's not a duplicate and not already in the list
            if col not in treatment_variables:
                # Check that the column is not constant (has variation)
                if causal_df_3w[col].nunique() > 1:
                    treatment_variables.append(col)

print(f"\nTotal treatment variables: {len(treatment_variables)}")
print(f"Main effects: Bachelor_Degree, Female")
print(f"2-way interaction: 1")
print(f"3-way interactions: {len(treatment_variables) - 3}")
print(f"\nTreatment variables: {treatment_variables}")

# Test DoubleML data creation
try:
    import doubleml as dml
    from sklearn.linear_model import Lasso

    obj_dml_data = dml.DoubleMLData(
        causal_df_3w,
        y_col='Hourly_Salary_log',
        d_cols=treatment_variables,
        use_other_treat_as_covariate=True
    )

    print(f"\nDoubleMLData object created successfully!")
    print(f"Number of observations: {obj_dml_data.n_obs}")
    print(f"Number of covariates: {obj_dml_data.n_coefs}")

    # Create learners
    ml_l = Lasso(fit_intercept=True, alpha=0.1)
    ml_m = Lasso(fit_intercept=True, alpha=0.1)

    # Create DML object
    dml_plr = dml.DoubleMLPLR(
        obj_dml_data,
        ml_l=ml_l,
        ml_m=ml_m,
        ml_g=None,
        n_folds=2,  # Use fewer folds for testing
        n_rep=1     # Use fewer reps for testing
    )

    print("\nDoubleMLPLR object created successfully!")

    # Fit the model
    print("\nFitting model...")
    dml_plr.fit()
    print("Model fitted successfully!")

    # Display results
    print("\n" + "="*60)
    print("MODEL RESULTS")
    print("="*60)
    print(dml_plr.summary)

    print("\n✓ All validation tests passed!")

except Exception as e:
    print(f"\n✗ Error during validation: {str(e)}")
    import traceback
    traceback.print_exc()
