"""
DML Estimator Component using DoubleML library
Implements Double Machine Learning for causal effect estimation
Following the logic from 04_11_DML.ipynb notebook
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from itertools import combinations


class DMLEstimatorDoubleML:
    """
    Double Machine Learning estimator using DoubleML library
    Estimates main effect and interaction effects
    """

    def __init__(self, data, dag, treatment, outcome, column_types, interaction_variables=None):
        """
        Initialize DML estimator

        Args:
            data (pd.DataFrame): The dataset
            dag (nx.DiGraph): The causal DAG
            treatment (str): Treatment variable name
            outcome (str): Outcome variable name
            column_types (dict): Variable type specifications
            interaction_variables (list): Variables to create interaction terms from
        """
        self.data = data.copy()
        self.dag = dag
        self.treatment = treatment
        self.outcome = outcome
        self.column_types = column_types
        self.interaction_variables = interaction_variables or []
        self.model = None
        self.interaction_terms = []

    def estimate_ate(self, discrete_treatment=True, n_splits=5, alpha=0.05):
        """
        Estimate Average Treatment Effect using DML with doubleml library
        Estimates main effect and all interaction effects

        Args:
            discrete_treatment (bool): Whether treatment is discrete
            n_splits (int): Number of cross-validation splits
            alpha (float): Significance level for confidence intervals

        Returns:
            dict: Results including ATE, standard error, confidence intervals, interaction results
        """
        try:
            import doubleml as dml
            from scipy import stats

            # Preprocess data
            processed_data = self._preprocess_data()

            # Construct interaction terms
            processed_data = self._construct_interaction_terms(processed_data)

            # Create list of all treatments: main treatment + all interactions
            treatment_variables = [self.treatment]

            # Add interaction terms as additional treatments
            for term in self.interaction_terms:
                treatment_variables.append(term['name'])

            print(f"Estimating effects for {len(treatment_variables)} treatments: main + {len(self.interaction_terms)} interactions")

            # Create DoubleML data object
            obj_dml_data = dml.DoubleMLData(
                processed_data,
                y_col=self.outcome,
                d_cols=treatment_variables,
                use_other_treat_as_covariate=True  # Other treatments become covariates
            )

            # Set up learners (using Lasso as in the notebook)
            ml_l = Lasso(fit_intercept=True, alpha=1.0)  # Outcome model
            ml_m = Lasso(fit_intercept=True, alpha=1.0)  # Treatment model

            # Create DML model (Partially Linear Regression)
            dml_plr = dml.DoubleMLPLR(
                obj_dml_data,
                ml_l=ml_l,
                ml_m=ml_m,
                ml_g=None,
                n_folds=n_splits,
                n_rep=5  # 5 repetitions for cross-fitting
            )

            # Fit the model
            print("Fitting DML model...")
            dml_plr.fit()

            # Get bootstrap confidence intervals
            print("Computing bootstrap confidence intervals...")
            dml_plr.bootstrap(n_rep_boot=1000)
            conf_int_df = dml_plr.confint(joint=True, level=1-alpha)

            # Get adjusted p-values
            print("Computing adjusted p-values...")
            p_val_df = dml_plr.p_adjust()

            # Perform sensitivity analysis
            print("Performing sensitivity analysis...")
            try:
                dml_plr.sensitivity_analysis()
                sensitivity_df = pd.DataFrame({
                    'rv_percent': dml_plr.sensitivity_params['rv'] * 100
                }, index=treatment_variables)
            except Exception as e:
                print(f"Sensitivity analysis failed: {e}")
                sensitivity_df = pd.DataFrame({
                    'rv_percent': [0.0] * len(treatment_variables)
                }, index=treatment_variables)

            # Extract main treatment effect
            main_ate = float(dml_plr.coef[0])  # First treatment is the main one
            main_se = float(dml_plr.se[0])
            main_ci_lower = float(conf_int_df.iloc[0, 0])
            main_ci_upper = float(conf_int_df.iloc[0, 1])
            main_p_value = float(p_val_df.iloc[0, 1])

            # Extract interaction effects
            interaction_results = []
            for i, term in enumerate(self.interaction_terms):
                idx = i + 1  # +1 because main treatment is at index 0

                result = {
                    'term': term['name'],
                    'variables': term['variables'],
                    'order': term['order'],
                    'coefficient': float(dml_plr.coef[idx]),
                    'se': float(dml_plr.se[idx]),
                    'ci_lower': float(conf_int_df.iloc[idx, 0]),
                    'ci_upper': float(conf_int_df.iloc[idx, 1]),
                    'p_value': float(p_val_df.iloc[idx, 1]),
                    'significant': p_val_df.iloc[idx, 1] < alpha,
                    'rv_percent': float(sensitivity_df.iloc[idx, 0]) if len(sensitivity_df) > idx else 0.0
                }
                interaction_results.append(result)

            # Sort by absolute coefficient
            interaction_results.sort(key=lambda x: abs(x['coefficient']), reverse=True)

            # Identify confounders from DAG
            confounders = self._identify_confounders()

            results = {
                'ate': main_ate,
                'se': main_se,
                'ci_lower': main_ci_lower,
                'ci_upper': main_ci_upper,
                'p_value': main_p_value,
                'confounders': confounders,
                'n_samples': len(processed_data),
                'model_summary': f"DoubleML PLR with {n_splits} folds and 5 repetitions",
                'interaction_terms': [{'name': t['name'], 'variables': t['variables'], 'order': t['order']}
                                     for t in self.interaction_terms],
                'interaction_results': interaction_results
            }

            return results

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"DML estimation failed: {str(e)}")

    def _construct_interaction_terms(self, data):
        """
        Construct interaction terms from selected variables
        Creates both 2-way and 3-way interactions

        Args:
            data (pd.DataFrame): Preprocessed data

        Returns:
            pd.DataFrame: Data with interaction terms added
        """
        if not self.interaction_variables:
            return data

        result_data = data.copy()
        self.interaction_terms = []

        # Filter interaction variables to only those that exist in data
        valid_interaction_vars = [v for v in self.interaction_variables if v in data.columns]

        # Create 2-way interactions
        for var1, var2 in combinations(valid_interaction_vars, 2):
            # Avoid creating interactions with same prefix (e.g., Father_X * Father_Y)
            var1_prefix = var1.split('_')[0]
            var2_prefix = var2.split('_')[0]

            if var1_prefix != var2_prefix or var1_prefix in ['Treatment', 'Outcome']:
                interaction_name = f"{var1}:{var2}"
                result_data[interaction_name] = data[var1] * data[var2]
                self.interaction_terms.append({
                    'name': interaction_name,
                    'variables': [var1, var2],
                    'order': 2
                })

        # Create 3-way interactions
        if len(valid_interaction_vars) >= 3:
            for var1, var2, var3 in combinations(valid_interaction_vars, 3):
                # Avoid creating interactions with same prefix
                var1_prefix = var1.split('_')[0]
                var2_prefix = var2.split('_')[0]
                var3_prefix = var3.split('_')[0]

                if (var1_prefix != var2_prefix and var1_prefix != var3_prefix and var2_prefix != var3_prefix) \
                        or any(p in ['Treatment', 'Outcome'] for p in [var1_prefix, var2_prefix, var3_prefix]):
                    interaction_name = f"{var1}:{var2}:{var3}"
                    result_data[interaction_name] = data[var1] * data[var2] * data[var3]
                    self.interaction_terms.append({
                        'name': interaction_name,
                        'variables': [var1, var2, var3],
                        'order': 3
                    })

        print(f"Created {len(self.interaction_terms)} interaction terms ({sum(1 for t in self.interaction_terms if t['order']==2)} two-way, {sum(1 for t in self.interaction_terms if t['order']==3)} three-way)")

        return result_data

    def _preprocess_data(self):
        """
        Preprocess data based on variable types

        Returns:
            pd.DataFrame: Preprocessed data
        """
        processed = self.data.copy()

        # Handle missing values
        processed = processed.dropna(subset=[self.treatment, self.outcome])

        # Encode categorical variables
        for col, dtype in self.column_types.items():
            if col not in processed.columns:
                continue

            if dtype in ['binary', 'categorical', 'ordinal']:
                # Label encoding
                le = LabelEncoder()
                # Handle any remaining NaN
                mask = processed[col].notna()
                if mask.any():
                    processed.loc[mask, col] = le.fit_transform(
                        processed.loc[mask, col].astype(str)
                    )

        # Convert to numeric
        for col in processed.columns:
            try:
                processed[col] = pd.to_numeric(processed[col], errors='coerce')
            except:
                pass

        # Drop any remaining NaN
        processed = processed.dropna()

        return processed

    def _identify_confounders(self):
        """
        Identify confounding variables from the DAG

        Returns:
            list: List of confounder variable names
        """
        confounders = set()

        # Find all variables that are:
        # 1. Causes of treatment
        # 2. Causes of outcome
        # (i.e., common causes - confounders)

        treatment_causes = set(self.dag.predecessors(self.treatment)) if self.dag.has_node(self.treatment) else set()
        outcome_causes = set(self.dag.predecessors(self.outcome)) if self.dag.has_node(self.outcome) else set()

        # Common causes are confounders
        confounders = treatment_causes.intersection(outcome_causes)

        # Also include any causes of treatment (for backdoor adjustment)
        confounders = confounders.union(treatment_causes)

        # Remove treatment and outcome themselves
        confounders.discard(self.treatment)
        confounders.discard(self.outcome)

        return list(confounders)
