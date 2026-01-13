"""
DML Estimator Component using DoubleML library
Implements Double Machine Learning for causal effect estimation
Following the logic from Sample_Analysis.ipynb notebook
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from itertools import combinations
import re


class DMLEstimatorDoubleML:
    """
    Double Machine Learning estimator using DoubleML library
    Estimates main effect and interaction effects
    """

    def __init__(self, data, dag, treatment, outcome, column_types,
                 two_way_interaction_variables=None, three_way_interaction_variables=None,
                 interaction_variables=None):
        """
        Initialize DML estimator

        Args:
            data (pd.DataFrame): The dataset
            dag (nx.DiGraph): The causal DAG
            treatment (str): Treatment variable name
            outcome (str): Outcome variable name
            column_types (dict): Variable type specifications
            two_way_interaction_variables (list): Variables for two-way interactions with treatment
            three_way_interaction_variables (list): Variables for three-way interactions
            interaction_variables (list): DEPRECATED - Use two_way and three_way instead
        """
        self.data = data.copy()
        self.dag = dag
        self.treatment = treatment
        self.outcome = outcome
        self.column_types = column_types
        self.two_way_interaction_variables = two_way_interaction_variables or []
        self.three_way_interaction_variables = three_way_interaction_variables or []
        # Legacy support
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

            # Create list of all treatments: main treatment + two_way vars + all interactions
            # Following Sample_Analysis.ipynb logic: include both main treatments
            treatment_variables = [self.treatment]

            # Add two_way interaction variables as separate main effects (like Gender_Female in Sample_Analysis)
            for var in self.two_way_interaction_variables:
                if var in processed_data.columns and var not in treatment_variables:
                    treatment_variables.append(var)

            # Add interaction terms as additional treatments
            for term in self.interaction_terms:
                treatment_variables.append(term['name'])

            print(f"Estimating effects for {len(treatment_variables)} treatments:")
            print(f"  - Main treatment: {self.treatment}")
            print(f"  - Additional main effects: {len(self.two_way_interaction_variables)}")
            print(f"  - Interaction terms: {len(self.interaction_terms)}")
            print(f"  - Total: {len(treatment_variables)}")

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

            # Extract all treatment effects
            all_results = []
            for idx, treat_var in enumerate(treatment_variables):
                result = {
                    'term': treat_var,
                    'variables': [treat_var] if idx < (1 + len(self.two_way_interaction_variables)) else
                                 next((t['variables'] for t in self.interaction_terms if t['name'] == treat_var), [treat_var]),
                    'order': 1 if idx < (1 + len(self.two_way_interaction_variables)) else
                            next((t['order'] for t in self.interaction_terms if t['name'] == treat_var), 1),
                    'coefficient': float(dml_plr.coef[idx]),
                    'se': float(dml_plr.se[idx]),
                    't_statistic': float(dml_plr.coef[idx] / dml_plr.se[idx]),
                    'ci_lower': float(conf_int_df.iloc[idx, 0]),
                    'ci_upper': float(conf_int_df.iloc[idx, 1]),
                    'p_value': float(p_val_df.iloc[idx, 1]),
                    'significant': p_val_df.iloc[idx, 1] < alpha,
                    'sig_1pct': p_val_df.iloc[idx, 1] < 0.01,
                    'sig_5pct': p_val_df.iloc[idx, 1] < 0.05,
                    'sig_10pct': p_val_df.iloc[idx, 1] < 0.10,
                    'rv_percent': float(sensitivity_df.iloc[idx, 0]) if len(sensitivity_df) > idx else 0.0
                }
                all_results.append(result)

            # Sort by p-value (like Sample_Analysis.ipynb)
            all_results.sort(key=lambda x: x['p_value'])

            # Extract main treatment effect (first one)
            main_ate = float(dml_plr.coef[0])
            main_se = float(dml_plr.se[0])
            main_ci_lower = float(conf_int_df.iloc[0, 0])
            main_ci_upper = float(conf_int_df.iloc[0, 1])
            main_p_value = float(p_val_df.iloc[0, 1])

            # Separate interaction results from main effects
            interaction_results = [r for r in all_results if r['order'] >= 2]

            # Identify confounders from DAG
            confounders = self._identify_confounders()

            # Create detailed results dataframe (like Sample_Analysis.ipynb)
            detailed_df = pd.DataFrame({
                'treatment': [r['term'] for r in all_results],
                'coefficient': [r['coefficient'] for r in all_results],
                'std_error': [r['se'] for r in all_results],
                't_statistic': [r['t_statistic'] for r in all_results],
                'p_value': [r['p_value'] for r in all_results],
                'ci_lower_95': [r['ci_lower'] for r in all_results],
                'ci_upper_95': [r['ci_upper'] for r in all_results],
                'sig_1pct': [r['sig_1pct'] for r in all_results],
                'sig_5pct': [r['sig_5pct'] for r in all_results],
                'sig_10pct': [r['sig_10pct'] for r in all_results]
            })

            # Already sorted by p-value

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
                'interaction_results': interaction_results,
                'detailed_results_df': detailed_df
            }

            return results

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"DML estimation failed: {str(e)}")

    def _construct_interaction_terms(self, data):
        """
        Construct ALL combinatorially possible interaction terms following Sample_Analysis.ipynb logic
        Steps:
        1. Generate ALL 2-way combinations of variables (excluding outcome)
        2. Filter out interactions within same category (e.g., no Father_X:Father_Y)
        3. Generate ALL 3-way combinations of variables (excluding outcome)
        4. Filter out interactions within same category
        5. Construct treatment variables based on user input from Step 3

        Args:
            data (pd.DataFrame): Preprocessed data

        Returns:
            pd.DataFrame: Data with ALL interaction terms added
        """
        print("\n" + "="*80)
        print("CONSTRUCTING INTERACTION TERMS (Sample_Analysis.ipynb logic)")
        print("="*80)

        result_data = data.copy()
        self.interaction_terms = []

        # Get all variable names (excluding outcome)
        all_vars = [col for col in data.columns if col != self.outcome]
        print(f"\nTotal variables (excluding outcome): {len(all_vars)}")

        # ========================================
        # STEP 1: Generate ALL 2-way combinations
        # ========================================
        print("\n[STEP 1] Generating all 2-way combinations...")
        all_combinations_2w = list(combinations(all_vars, 2))
        print(f"Total 2-way combinations: {len(all_combinations_2w)}")

        # Filter: Remove interactions within same category
        interaction_terms_2way = []
        for combi in all_combinations_2w:
            # Extract category prefix (first 6 characters)
            prefix1 = combi[0][:6] if len(combi[0]) >= 6 else combi[0]
            prefix2 = combi[1][:6] if len(combi[1]) >= 6 else combi[1]

            # Only keep if different categories
            if prefix1 != prefix2:
                column_name = f"{combi[0]}:{combi[1]}"
                result_data[column_name] = data[combi[0]] * data[combi[1]]
                interaction_terms_2way.append(column_name)

        print(f"2-way interactions created (after filtering): {len(interaction_terms_2way)}")

        # ========================================
        # STEP 2: Generate ALL 3-way combinations
        # ========================================
        print("\n[STEP 2] Generating all 3-way combinations...")
        all_combinations_3w = list(combinations(all_vars, 3))
        print(f"Total 3-way combinations: {len(all_combinations_3w)}")

        # Filter: Remove interactions within same category
        interaction_terms_3way = []
        for combi in all_combinations_3w:
            # Extract category prefixes
            prefix1 = combi[0][:6] if len(combi[0]) >= 6 else combi[0]
            prefix2 = combi[1][:6] if len(combi[1]) >= 6 else combi[1]
            prefix3 = combi[2][:6] if len(combi[2]) >= 6 else combi[2]

            # Only keep if all different categories
            if (prefix1 != prefix2) and (prefix1 != prefix3) and (prefix2 != prefix3):
                column_name = f"{combi[0]}:{combi[1]}:{combi[2]}"
                result_data[column_name] = data[combi[0]] * data[combi[1]] * data[combi[2]]
                interaction_terms_3way.append(column_name)

        print(f"3-way interactions created (after filtering): {len(interaction_terms_3way)}")

        # ========================================
        # STEP 3: Construct treatment variables based on user input
        # ========================================
        print("\n[STEP 3] Constructing treatment variables based on user selection...")

        # Legacy support
        if self.interaction_variables and not self.two_way_interaction_variables:
            self.two_way_interaction_variables = self.interaction_variables

        if not self.two_way_interaction_variables:
            print("No interaction variables specified by user - using main treatment only")
            return result_data

        # Filter interaction variables to only those that exist in data
        valid_two_way = [v for v in self.two_way_interaction_variables if v in data.columns]
        valid_three_way = [v for v in self.three_way_interaction_variables if v in data.columns]

        print(f"User-selected two-way variables: {valid_two_way}")
        print(f"User-selected three-way variables: {valid_three_way}")

        # Now identify which interactions to treat as "treatments of interest"
        # Based on Sample_Analysis.ipynb: treatment, treatment:var, treatment:var1:var2

        # Pattern matching for treatment variables
        # Format: treatment:X or treatment:X:Y where X and Y are in user-selected variables
        treatment_pattern_2way = re.compile(
            f"^{re.escape(self.treatment)}:(.+)$"
        )
        treatment_pattern_3way = re.compile(
            f"^{re.escape(self.treatment)}:(.+):(.+)$"
        )

        # Find matching 2-way interactions
        for interaction in interaction_terms_2way:
            match = treatment_pattern_2way.match(interaction)
            if match:
                var = match.group(1)
                if var in valid_two_way:
                    self.interaction_terms.append({
                        'name': interaction,
                        'variables': [self.treatment, var],
                        'order': 2
                    })

        # Find matching 3-way interactions
        for interaction in interaction_terms_3way:
            match = treatment_pattern_3way.match(interaction)
            if match:
                var1 = match.group(1)
                var2 = match.group(2)
                # Check if both variables are in user selection
                # var1 should be in two_way, var2 should be in two_way or three_way
                if var1 in valid_two_way and (var2 in valid_two_way or var2 in valid_three_way):
                    self.interaction_terms.append({
                        'name': interaction,
                        'variables': [self.treatment, var1, var2],
                        'order': 3
                    })

        n_two_way = sum(1 for t in self.interaction_terms if t['order'] == 2)
        n_three_way = sum(1 for t in self.interaction_terms if t['order'] == 3)

        print(f"\nTreatment interaction terms identified:")
        print(f"  - 2-way: {n_two_way}")
        print(f"  - 3-way: {n_three_way}")
        print(f"  - Total: {len(self.interaction_terms)}")

        print("\n" + "="*80)

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
