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
        self.categorical_mapping = {}  # Maps original categorical variable name to list of dummy columns

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

            # Preprocess data (includes dummification of categorical variables)
            processed_data = self._preprocess_data()

            # Construct interaction terms
            processed_data = self._construct_interaction_terms(processed_data)

            # Create list of all treatments following Sample_Analysis.ipynb logic:
            # After dummification, categorical variables become multiple columns
            # Each dummy column is a separate treatment effect to estimate
            treatment_variables = []

            # Add main treatment variable(s)
            # If treatment was categorical, it's now multiple dummy columns
            if hasattr(self, 'categorical_mapping') and self.treatment in self.categorical_mapping:
                # Treatment was categorical - add all dummy columns
                for dummy_col in self.categorical_mapping[self.treatment]:
                    if dummy_col in processed_data.columns:
                        treatment_variables.append(dummy_col)
                print(f"Treatment '{self.treatment}' was categorical, now {len([d for d in self.categorical_mapping[self.treatment] if d in processed_data.columns])} dummy columns")
            elif self.treatment in processed_data.columns:
                # Treatment is binary or continuous
                treatment_variables.append(self.treatment)

            # Add two_way interaction variables as separate main effects
            # These are like Gender_Female in Sample_Analysis - also treated as main effects
            for var in self.two_way_interaction_variables:
                # Check if this variable was dummified
                if hasattr(self, 'categorical_mapping') and var in self.categorical_mapping:
                    # Variable was categorical - add all its dummy columns
                    for dummy_col in self.categorical_mapping[var]:
                        if dummy_col in processed_data.columns and dummy_col not in treatment_variables:
                            treatment_variables.append(dummy_col)
                elif var in processed_data.columns and var not in treatment_variables:
                    # Variable is binary or continuous
                    treatment_variables.append(var)

            # Add interaction terms as additional treatments
            for term in self.interaction_terms:
                if term['name'] in processed_data.columns:
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

            # Fit the model (following Sample_Analysis.ipynb)
            print("Fitting DML model...")
            dml_plr.fit()
            print("Model fitted successfully!")

            # Get confidence intervals (following Sample_Analysis.ipynb)
            print("Computing confidence intervals...")
            conf_int_95 = dml_plr.confint(level=0.95)
            conf_int_99 = dml_plr.confint(level=0.99)

            # Get p-values (following Sample_Analysis.ipynb)
            print("Extracting p-values...")
            pvals = dml_plr.pval

            # Extract all treatment effects (following Sample_Analysis.ipynb)
            all_results = []
            for idx, treat_var in enumerate(treatment_variables):
                # Determine if this is a main effect or interaction
                is_interaction = treat_var in [t['name'] for t in self.interaction_terms]

                if is_interaction:
                    term_info = next((t for t in self.interaction_terms if t['name'] == treat_var), None)
                    variables = term_info['variables'] if term_info else [treat_var]
                    order = term_info['order'] if term_info else 1
                else:
                    variables = [treat_var]
                    order = 1

                result = {
                    'term': treat_var,
                    'variables': variables,
                    'order': order,
                    'coefficient': float(dml_plr.coef[idx]),
                    'se': float(dml_plr.se[idx]),
                    't_statistic': float(dml_plr.coef[idx] / dml_plr.se[idx]),
                    'ci_lower_95': float(conf_int_95.iloc[idx, 0]),
                    'ci_upper_95': float(conf_int_95.iloc[idx, 1]),
                    'ci_lower_99': float(conf_int_99.iloc[idx, 0]),
                    'ci_upper_99': float(conf_int_99.iloc[idx, 1]),
                    'p_value': float(pvals[idx]),
                    'significant': pvals[idx] < alpha,
                    'sig_1pct': pvals[idx] < 0.01,
                    'sig_5pct': pvals[idx] < 0.05,
                    'sig_10pct': pvals[idx] < 0.10
                }
                all_results.append(result)

            # Sort by p-value (like Sample_Analysis.ipynb)
            all_results.sort(key=lambda x: x['p_value'])

            # Extract main treatment effect (first treatment variable in the original list, not sorted)
            # Find the first occurrence of the original treatment in treatment_variables
            main_idx = 0  # Default to first
            for i, tv in enumerate(treatment_variables):
                if tv == self.treatment or (hasattr(self, 'categorical_mapping') and
                                           self.treatment in self.categorical_mapping and
                                           tv in self.categorical_mapping.get(self.treatment, [])):
                    main_idx = i
                    break

            main_ate = float(dml_plr.coef[main_idx])
            main_se = float(dml_plr.se[main_idx])
            main_ci_lower = float(conf_int_95.iloc[main_idx, 0])
            main_ci_upper = float(conf_int_95.iloc[main_idx, 1])
            main_p_value = float(pvals[main_idx])

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
                'ci_lower_95': [r['ci_lower_95'] for r in all_results],
                'ci_upper_95': [r['ci_upper_95'] for r in all_results],
                'ci_lower_99': [r['ci_lower_99'] for r in all_results],
                'ci_upper_99': [r['ci_upper_99'] for r in all_results],
                'sig_1pct': [r['sig_1pct'] for r in all_results],
                'sig_5pct': [r['sig_5pct'] for r in all_results],
                'sig_10pct': [r['sig_10pct'] for r in all_results]
            })

            # Already sorted by p-value

            print("\n" + "="*80)
            print("DML ESTIMATION COMPLETE")
            print("="*80)
            print(f"Total treatments estimated: {len(treatment_variables)}")
            print(f"Significant at p<0.05: {sum(1 for r in all_results if r['sig_5pct'])}")
            print(f"Significant at p<0.01: {sum(1 for r in all_results if r['sig_1pct'])}")
            print("="*80)

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
                'detailed_results_df': detailed_df,
                'all_results': all_results  # Include all results for display
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
        # Use flexible regex pattern matching like in the notebook

        # Create regex pattern for treatment (flexible position)
        treatment_pattern = re.compile(f".*{re.escape(self.treatment)}.*", re.IGNORECASE)

        # Find matching 2-way interactions
        # Logic: Find all 2-way interactions that contain BOTH treatment AND a two_way variable
        for interaction in interaction_terms_2way:
            # Check if interaction contains treatment
            if re.search(treatment_pattern, interaction):
                # Check if it contains any of the two_way variables
                for var in valid_two_way:
                    var_pattern = re.compile(f".*{re.escape(var)}.*", re.IGNORECASE)
                    if re.search(var_pattern, interaction):
                        # Verify it has exactly 1 colon (2-way interaction)
                        if interaction.count(':') == 1:
                            # Check that the column is not constant (has variation)
                            if result_data[interaction].nunique() > 1:
                                # Extract the actual variables from the interaction name
                                parts = interaction.split(':')
                                self.interaction_terms.append({
                                    'name': interaction,
                                    'variables': parts,
                                    'order': 2
                                })
                                break  # Only add once per interaction

        # Find matching 3-way interactions
        # Logic: Find all 3-way interactions that contain treatment AND a two_way variable AND a three_way variable
        for interaction in interaction_terms_3way:
            # Check if interaction contains treatment
            if re.search(treatment_pattern, interaction):
                # Check if it contains any of the two_way variables
                contains_two_way = False
                matching_two_way_var = None
                for var in valid_two_way:
                    var_pattern = re.compile(f".*{re.escape(var)}.*", re.IGNORECASE)
                    if re.search(var_pattern, interaction):
                        contains_two_way = True
                        matching_two_way_var = var
                        break

                if contains_two_way:
                    # Check if it contains any of the three_way variables (or another two_way variable)
                    contains_third = False
                    for var in valid_three_way + valid_two_way:
                        if var != matching_two_way_var:  # Don't match the same variable twice
                            var_pattern = re.compile(f".*{re.escape(var)}.*", re.IGNORECASE)
                            if re.search(var_pattern, interaction):
                                contains_third = True
                                break

                    if contains_third:
                        # Verify it has exactly 2 colons (3-way interaction)
                        if interaction.count(':') == 2:
                            # Check that the column is not constant (has variation)
                            if result_data[interaction].nunique() > 1:
                                # Extract the actual variables from the interaction name
                                parts = interaction.split(':')
                                self.interaction_terms.append({
                                    'name': interaction,
                                    'variables': parts,
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
        Following Sample_Analysis.ipynb logic:
        - Dummify categorical variables using pd.get_dummies
        - Drop one category per group to avoid dummy trap
        - Keep binary and continuous variables as-is

        Returns:
            pd.DataFrame: Preprocessed data with dummified categorical variables
        """
        processed = self.data.copy()

        # Handle missing values
        processed = processed.dropna(subset=[self.treatment, self.outcome])

        # Track which columns to drop (one per category to avoid dummy trap)
        columns_to_drop = []

        # Store mapping of original categorical variable to its dummified columns
        self.categorical_mapping = {}

        # Process each variable based on its type
        for col, dtype in self.column_types.items():
            if col not in processed.columns:
                continue

            if dtype in ['categorical', 'ordinal']:
                # Dummify categorical/ordinal variables
                print(f"Dummifying categorical variable: {col}")
                dummies = pd.get_dummies(processed[col], prefix=col, dtype=int)

                # Store the mapping
                self.categorical_mapping[col] = list(dummies.columns)

                # Add dummies to processed data
                processed = pd.concat([processed, dummies], axis=1)

                # Drop one category to avoid dummy trap (drop first category)
                first_dummy = dummies.columns[0]
                columns_to_drop.append(first_dummy)
                print(f"  Created {len(dummies.columns)} dummy columns, will drop {first_dummy} to avoid dummy trap")

                # Remove original categorical column
                processed = processed.drop(col, axis=1)

            elif dtype == 'binary':
                # Keep binary variables as-is, just ensure they're numeric
                processed[col] = pd.to_numeric(processed[col], errors='coerce')
            else:
                # Continuous variables - keep as-is
                processed[col] = pd.to_numeric(processed[col], errors='coerce')

        # Drop one category per group to avoid dummy trap
        if columns_to_drop:
            print(f"\nDropping {len(columns_to_drop)} columns to avoid dummy trap:")
            for col in columns_to_drop:
                print(f"  - {col}")
            processed = processed.drop(columns_to_drop, axis=1, errors='ignore')

        # Convert all remaining columns to numeric
        for col in processed.columns:
            try:
                processed[col] = pd.to_numeric(processed[col], errors='coerce')
            except:
                pass

        # Drop any remaining NaN
        initial_rows = len(processed)
        processed = processed.dropna()
        final_rows = len(processed)
        if initial_rows != final_rows:
            print(f"\nDropped {initial_rows - final_rows} rows due to missing values")

        print(f"\nFinal preprocessed data shape: {processed.shape}")
        print(f"Columns: {list(processed.columns)}")

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
