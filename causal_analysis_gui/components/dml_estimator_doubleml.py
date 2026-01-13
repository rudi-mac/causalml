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
                 interaction_variables=None, categorical_drop_values=None):
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
            categorical_drop_values (dict): Dict mapping categorical variable names to values to drop (reference categories)
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
        self.categorical_drop_values = categorical_drop_values or {}
        self.model = None
        self.interaction_terms = []
        self.categorical_mapping = {}  # Maps original categorical variable name to list of dummy columns

    def preview_treatment_variables(self):
        """
        Preview the treatment variables that will be estimated WITHOUT running the full analysis.
        This is used to show users what will be estimated before they click "Run DML Analysis".

        Returns:
            dict: {
                'main_treatment_vars': list of main treatment variable names,
                'two_way_interaction_vars': list of 2-way interaction names,
                'three_way_interaction_vars': list of 3-way interaction names,
                'treatment_variables': complete ordered list of all treatment variables
            }
        """
        # Preprocess data (includes dummification of categorical variables)
        processed_data = self._preprocess_data()

        # Construct interaction terms
        processed_data = self._construct_interaction_terms(processed_data)

        # Build treatment_variables list following the same logic as estimate_ate
        treatment_variables = []

        # Step 1: Add ONLY the main treatment variable (the one selected in Step 1)
        if hasattr(self, 'categorical_mapping') and self.treatment in self.categorical_mapping:
            for dummy_col in self.categorical_mapping[self.treatment]:
                if dummy_col in processed_data.columns:
                    treatment_variables.append(dummy_col)
        elif self.treatment in processed_data.columns:
            treatment_variables.append(self.treatment)

        # Step 2: Add interaction terms
        for term in self.interaction_terms:
            if term['name'] in processed_data.columns:
                treatment_variables.append(term['name'])

        # Categorize treatment variables
        interaction_names = [t['name'] for t in self.interaction_terms]
        main_treatment_vars = [tv for tv in treatment_variables if tv not in interaction_names]
        two_way_interaction_vars = [t['name'] for t in self.interaction_terms if t['order'] == 2]
        three_way_interaction_vars = [t['name'] for t in self.interaction_terms if t['order'] == 3]

        return {
            'main_treatment_vars': main_treatment_vars,
            'two_way_interaction_vars': two_way_interaction_vars,
            'three_way_interaction_vars': three_way_interaction_vars,
            'treatment_variables': treatment_variables
        }

    def estimate_ate(self, discrete_treatment=True, n_splits=5, alpha=0.05, progress_callback=None):
        """
        Estimate Average Treatment Effect using DML with doubleml library
        Estimates main effect and all interaction effects

        Args:
            discrete_treatment (bool): Whether treatment is discrete
            n_splits (int): Number of cross-validation splits
            alpha (float): Significance level for confidence intervals
            progress_callback (callable): Optional callback function for progress updates
                                        Should accept (current, total, message) parameters

        Returns:
            dict: Results including ATE, standard error, confidence intervals, interaction results
        """
        try:
            import doubleml as dml
            from scipy import stats

            # Preprocess data (includes dummification of categorical variables)
            if progress_callback:
                progress_callback(0.1, 1.0, "Preprocessing data and handling categorical variables...")
            processed_data = self._preprocess_data()

            # Construct interaction terms
            if progress_callback:
                progress_callback(0.2, 1.0, "Constructing interaction terms...")
            processed_data = self._construct_interaction_terms(processed_data)

            # Create list of treatment variables to estimate:
            # IMPORTANT: Main treatment is ONLY the treatment variable from Step 1
            # All 2-way and 3-way interactions must include this treatment variable
            # (User requirement: "the main treatment effect can only be one variable")
            treatment_variables = []

            # Step 1: Add ONLY the main treatment variable (the one selected in Step 1)
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

            # Step 2: Add interaction terms as additional treatments
            # These include all 2-way and 3-way interactions (which already include treatment by construction)
            for term in self.interaction_terms:
                if term['name'] in processed_data.columns:
                    treatment_variables.append(term['name'])

            # Count main treatment effects and interaction terms
            # Main treatments are those NOT in interaction_terms list
            interaction_names = [t['name'] for t in self.interaction_terms]
            n_main_treatment = len([tv for tv in treatment_variables if tv not in interaction_names])
            n_two_way = len([t for t in self.interaction_terms if t['order'] == 2])
            n_three_way = len([t for t in self.interaction_terms if t['order'] == 3])

            print(f"\nEstimating effects for {len(treatment_variables)} treatments:")
            print(f"  - Main treatment effects: {n_main_treatment}")
            print(f"  - 2-way interaction terms: {n_two_way}")
            print(f"  - 3-way interaction terms: {n_three_way}")
            print(f"  - Total: {len(treatment_variables)}")
            print(f"\nTreatment variables list: {treatment_variables}")

            # Create DoubleML data object
            if progress_callback:
                progress_callback(0.3, 1.0, f"Setting up DML model for {len(treatment_variables)} treatment variables...")
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
            if progress_callback:
                progress_callback(0.4, 1.0, f"Fitting DML model with {n_splits} folds and 5 repetitions...")
            dml_plr.fit()
            print("Model fitted successfully!")

            # Get confidence intervals (following Sample_Analysis.ipynb)
            print("Computing confidence intervals...")
            if progress_callback:
                progress_callback(0.8, 1.0, "Computing confidence intervals...")
            conf_int_95 = dml_plr.confint(level=0.95)
            conf_int_99 = dml_plr.confint(level=0.99)

            # Get p-values (following Sample_Analysis.ipynb)
            print("Extracting p-values...")
            if progress_callback:
                progress_callback(0.9, 1.0, "Extracting p-values and preparing results...")
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

            # Categorize treatment variables for display
            if progress_callback:
                progress_callback(1.0, 1.0, "Analysis complete!")
            main_treatment_vars = [tv for tv in treatment_variables if tv not in [t['name'] for t in self.interaction_terms]]
            two_way_interaction_vars = [t['name'] for t in self.interaction_terms if t['order'] == 2]
            three_way_interaction_vars = [t['name'] for t in self.interaction_terms if t['order'] == 3]

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
                'all_results': all_results,  # Include all results for display
                'main_treatment_vars': main_treatment_vars,
                'two_way_interaction_vars': two_way_interaction_vars,
                'three_way_interaction_vars': three_way_interaction_vars,
                'treatment_variables': treatment_variables  # Full list in order
            }

            return results

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"DML estimation failed: {str(e)}")

    def _construct_interaction_terms(self, data):
        """
        Construct ALL combinatorially possible interaction terms following Sample_Analysis.ipynb logic
        Steps (from Sample_Analysis.ipynb cells 35-37):
        1. Create unique_values list: all columns EXCEPT outcome (includes treatment)
        2. Generate ALL 2-way combinations from unique_values: list(combinations(unique_values, 2))
        3. Filter out interactions within same category (e.g., no Father_X:Father_Y)
        4. Generate ALL 3-way combinations from unique_values: list(combinations(unique_values, 3))
        5. Filter out interactions within same category
        6. Select only interactions that include the main treatment variable

        Args:
            data (pd.DataFrame): Preprocessed data

        Returns:
            pd.DataFrame: Data with ALL interaction terms added
        """
        print("\n" + "="*80)
        print("CONSTRUCTING INTERACTION TERMS (Sample_Analysis.ipynb logic)")
        print("="*80)

        self.interaction_terms = []

        # Following Sample_Analysis.ipynb cell 35: unique_values = causal_df.drop('Hourly_Salary_log',axis=1).columns
        # This gets ALL columns except outcome (includes treatment)
        unique_values = [col for col in data.columns if col != self.outcome]
        print(f"\nStep 1: Created unique_values list (all columns except outcome): {len(unique_values)} variables")
        print(f"unique_values = {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}")

        # ========================================
        # STEP 2: Generate ALL 2-way combinations (Sample_Analysis.ipynb cell 35)
        # ========================================
        print("\n[STEP 2] Generating all 2-way combinations from unique_values...")
        all_combinations_2w = list(combinations(unique_values, 2))
        print(f"Total 2-way combinations: {len(all_combinations_2w)}")

        # Build all 2-way interaction columns at once to avoid DataFrame fragmentation
        # CRITICAL FILTERING LOGIC (from Sample_Analysis.ipynb cell 35):
        # Only create interactions between variables from DIFFERENT categorical groups.
        # E.g., Ethnic_Asian:Ethnic_Black should NOT be created (both start with "Ethnic")
        # But Ethnic_Asian:Region_South SHOULD be created (different prefixes)
        interaction_terms_2way = []
        two_way_dict = {}
        for combi in all_combinations_2w:
            # Extract category prefix (first 6 characters) to identify which categorical variable
            # each dummy column belongs to (e.g., "Ethnic", "Region", "Father", "Mother", "Wealth")
            prefix1 = combi[0][:6]
            prefix2 = combi[1][:6]

            # Following Sample_Analysis.ipynb logic: combi[0][:6] != combi[1][:6]
            # Only keep interactions where the two columns come from DIFFERENT categorical variables
            if prefix1 != prefix2:
                column_name = f"{combi[0]}:{combi[1]}"
                two_way_dict[column_name] = data[combi[0]] * data[combi[1]]
                interaction_terms_2way.append(column_name)

        print(f"2-way interactions created (after filtering): {len(interaction_terms_2way)}")

        # ========================================
        # STEP 3: Generate ALL 3-way combinations (Sample_Analysis.ipynb cell 37)
        # ========================================
        print("\n[STEP 3] Generating all 3-way combinations from unique_values...")
        all_combinations_3w = list(combinations(unique_values, 3))
        print(f"Total 3-way combinations: {len(all_combinations_3w)}")

        # Build all 3-way interaction columns at once to avoid DataFrame fragmentation
        # CRITICAL FILTERING LOGIC (from Sample_Analysis.ipynb cell 37):
        # Only create interactions between variables from THREE DIFFERENT categorical groups.
        # E.g., Ethnic_Asian:Father_College:Mother_12_grades SHOULD be created (all different prefixes)
        # But Ethnic_Asian:Ethnic_Black:Region_South should NOT (two "Ethnic" columns)
        interaction_terms_3way = []
        three_way_dict = {}
        for combi in all_combinations_3w:
            # Extract category prefixes (first 6 characters) to identify which categorical variable
            # each dummy column belongs to (e.g., "Ethnic", "Region", "Father", "Mother", "Wealth")
            prefix1 = combi[0][:6]
            prefix2 = combi[1][:6]
            prefix3 = combi[2][:6]

            # Following Sample_Analysis.ipynb logic:
            # (combi[0][:6] != combi[1][:6]) & (combi[0][:6] != combi[2][:6]) & (combi[1][:6] != combi[2][:6])
            # Only keep interactions where ALL THREE columns come from DIFFERENT categorical variables
            if (prefix1 != prefix2) and (prefix1 != prefix3) and (prefix2 != prefix3):
                column_name = f"{combi[0]}:{combi[1]}:{combi[2]}"
                three_way_dict[column_name] = data[combi[0]] * data[combi[1]] * data[combi[2]]
                interaction_terms_3way.append(column_name)

        print(f"3-way interactions created (after filtering): {len(interaction_terms_3way)}")

        # Concatenate all interaction columns at once to avoid fragmentation warning
        print("\n[COMBINING] Concatenating all interaction columns to avoid DataFrame fragmentation...")
        two_way_df = pd.DataFrame(two_way_dict)
        three_way_df = pd.DataFrame(three_way_dict)
        result_data = pd.concat([data, two_way_df, three_way_df], axis=1).copy()
        print(f"Combined data shape: {result_data.shape}")

        # ========================================
        # STEP 4: Construct treatment interaction terms based on user input
        # Following Sample_Analysis.ipynb cell 12: All interactions MUST include main treatment variable
        # ========================================
        print("\n[STEP 4] Constructing treatment interaction terms based on user selection...")

        # Get list of treatment variable(s) - handle dummified treatment
        treatment_vars = []
        if hasattr(self, 'categorical_mapping') and self.treatment in self.categorical_mapping:
            treatment_vars = [d for d in self.categorical_mapping[self.treatment] if d in data.columns]
        elif self.treatment in data.columns:
            treatment_vars = [self.treatment]

        print(f"Main treatment variable(s): {treatment_vars}")

        # Legacy support
        if self.interaction_variables and not self.two_way_interaction_variables:
            self.two_way_interaction_variables = self.interaction_variables

        if not self.two_way_interaction_variables:
            print("No interaction variables specified by user - using main treatment only")
            return result_data

        # Filter interaction variables to only those that exist in data after preprocessing
        # Handle both original and dummified variable names
        valid_two_way = []
        for v in self.two_way_interaction_variables:
            if v in data.columns:
                valid_two_way.append(v)
            # Also check if variable was dummified
            elif hasattr(self, 'categorical_mapping') and v in self.categorical_mapping:
                # Add all dummy columns for this variable
                valid_two_way.extend([d for d in self.categorical_mapping[v] if d in data.columns])

        valid_three_way = []
        for v in self.three_way_interaction_variables:
            if v in data.columns:
                valid_three_way.append(v)
            # Also check if variable was dummified
            elif hasattr(self, 'categorical_mapping') and v in self.categorical_mapping:
                # Add all dummy columns for this variable
                valid_three_way.extend([d for d in self.categorical_mapping[v] if d in data.columns])

        print(f"User-selected two-way variables (after dummification): {valid_two_way}")
        print(f"User-selected three-way variables (after dummification): {valid_three_way}")

        # CRITICAL: All 2-way and 3-way interactions MUST include the main treatment variable
        # Following Sample_Analysis.ipynb logic

        # Find ALL 2-way interactions that contain treatment AND a two_way variable
        for interaction in interaction_terms_2way:
            parts = interaction.split(':')
            # Check if interaction contains any treatment variable
            has_treatment = any(tv in parts for tv in treatment_vars)

            # Only keep interactions that include treatment
            if has_treatment:
                # Also check if the other variable is in the user-selected two_way list
                other_vars = [p for p in parts if p not in treatment_vars]
                has_two_way_var = any(v in valid_two_way for v in other_vars)

                if has_two_way_var:
                    # Check that the column is not constant (has variation)
                    if result_data[interaction].nunique() > 1:
                        self.interaction_terms.append({
                            'name': interaction,
                            'variables': parts,
                            'order': 2
                        })

        # Find ALL 3-way interactions that contain treatment AND two other selected variables
        for interaction in interaction_terms_3way:
            parts = interaction.split(':')
            # Check if interaction contains treatment
            has_treatment = any(tv in parts for tv in treatment_vars)

            # Only keep interactions that include treatment
            if has_treatment:
                # Get the other two variables (not treatment)
                other_vars = [p for p in parts if p not in treatment_vars]

                # For 3-way, we need at least 2 other variables besides treatment
                # These can be from two_way list or three_way list
                all_selected_vars = list(set(valid_two_way + valid_three_way))
                matching_vars = [v for v in other_vars if v in all_selected_vars]

                # We need at least 2 matching variables (besides treatment) to form a valid 3-way interaction
                if len(matching_vars) >= 2:
                    # Check that the column is not constant (has variation)
                    if result_data[interaction].nunique() > 1:
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
        - Drop user-selected category per group to avoid dummy trap
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

                # Drop user-selected category to avoid dummy trap
                # If user hasn't specified, default to first category
                if col in self.categorical_drop_values:
                    # User specified which value to drop
                    drop_value = self.categorical_drop_values[col]
                    drop_column = f"{col}_{drop_value}"
                    if drop_column in dummies.columns:
                        columns_to_drop.append(drop_column)
                        print(f"  Created {len(dummies.columns)} dummy columns, will drop user-selected '{drop_column}' to avoid dummy trap")
                    else:
                        # Fallback if the column name doesn't match (shouldn't happen)
                        first_dummy = dummies.columns[0]
                        columns_to_drop.append(first_dummy)
                        print(f"  WARNING: Could not find '{drop_column}', dropping first category '{first_dummy}' instead")
                else:
                    # Default behavior: drop first category
                    first_dummy = dummies.columns[0]
                    columns_to_drop.append(first_dummy)
                    print(f"  Created {len(dummies.columns)} dummy columns, will drop first category '{first_dummy}' to avoid dummy trap")

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
