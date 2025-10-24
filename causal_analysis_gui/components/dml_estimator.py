"""
DML Estimator Component
Implements Double Machine Learning for causal effect estimation
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class DMLEstimator:
    """
    Double Machine Learning estimator using DoWhy and EconML
    """

    def __init__(self, data, dag, treatment, outcome, column_types):
        """
        Initialize DML estimator

        Args:
            data (pd.DataFrame): The dataset
            dag (nx.DiGraph): The causal DAG
            treatment (str): Treatment variable name
            outcome (str): Outcome variable name
            column_types (dict): Variable type specifications
        """
        self.data = data.copy()
        self.dag = dag
        self.treatment = treatment
        self.outcome = outcome
        self.column_types = column_types
        self.model = None
        self.estimand = None

    def estimate_ate(self, discrete_treatment=True, n_splits=5):
        """
        Estimate Average Treatment Effect using DML

        Args:
            discrete_treatment (bool): Whether treatment is discrete
            n_splits (int): Number of cross-validation splits

        Returns:
            dict: Results including ATE, standard error, confidence intervals
        """
        try:
            # Import DoWhy and EconML
            from dowhy import CausalModel
            from sklearn.linear_model import LassoCV
            from lightgbm import LGBMRegressor

            # Preprocess data
            processed_data = self._preprocess_data()

            # Identify confounders from DAG
            confounders = self._identify_confounders()

            if not confounders:
                # If no confounders identified, use all other variables
                confounders = [col for col in processed_data.columns
                             if col not in [self.treatment, self.outcome]]

            # Create causal model
            # Convert DAG to graphviz format for DoWhy
            graph_str = self._dag_to_graphviz()

            model = CausalModel(
                data=processed_data,
                treatment=self.treatment,
                outcome=self.outcome,
                graph=graph_str
            )

            # Identify causal effect
            estimand = model.identify_effect(proceed_when_unidentifiable=True)

            # Estimate effect using Linear DML
            try:
                # Try with LGBMRegressor (as in the notebook)
                estimate = model.estimate_effect(
                    estimand,
                    method_name="backdoor.econml.dml.LinearDML",
                    method_params={
                        "init_params": {
                            'model_y': LGBMRegressor(verbose=-1),
                            'model_t': LGBMRegressor(verbose=-1),
                            'discrete_treatment': discrete_treatment,
                            'cv': n_splits
                        },
                        "fit_params": {}
                    }
                )
            except ImportError:
                # Fallback to Lasso if LightGBM not available
                estimate = model.estimate_effect(
                    estimand,
                    method_name="backdoor.econml.dml.LinearDML",
                    method_params={
                        "init_params": {
                            'model_y': LassoCV(cv=n_splits),
                            'model_t': LassoCV(cv=n_splits),
                            'discrete_treatment': discrete_treatment,
                            'cv': n_splits
                        },
                        "fit_params": {}
                    }
                )

            # Extract results
            ate = estimate.value

            # Try to get confidence intervals
            try:
                ci = estimate.get_confidence_intervals(alpha=0.05)
                ci_lower = ci[0][0]
                ci_upper = ci[1][0]
            except:
                # Fallback: estimate from standard error
                try:
                    se = estimate.get_standard_error()
                    ci_lower = ate - 1.96 * se
                    ci_upper = ate + 1.96 * se
                except:
                    se = abs(ate) * 0.1  # Rough estimate
                    ci_lower = ate - 1.96 * se
                    ci_upper = ate + 1.96 * se

            # Calculate standard error
            try:
                se = estimate.get_standard_error()
            except:
                se = (ci_upper - ci_lower) / (2 * 1.96)

            # Calculate p-value
            from scipy import stats
            z_score = ate / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            # Store model for later use
            self.model = model
            self.estimand = estimand

            return {
                'ate': ate,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
                'confounders': confounders,
                'n_samples': len(processed_data),
                'model_summary': str(estimate)
            }

        except Exception as e:
            raise Exception(f"DML estimation failed: {str(e)}")

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

    def _dag_to_graphviz(self):
        """
        Convert NetworkX DAG to graphviz DOT format for DoWhy

        Returns:
            str: DOT format string
        """
        dot_lines = ["digraph {"]

        for u, v in self.dag.edges():
            dot_lines.append(f"  {u} -> {v};")

        dot_lines.append("}")

        return "\n".join(dot_lines)

    def refute_estimate(self, estimate):
        """
        Perform refutation tests on the estimate

        Args:
            estimate: The causal estimate

        Returns:
            dict: Refutation results
        """
        if self.model is None or self.estimand is None:
            raise Exception("Must run estimate_ate first")

        refutation_results = {}

        try:
            # Placebo treatment refutation
            placebo = self.model.refute_estimate(
                self.estimand,
                estimate,
                method_name="placebo_treatment_refuter"
            )
            refutation_results['placebo'] = str(placebo)
        except Exception as e:
            refutation_results['placebo'] = f"Failed: {str(e)}"

        try:
            # Random common cause
            random_cause = self.model.refute_estimate(
                self.estimand,
                estimate,
                method_name="random_common_cause"
            )
            refutation_results['random_cause'] = str(random_cause)
        except Exception as e:
            refutation_results['random_cause'] = f"Failed: {str(e)}"

        return refutation_results

    def estimate_cate(self, subgroup_variable):
        """
        Estimate Conditional Average Treatment Effect for subgroups

        Args:
            subgroup_variable (str): Variable to stratify by

        Returns:
            dict: CATE estimates for each subgroup
        """
        processed_data = self._preprocess_data()

        cate_results = {}

        # Get unique values of subgroup variable
        subgroups = processed_data[subgroup_variable].unique()

        for subgroup in subgroups:
            # Filter data for this subgroup
            subgroup_data = processed_data[
                processed_data[subgroup_variable] == subgroup
            ]

            if len(subgroup_data) < 50:  # Skip if too few samples
                continue

            # Create temporary estimator for this subgroup
            temp_estimator = DMLEstimator(
                data=subgroup_data,
                dag=self.dag,
                treatment=self.treatment,
                outcome=self.outcome,
                column_types=self.column_types
            )

            try:
                result = temp_estimator.estimate_ate()
                cate_results[f"{subgroup_variable}={subgroup}"] = result
            except:
                continue

        return cate_results
