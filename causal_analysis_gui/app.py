"""
Interactive Causal Analysis Tool
A GUI application for Double Machine Learning (DML) causal inference
Based on the DoWhy/EconML framework
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from components.data_loader import DataLoader
from components.dag_editor import DAGEditor
from components.variable_config import VariableConfigurator
from components.dml_estimator import DMLEstimator
from utils.graph_utils import GraphValidator

# Page configuration
st.set_page_config(
    page_title="Causal Analysis Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'data' not in st.session_state:
    st.session_state.data = None
if 'column_types' not in st.session_state:
    st.session_state.column_types = {}
if 'dag' not in st.session_state:
    st.session_state.dag = None
if 'treatment' not in st.session_state:
    st.session_state.treatment = None
if 'outcome' not in st.session_state:
    st.session_state.outcome = None
if 'results' not in st.session_state:
    st.session_state.results = None

def main():
    """Main application flow"""

    # Title and description
    st.title("🔬 Interactive Causal Analysis Tool")
    st.markdown("""
    This tool enables you to perform **Double Machine Learning (DML)** causal analysis
    on your own data. Upload a CSV file, define the causal structure, and estimate treatment effects.
    """)

    # Sidebar for navigation
    with st.sidebar:
        st.header("Workflow Steps")
        step = st.radio(
            "Current Step:",
            [
                "1️⃣ Upload Data",
                "2️⃣ Configure Data Types",
                "3️⃣ Build Causal DAG",
                "4️⃣ Specify Variables",
                "5️⃣ Run DML Analysis",
                "6️⃣ View Results"
            ],
            index=st.session_state.step - 1
        )
        st.session_state.step = int(step[0])

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Double Machine Learning** combines:
        - Causal inference theory (DAGs)
        - Machine learning (LASSO, Random Forests)
        - Rigorous statistical estimation

        Built with DoWhy + EconML
        """)

    # Main content area based on current step
    if st.session_state.step == 1:
        step_1_upload_data()
    elif st.session_state.step == 2:
        step_2_configure_types()
    elif st.session_state.step == 3:
        step_3_build_dag()
    elif st.session_state.step == 4:
        step_4_specify_variables()
    elif st.session_state.step == 5:
        step_5_run_analysis()
    elif st.session_state.step == 6:
        step_6_view_results()


def step_1_upload_data():
    """Step 1: Upload CSV file"""
    st.header("Step 1: Upload Your Data")

    st.markdown("""
    Upload a CSV file containing your data. Each column will become a **node** in the causal DAG.

    **Requirements:**
    - CSV format with headers
    - No missing values in key variables
    - At least one treatment and one outcome variable
    """)

    data_loader = DataLoader()
    data = data_loader.load_data()

    if data is not None:
        st.session_state.data = data

        # Show data preview
        st.success(f"✅ Data loaded successfully! Shape: {data.shape}")

        with st.expander("📊 Data Preview", expanded=True):
            st.dataframe(data.head(20), use_container_width=True)

        with st.expander("📈 Basic Statistics"):
            st.dataframe(data.describe(), use_container_width=True)

        # Button to proceed
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("➡️ Proceed to Configure Data Types", type="primary", use_container_width=True):
                st.session_state.step = 2
                st.rerun()


def step_2_configure_types():
    """Step 2: Configure data types for each variable"""
    st.header("Step 2: Configure Data Types")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first (Step 1)")
        return

    st.markdown("""
    Specify the **data type** for each variable. This helps the DML algorithm
    choose appropriate models and preprocessing steps.
    """)

    configurator = VariableConfigurator(st.session_state.data)
    column_types = configurator.configure_types()

    if column_types:
        st.session_state.column_types = column_types

        # Show summary
        st.success("✅ Data types configured!")

        with st.expander("📋 Configuration Summary"):
            type_df = pd.DataFrame([
                {"Column": col, "Type": dtype}
                for col, dtype in column_types.items()
            ])
            st.dataframe(type_df, use_container_width=True)

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Back to Upload", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        with col3:
            if st.button("➡️ Proceed to Build DAG", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()


def step_3_build_dag():
    """Step 3: Build the causal DAG"""
    st.header("Step 3: Build Causal DAG")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first (Step 1)")
        return

    st.markdown("""
    Create a **Directed Acyclic Graph (DAG)** representing your causal assumptions.
    - Each column is a **node**
    - Draw **directed edges** from causes to effects
    - The graph must be **acyclic** (no loops)
    """)

    dag_editor = DAGEditor(list(st.session_state.data.columns))
    dag = dag_editor.create_dag()

    if dag is not None:
        st.session_state.dag = dag

        # Validate DAG
        validator = GraphValidator()
        is_valid, message = validator.validate_dag(dag)

        if is_valid:
            st.success(f"✅ Valid DAG with {dag.number_of_nodes()} nodes and {dag.number_of_edges()} edges")
        else:
            st.error(f"❌ Invalid DAG: {message}")
            return

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Back to Data Types", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
        with col3:
            if is_valid and st.button("➡️ Proceed to Specify Variables", type="primary", use_container_width=True):
                st.session_state.step = 4
                st.rerun()


def step_4_specify_variables():
    """Step 4: Specify treatment and outcome variables"""
    st.header("Step 4: Specify Treatment & Outcome")

    if st.session_state.dag is None:
        st.warning("⚠️ Please create a DAG first (Step 3)")
        return

    st.markdown("""
    Select which variables represent:
    - **Treatment**: The intervention or exposure of interest
    - **Outcome**: The result or effect you want to measure
    """)

    # Treatment selection
    st.subheader("🎯 Treatment Variable")
    treatment = st.selectbox(
        "Select the treatment variable:",
        options=list(st.session_state.data.columns),
        index=None,
        help="The variable whose causal effect you want to estimate"
    )

    # Outcome selection
    st.subheader("📊 Outcome Variable")
    outcome = st.selectbox(
        "Select the outcome variable:",
        options=list(st.session_state.data.columns),
        index=None,
        help="The variable that is affected by the treatment"
    )

    if treatment and outcome:
        if treatment == outcome:
            st.error("❌ Treatment and outcome must be different variables!")
            return

        st.session_state.treatment = treatment
        st.session_state.outcome = outcome

        # Show confounders from DAG
        dag = st.session_state.dag

        # Find confounders (common causes of treatment and outcome)
        confounders = set()
        for node in dag.nodes():
            if node != treatment and node != outcome:
                if dag.has_edge(node, treatment) and dag.has_edge(node, outcome):
                    confounders.add(node)

        if confounders:
            st.info(f"📌 Identified confounders from DAG: {', '.join(confounders)}")
        else:
            st.warning("⚠️ No confounders identified. Make sure your DAG includes common causes.")

        # Show causal path
        st.success(f"✅ Treatment: **{treatment}** → Outcome: **{outcome}**")

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Back to DAG", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
        with col3:
            if st.button("➡️ Proceed to Run Analysis", type="primary", use_container_width=True):
                st.session_state.step = 5
                st.rerun()


def step_5_run_analysis():
    """Step 5: Run DML analysis"""
    st.header("Step 5: Run DML Analysis")

    if st.session_state.treatment is None or st.session_state.outcome is None:
        st.warning("⚠️ Please specify treatment and outcome variables (Step 4)")
        return

    st.markdown("""
    Ready to estimate the causal effect! The analysis will:
    1. Identify confounders from the DAG
    2. Use Double Machine Learning with LASSO
    3. Estimate the Average Treatment Effect (ATE)
    4. Provide confidence intervals
    """)

    # Model configuration
    with st.expander("⚙️ Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            discrete_treatment = st.checkbox(
                "Discrete Treatment",
                value=st.session_state.column_types.get(st.session_state.treatment, 'continuous') in ['binary', 'categorical'],
                help="Check if treatment is binary or categorical"
            )

        with col2:
            n_splits = st.number_input(
                "Cross-validation folds",
                min_value=2,
                max_value=10,
                value=5,
                help="Number of folds for cross-fitting"
            )

    # Run analysis button
    if st.button("🚀 Run DML Analysis", type="primary", use_container_width=True):
        with st.spinner("Running Double Machine Learning... This may take a few minutes."):
            try:
                estimator = DMLEstimator(
                    data=st.session_state.data,
                    dag=st.session_state.dag,
                    treatment=st.session_state.treatment,
                    outcome=st.session_state.outcome,
                    column_types=st.session_state.column_types
                )

                results = estimator.estimate_ate(
                    discrete_treatment=discrete_treatment,
                    n_splits=n_splits
                )

                st.session_state.results = results
                st.success("✅ Analysis complete!")

                # Show quick summary
                st.metric(
                    label=f"Average Treatment Effect of {st.session_state.treatment}",
                    value=f"{results['ate']:.4f}",
                    delta=f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]"
                )

                # Auto-proceed to results
                if st.button("➡️ View Detailed Results", type="primary", use_container_width=True):
                    st.session_state.step = 6
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)

    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back to Variables", use_container_width=True):
            st.session_state.step = 4
            st.rerun()


def step_6_view_results():
    """Step 6: View results and visualizations"""
    st.header("Step 6: Results")

    if st.session_state.results is None:
        st.warning("⚠️ Please run the analysis first (Step 5)")
        return

    results = st.session_state.results

    st.markdown(f"""
    ### Causal Effect Estimation
    **Treatment:** {st.session_state.treatment}
    **Outcome:** {st.session_state.outcome}
    """)

    # Main results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Average Treatment Effect (ATE)", f"{results['ate']:.4f}")

    with col2:
        st.metric("Standard Error", f"{results['se']:.4f}")

    with col3:
        st.metric("P-value", f"{results.get('p_value', 'N/A')}")

    # Confidence interval
    st.markdown("#### 95% Confidence Interval")
    st.info(f"[{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")

    # Interpretation
    st.markdown("#### 📝 Interpretation")
    if results.get('p_value') and results['p_value'] < 0.05:
        st.success(f"""
        The treatment **{st.session_state.treatment}** has a **statistically significant** effect on
        **{st.session_state.outcome}**. On average, the treatment causes a change of
        **{results['ate']:.4f}** units in the outcome.
        """)
    else:
        st.warning(f"""
        The effect of **{st.session_state.treatment}** on **{st.session_state.outcome}** is not
        statistically significant at the 0.05 level. We cannot confidently conclude there is a causal effect.
        """)

    # Additional visualizations
    with st.expander("📊 Detailed Results"):
        if 'model_summary' in results:
            st.text(results['model_summary'])

    # Export results
    st.markdown("#### 💾 Export Results")

    results_df = pd.DataFrame({
        'Treatment': [st.session_state.treatment],
        'Outcome': [st.session_state.outcome],
        'ATE': [results['ate']],
        'Standard Error': [results['se']],
        'CI Lower': [results['ci_lower']],
        'CI Upper': [results['ci_upper']],
        'P-value': [results.get('p_value', 'N/A')]
    })

    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name="causal_analysis_results.csv",
        mime="text/csv"
    )

    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back to Analysis", use_container_width=True):
            st.session_state.step = 5
            st.rerun()
    with col2:
        if st.button("🔄 Start New Analysis", use_container_width=True):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
