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
    page_title="Graph-Based Double Machine Learning",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0  # Start with explanation page
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
if 'interaction_variables' not in st.session_state:
    st.session_state.interaction_variables = []
if 'results' not in st.session_state:
    st.session_state.results = None

def main():
    """Main application flow"""

    # Title and description
    st.title("🔬 Graph-Based Double Machine Learning")
    st.markdown("""
    This tool enables you to discover **significant interaction effects** using **Graph-Based Double Machine Learning (DML)**.
    Define causal structures, select interactions of interest, and robustly estimate heterogeneous treatment effects.
    """)

    # Sidebar for navigation
    with st.sidebar:
        st.header("Workflow Steps")
        step = st.radio(
            "Current Step:",
            [
                "0️⃣ Workflow Overview",
                "1️⃣ Upload Data",
                "2️⃣ Configure Data Types",
                "3️⃣ Build Causal DAG",
                "4️⃣ Specify Interactions",
                "5️⃣ Specify Variables",
                "6️⃣ Run DML Analysis",
                "7️⃣ View Results"
            ],
            index=st.session_state.step
        )
        st.session_state.step = int(step[0])

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Graph-Based DML** combines:
        - Causal inference theory (DAGs)
        - Machine learning (LASSO, LightGBM)
        - Interaction effect discovery
        - Robust statistical estimation

        Focus: Finding significant **two-way and three-way interactions**

        Built with DoWhy + EconML
        """)

    # Main content area based on current step
    if st.session_state.step == 0:
        step_0_workflow_overview()
    elif st.session_state.step == 1:
        step_1_upload_data()
    elif st.session_state.step == 2:
        step_2_configure_types()
    elif st.session_state.step == 3:
        step_3_build_dag()
    elif st.session_state.step == 4:
        step_4_specify_interactions()
    elif st.session_state.step == 5:
        step_5_specify_variables()
    elif st.session_state.step == 6:
        step_6_run_analysis()
    elif st.session_state.step == 7:
        step_7_view_results()


def step_0_workflow_overview():
    """Step 0: Workflow explanation and overview"""
    st.header("📚 Workflow Overview: Graph-Based Double Machine Learning")

    st.markdown("""
    ### What is Graph-Based Double Machine Learning?

    **Graph-Based Double Machine Learning** is a powerful methodology that combines:
    - **Directed Acyclic Graphs (DAGs)** to represent causal assumptions
    - **Double Machine Learning (DML)** to handle high-dimensional settings
    - **Interaction term exploration** to discover heterogeneous effects

    This approach enables you to **robustly explore and estimate interaction effects** without
    the limitations of traditional regression that requires pre-selecting a small subset of interactions.

    ---

    ### The Workflow

    This tool guides you through a systematic 7-step process:
    """)

    # Display workflow steps
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        #### Steps:
        1. **Upload Data**
        2. **Configure Data Types**
        3. **Build Causal DAG**
        4. **Specify Interactions**
        5. **Specify Variables**
        6. **Run DML Analysis**
        7. **View Results**
        """)

    with col2:
        st.markdown("""
        #### What Happens:
        - Load your CSV dataset
        - Identify variable types (continuous, binary, categorical)
        - Encode your causal assumptions in a DAG
        - **Select variables for interaction term construction**
        - Choose treatment and outcome variables
        - Estimate effects using DML with LASSO/LightGBM
        - Review significant interaction terms
        """)

    st.markdown("---")

    # Workflow diagram (text-based representation)
    st.subheader("🔄 Visual Workflow")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   Identify Phenomenon & Target Theory              │
    │                                                     │
    └────────────────┬────────────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────────────────┐
    │  GRAPH-BASED DOUBLE MACHINE LEARNING                │
    │                                                     │
    │  1. Encode Existing Knowledge into DAG              │
    │  2. Collect and Pre-Process Data                    │
    │  3. Specify Learners (e.g., LASSO)                 │
    │  4. Specify Interactions of Interest   ◄────────┐  │
    │  5. Fit Treatment & Outcome Models                  │
    │  6. Perform Sensitivity Analysis                    │
    │  7. Select Robust Interactions                      │
    │                                                     │
    └────────────────┬────────────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   Formulate Implications for Theory and Practice    │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("---")

    # Key concepts
    st.subheader("🔑 Key Concepts")

    with st.expander("📊 What are Interaction Effects?"):
        st.markdown("""
        **Interaction effects** (or effect modifiers) occur when the effect of one variable on an outcome
        depends on the level of another variable.

        **Example:** The effect of obtaining a college degree on salary might differ by:
        - Gender (two-way interaction: Degree × Gender)
        - Gender AND family wealth (three-way interaction: Degree × Gender × Wealth)

        Traditional regression typically pre-selects a few interactions to avoid overfitting.
        Graph-Based DML allows you to explore ALL possible interactions robustly.
        """)

    with st.expander("🎯 Why Double Machine Learning?"):
        st.markdown("""
        **Double Machine Learning (DML)** addresses key challenges:

        1. **Regularization Bias**: ML models like LASSO shrink coefficients, which can bias
           causal estimates. DML uses orthogonalization to eliminate this bias.

        2. **High-Dimensional Settings**: When you have many variables and interaction terms,
           traditional OLS suffers from:
           - Overfitting
           - Multicollinearity
           - Poor generalization

        3. **Valid Inference**: DML provides statistically valid p-values and confidence intervals
           even with ML models.

        **How it works:**
        - Separately estimates parts of treatment and outcome influenced by confounders
        - Uses residuals (free from confounding) to estimate the direct causal effect
        - Applies cross-fitting to avoid overfitting
        """)

    with st.expander("📈 Why Directed Acyclic Graphs (DAGs)?"):
        st.markdown("""
        **DAGs** help you:

        1. **Encode Causal Assumptions**: Explicitly represent your theory about how variables relate
        2. **Identify Confounders**: Systematically find variables that need to be controlled
        3. **Avoid Bad Controls**: Prevent including mediators or colliders that bias estimates
        4. **Make Assumptions Transparent**: Others can scrutinize and critique your causal model

        **Important:** DAGs are theory-driven. The quality of your causal estimates depends on
        correctly specifying the graph based on domain knowledge.
        """)

    st.markdown("---")

    # Focus of this tool
    st.info("""
    ### 🎯 Focus of This Tool

    This application is specifically designed to help you **discover significant interaction effects**:

    - **Two-way interactions**: How two variables jointly modify treatment effects
    - **Three-way interactions**: How three variables jointly modify treatment effects

    By algorithmically handling interaction terms, you can uncover complex patterns in your data
    that traditional analysis might miss, while maintaining statistical rigor through DML's
    doubly-robust estimation and sensitivity analysis.
    """)

    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("➡️ Start Analysis", type="primary", use_container_width=True):
            st.session_state.step = 1
            st.rerun()


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
            if is_valid and st.button("➡️ Proceed to Specify Interactions", type="primary", use_container_width=True):
                st.session_state.step = 4
                st.rerun()


def step_4_specify_interactions():
    """Step 4: Specify which variables to use for interaction terms"""
    st.header("Step 4: Specify Interactions of Interest")

    if st.session_state.dag is None:
        st.warning("⚠️ Please create a DAG first (Step 3)")
        return

    st.markdown("""
    Select variables to construct **interaction terms**. The tool will create:
    - **Two-way interactions**: All pairwise combinations (e.g., A×B, A×C, B×C)
    - **Three-way interactions**: All three-variable combinations (e.g., A×B×C)

    ### Why Select a Subset?
    While Graph-Based DML can handle many interaction terms, focusing on theoretically-relevant
    variables helps maintain interpretability and computational efficiency.

    **Tip:** Include variables that are root nodes in your DAG (confounders, not mediators).
    """)

    # Get all variables from DAG
    all_variables = list(st.session_state.dag.nodes())

    if not all_variables:
        st.error("❌ No variables found in DAG")
        return

    # Identify root nodes (variables with no predecessors) - these are good candidates
    root_nodes = [node for node in all_variables if st.session_state.dag.in_degree(node) == 0]

    st.info(f"💡 **Root nodes in your DAG** (good candidates for interactions): {', '.join(root_nodes) if root_nodes else 'None'}")

    st.subheader("Select Variables for Interaction Terms")

    # Multi-select for interaction variables
    selected_vars = st.multiselect(
        "Choose variables to include in interaction terms:",
        options=all_variables,
        default=st.session_state.interaction_variables if st.session_state.interaction_variables else None,
        help="Select 2 or more variables. The tool will create all possible 2-way and 3-way interactions."
    )

    if selected_vars:
        st.session_state.interaction_variables = selected_vars

        # Calculate number of interactions that will be created
        from itertools import combinations
        n_vars = len(selected_vars)
        n_two_way = len(list(combinations(selected_vars, 2)))
        n_three_way = len(list(combinations(selected_vars, 3))) if n_vars >= 3 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Variables Selected", n_vars)
        with col2:
            st.metric("Two-Way Interactions", n_two_way)
        with col3:
            st.metric("Three-Way Interactions", n_three_way)

        # Show preview of interactions
        with st.expander("📋 Preview of Interaction Terms", expanded=True):
            st.markdown("**Two-way interactions:**")
            two_way_interactions = [f"{a} × {b}" for a, b in combinations(selected_vars, 2)]
            if two_way_interactions:
                # Display in columns
                cols = st.columns(3)
                for i, interaction in enumerate(two_way_interactions):
                    cols[i % 3].markdown(f"- {interaction}")
            else:
                st.markdown("*None (need at least 2 variables)*")

            if n_vars >= 3:
                st.markdown("**Three-way interactions:**")
                three_way_interactions = [f"{a} × {b} × {c}" for a, b, c in combinations(selected_vars, 3)]
                if three_way_interactions:
                    cols = st.columns(3)
                    for i, interaction in enumerate(three_way_interactions):
                        cols[i % 3].markdown(f"- {interaction}")
            else:
                st.markdown("**Three-way interactions:** *Need at least 3 variables*")

        # Warning for too many interactions
        total_interactions = n_two_way + n_three_way
        if total_interactions > 100:
            st.warning(f"⚠️ You've selected {total_interactions} interaction terms. This may take a long time to compute. Consider selecting fewer variables.")
        elif total_interactions > 50:
            st.info(f"ℹ️ You've selected {total_interactions} interaction terms. Analysis may take several minutes.")

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Back to DAG", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
        with col3:
            if len(selected_vars) >= 2 and st.button("➡️ Proceed to Specify Variables", type="primary", use_container_width=True):
                st.session_state.step = 5
                st.rerun()

    else:
        st.warning("⚠️ Please select at least 2 variables to create interaction terms.")

        # Navigation buttons (without proceed)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Back to DAG", use_container_width=True):
                st.session_state.step = 3
                st.rerun()


def step_5_specify_variables():
    """Step 5: Specify treatment and outcome variables"""
    st.header("Step 5: Specify Treatment & Outcome")

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
            if st.button("⬅️ Back to Interactions", use_container_width=True):
                st.session_state.step = 4
                st.rerun()
        with col3:
            if st.button("➡️ Proceed to Run Analysis", type="primary", use_container_width=True):
                st.session_state.step = 6
                st.rerun()


def step_6_run_analysis():
    """Step 6: Run DML analysis"""
    st.header("Step 6: Run DML Analysis")

    if st.session_state.treatment is None or st.session_state.outcome is None:
        st.warning("⚠️ Please specify treatment and outcome variables (Step 5)")
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
                    column_types=st.session_state.column_types,
                    interaction_variables=st.session_state.interaction_variables
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
                    st.session_state.step = 7
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)

    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back to Variables", use_container_width=True):
            st.session_state.step = 5
            st.rerun()


def step_7_view_results():
    """Step 7: View results and visualizations"""
    st.header("Step 7: Results")

    if st.session_state.results is None:
        st.warning("⚠️ Please run the analysis first (Step 6)")
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

    # Display interaction term results prominently
    if results.get('interaction_results') and len(results['interaction_results']) > 0:
        st.markdown("---")
        st.markdown("### 🔍 Interaction Term Analysis")
        st.markdown("""
        The following shows the estimated effects of interaction terms on the outcome.
        **Significant interactions** indicate heterogeneous treatment effects and can reveal
        important moderating relationships.
        """)

        interaction_results = results['interaction_results']

        # Summary metrics
        total_interactions = len(interaction_results)
        significant_interactions = sum(1 for r in interaction_results if r.get('significant', False))
        two_way = sum(1 for r in interaction_results if r.get('order') == 2)
        three_way = sum(1 for r in interaction_results if r.get('order') == 3)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Interactions", total_interactions)
        with col2:
            st.metric("Significant (p<0.05)", significant_interactions)
        with col3:
            st.metric("Two-way", two_way)
        with col4:
            st.metric("Three-way", three_way)

        # Show significant interactions first
        significant = [r for r in interaction_results if r.get('significant', False)]
        non_significant = [r for r in interaction_results if not r.get('significant', False)]

        if significant:
            st.markdown("#### ✅ Significant Interaction Terms")
            for result in significant:
                with st.expander(f"**{result['term']}** (p = {result['p_value']:.4f})", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("LASSO Coefficient", f"{result['lasso_coefficient']:.4f}")
                    with col2:
                        st.metric("Simple Coefficient", f"{result['simple_coefficient']:.4f}")
                    with col3:
                        st.metric("Correlation with Outcome", f"{result['correlation']:.4f}")

                    st.markdown(f"""
                    - **Variables:** {', '.join(result['variables'])}
                    - **Order:** {result['order']}-way interaction
                    - **Selected by LASSO:** {'Yes' if result.get('selected_by_lasso') else 'No'}
                    - **Interpretation:** This interaction term has a statistically significant association with the outcome.
                    """)

        if non_significant:
            with st.expander(f"📊 Non-Significant Interaction Terms ({len(non_significant)})"):
                # Create a dataframe for easy viewing
                import pandas as pd
                df = pd.DataFrame([{
                    'Term': r['term'],
                    'Variables': ' × '.join(r['variables']),
                    'Order': r['order'],
                    'LASSO Coef': f"{r['lasso_coefficient']:.4f}",
                    'Simple Coef': f"{r['simple_coefficient']:.4f}",
                    'P-value': f"{r['p_value']:.4f}",
                    'Selected': 'Yes' if r.get('selected_by_lasso') else 'No'
                } for r in non_significant])
                st.dataframe(df, use_container_width=True)

        # Export interaction results
        st.markdown("#### 💾 Export Interaction Results")
        interaction_df = pd.DataFrame(interaction_results)
        interaction_csv = interaction_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Interaction Results (CSV)",
            data=interaction_csv,
            file_name="interaction_analysis_results.csv",
            mime="text/csv"
        )

    elif results.get('interaction_terms') and len(results['interaction_terms']) > 0:
        st.info("Interaction terms were included in the analysis, but detailed results are not available.")
    else:
        st.info("No interaction terms were specified for this analysis.")

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
            st.session_state.step = 6
            st.rerun()
    with col2:
        if st.button("🔄 Start New Analysis", use_container_width=True):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
