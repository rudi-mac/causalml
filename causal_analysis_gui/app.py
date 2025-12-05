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
from components.dml_estimator_doubleml import DMLEstimatorDoubleML
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
if 'dag_variables' not in st.session_state:
    st.session_state.dag_variables = {}  # Dict of variable_name: {'type': 'continuous/binary/categorical/ordinal'}
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
                "1️⃣ Build Causal DAG",
                "2️⃣ Upload Data",
                "3️⃣ Specify Interactions",
                "4️⃣ Run DML Analysis",
                "5️⃣ View Results"
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
        step_1_build_dag()
    elif st.session_state.step == 2:
        step_2_upload_data()
    elif st.session_state.step == 3:
        step_3_specify_interactions()
    elif st.session_state.step == 4:
        step_4_run_analysis()
    elif st.session_state.step == 5:
        step_5_view_results()


def step_0_workflow_overview():
    """Step 0: Workflow explanation and overview"""
    st.header("📚 Workflow Overview: Graph-Based Double Machine Learning")

    # Display logo at the top
    try:
        st.image("Logo.png", use_container_width=True)
    except Exception:
        pass  # If logo not found, continue without it

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
        1. **Build Causal DAG**
        2. **Upload Data**
        3. **Specify Interactions**
        4. **Run DML Analysis**
        5. **View Results**
        """)

    with col2:
        st.markdown("""
        #### What Happens:
        - Define treatment & outcome, add variables to DAG with types
        - Load your CSV dataset matching DAG variables
        - **Select variables for interaction term construction**
        - Estimate effects using DML with LASSO
        - Review main and interaction effects with significance
        """)

    st.markdown("---")

    # Workflow diagram (using image)
    st.subheader("🔄 Visual Workflow")

    try:
        st.image("Workflow_Diagram.png", use_container_width=True)
    except Exception:
        # Fallback to text-based representation if image not found
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


def step_1_build_dag():
    """Step 1: Build the causal DAG with variable definitions"""
    st.header("Step 1: Build Causal DAG & Define Variables")

    # Initialize DAG method selection in session state
    if 'dag_creation_method' not in st.session_state:
        st.session_state.dag_creation_method = None

    # Ask user to choose method if not yet selected
    if st.session_state.dag_creation_method is None:
        st.markdown("""
        Choose how you want to define your causal DAG:
        """)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📝 Paste DAG Syntax", use_container_width=True, type="secondary"):
                st.session_state.dag_creation_method = "paste"
                st.rerun()

        with col2:
            if st.button("🔨 Build DAG Step-by-Step", use_container_width=True, type="primary"):
                st.session_state.dag_creation_method = "build"
                st.rerun()

        st.markdown("---")
        st.info("""
        **Paste DAG Syntax**: If you already have your DAG structure, paste it using NetworkX `add_nodes` and `add_edges` format.

        **Build DAG Step-by-Step**: Interactive builder to define variables and draw causal relationships.
        """)

    elif st.session_state.dag_creation_method == "paste":
        step_1_paste_dag()
    elif st.session_state.dag_creation_method == "build":
        step_1_build_dag_interactive()


def step_1_paste_dag():
    """Step 1 (Paste method): Paste DAG using NetworkX syntax"""
    st.markdown("### Paste DAG Syntax")

    # Button to switch method
    if st.button("🔄 Switch to Interactive Builder", type="secondary"):
        st.session_state.dag_creation_method = "build"
        st.rerun()

    st.markdown("""
    Paste your DAG structure using NetworkX DiGraph format with `add_nodes` and `add_edges` commands.
    """)

    with st.expander("📖 Format Guide & Example"):
        st.markdown("""
        Use NetworkX DiGraph format:
        ```python
        G.add_nodes_from(['Education', 'Age', 'Gender', 'Salary'])
        G.add_edges_from([
            ('Education', 'Salary'),
            ('Age', 'Salary'),
            ('Gender', 'Salary'),
            ('Age', 'Education')
        ])
        ```

        **Rules:**
        - Use `add_nodes_from([...])` with a list of node names
        - Use `add_edges_from([...])` with a list of tuples (source, target)
        - Each edge is a tuple: `('Cause', 'Effect')`
        - Node names must be strings
        """)

    # Text area for DAG syntax
    dag_syntax = st.text_area(
        "Paste your DAG syntax:",
        height=200,
        placeholder="G.add_nodes_from(['X', 'Y', 'Z'])\nG.add_edges_from([('X', 'Y'), ('X', 'Z')])",
        key="dag_syntax_input"
    )

    if dag_syntax:
        try:
            # Parse the DAG syntax
            import re

            # Extract nodes
            nodes_match = re.search(r'add_nodes_from\s*\(\s*\[(.*?)\]', dag_syntax, re.DOTALL)
            # Extract edges
            edges_match = re.search(r'add_edges_from\s*\(\s*\[(.*?)\]', dag_syntax, re.DOTALL)

            if not nodes_match:
                st.error("❌ No `add_nodes_from` statement found. Please include node definitions.")
                return

            # Parse nodes
            nodes_str = nodes_match.group(1)
            nodes = re.findall(r"['\"]([^'\"]+)['\"]", nodes_str)

            if not nodes:
                st.error("❌ No nodes found. Make sure nodes are quoted strings.")
                return

            # Parse edges
            edges = []
            if edges_match:
                edges_str = edges_match.group(1)
                edge_tuples = re.findall(r"\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)", edges_str)
                edges = [(src, tgt) for src, tgt in edge_tuples]

                # Validate that all edge nodes are in the node list
                edge_nodes = set()
                for src, tgt in edges:
                    edge_nodes.add(src)
                    edge_nodes.add(tgt)

                invalid_nodes = edge_nodes - set(nodes)
                if invalid_nodes:
                    st.error(f"❌ Edge nodes not in node list: {invalid_nodes}")
                    return

            # Build DAG
            dag = nx.DiGraph()
            dag.add_nodes_from(nodes)
            if edges:
                dag.add_edges_from(edges)

            # Validate DAG
            validator = GraphValidator()
            is_valid, message = validator.validate_dag(dag)

            if not is_valid:
                st.error(f"❌ Invalid DAG: {message}")
                return

            st.success(f"✅ Parsed DAG with {dag.number_of_nodes()} nodes and {dag.number_of_edges()} edges")

            # Visualize DAG
            st.markdown("#### DAG Visualization")
            from components.dag_editor import DAGEditor
            dag_editor = DAGEditor(nodes)
            fig = dag_editor._visualize_dag(dag)
            st.pyplot(fig)

            st.markdown("---")

            # Now ask user to define data types, treatment, and outcome
            st.markdown("### Define Variable Configuration")

            # Initialize configuration in session state
            if 'pasted_dag_config' not in st.session_state:
                st.session_state.pasted_dag_config = {node: {'type': 'continuous'} for node in nodes}
            if 'pasted_dag' not in st.session_state:
                st.session_state.pasted_dag = dag
            if 'pasted_dag_nodes' not in st.session_state:
                st.session_state.pasted_dag_nodes = nodes

            # Treatment selection
            st.subheader("🎯 Select Treatment Variable")
            treatment = st.selectbox(
                "Treatment:",
                options=nodes,
                index=0 if 'pasted_treatment' not in st.session_state else nodes.index(st.session_state.get('pasted_treatment', nodes[0])),
                key="pasted_treatment_select"
            )

            # Outcome selection
            st.subheader("📊 Select Outcome Variable")
            outcome_options = [n for n in nodes if n != treatment]
            outcome = st.selectbox(
                "Outcome:",
                options=outcome_options,
                index=0 if 'pasted_outcome' not in st.session_state else (outcome_options.index(st.session_state.get('pasted_outcome', outcome_options[0])) if st.session_state.get('pasted_outcome') in outcome_options else 0),
                key="pasted_outcome_select"
            )

            st.markdown("---")

            # Data type configuration for all nodes
            st.subheader("📝 Define Data Types for All Variables")

            variables_config = {}

            for node in nodes:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{node}**")
                with col2:
                    var_type = st.selectbox(
                        f"Data type:",
                        options=['continuous', 'binary', 'categorical', 'ordinal'],
                        index=0,
                        key=f"pasted_var_type_{node}",
                        label_visibility="collapsed"
                    )
                    variables_config[node] = {'type': var_type}

            # Save configuration button
            if st.button("✅ Confirm DAG Configuration", type="primary", use_container_width=True):
                st.session_state.dag = dag
                st.session_state.treatment = treatment
                st.session_state.outcome = outcome
                st.session_state.dag_variables = variables_config
                st.session_state.column_types = {var: config['type'] for var, config in variables_config.items()}

                st.success(f"✅ DAG configured successfully!")
                st.success(f"✅ Treatment: **{treatment}** → Outcome: **{outcome}**")

                # Show summary
                with st.expander("📋 Variable Configuration Summary"):
                    var_df = pd.DataFrame([
                        {"Variable": var, "Type": config['type'],
                         "Role": "Treatment" if var == treatment else ("Outcome" if var == outcome else "Covariate")}
                        for var, config in variables_config.items()
                    ])
                    st.dataframe(var_df, use_container_width=True)

                # Navigation button
                col1, col2, col3 = st.columns([1, 1, 1])
                with col3:
                    if st.button("➡️ Proceed to Upload Data", type="primary", use_container_width=True):
                        st.session_state.step = 2
                        st.rerun()

        except Exception as e:
            st.error(f"❌ Error parsing DAG syntax: {str(e)}")
            st.exception(e)


def step_1_build_dag_interactive():
    """Step 1 (Build method): Build the causal DAG interactively with variable definitions"""
    st.markdown("### Build DAG Interactively")

    # Button to switch method
    if st.button("🔄 Switch to Paste DAG Syntax", type="secondary"):
        st.session_state.dag_creation_method = "paste"
        st.rerun()

    st.markdown("""
    Start by defining your causal model:
    1. **Specify treatment and outcome variables** (required to start)
    2. **Add additional variables** to your causal graph
    3. **Define data types** for each variable (continuous, binary, categorical, ordinal)
    4. **Draw causal relationships** (directed edges from causes to effects)
    5. **Drag nodes** to arrange your DAG layout interactively
    """)

    dag_editor = DAGEditor([], interactive=True)
    dag, treatment, outcome, variables_config = dag_editor.create_dag_with_variables()

    if dag is not None and treatment and outcome and variables_config:
        st.session_state.dag = dag
        st.session_state.treatment = treatment
        st.session_state.outcome = outcome
        st.session_state.dag_variables = variables_config
        st.session_state.column_types = {var: config['type'] for var, config in variables_config.items()}

        # Validate DAG
        validator = GraphValidator()
        is_valid, message = validator.validate_dag(dag)

        if is_valid:
            st.success(f"✅ Valid DAG with {dag.number_of_nodes()} nodes and {dag.number_of_edges()} edges")
            st.success(f"✅ Treatment: **{treatment}** → Outcome: **{outcome}**")

            # Show summary
            with st.expander("📋 Variable Configuration Summary"):
                var_df = pd.DataFrame([
                    {"Variable": var, "Type": config['type'],
                     "Role": "Treatment" if var == treatment else ("Outcome" if var == outcome else "Covariate")}
                    for var, config in variables_config.items()
                ])
                st.dataframe(var_df, use_container_width=True)

            # Navigation buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            with col3:
                if st.button("➡️ Proceed to Upload Data", type="primary", use_container_width=True):
                    st.session_state.step = 2
                    st.rerun()
        else:
            st.error(f"❌ Invalid DAG: {message}")

def step_2_upload_data():
    """Step 2: Upload CSV file"""
    st.header("Step 2: Upload Your Data")

    if not st.session_state.dag or not st.session_state.treatment or not st.session_state.outcome:
        st.warning("⚠️ Please build your DAG first (Step 1)")
        return

    st.markdown(f"""
    Upload a CSV file with columns matching your DAG variables:
    **Required columns:** {', '.join(st.session_state.dag_variables.keys())}

    **Requirements:**
    - CSV format with headers matching DAG variable names
    - No missing values in key variables
    - Treatment: **{st.session_state.treatment}**
    - Outcome: **{st.session_state.outcome}**
    """)

    data_loader = DataLoader()
    data = data_loader.load_data()

    if data is not None:
        # Validate that data has all required columns
        required_cols = set(st.session_state.dag_variables.keys())
        data_cols = set(data.columns)
        missing_cols = required_cols - data_cols
        extra_cols = data_cols - required_cols

        if missing_cols:
            st.error(f"❌ Missing columns in data: {', '.join(missing_cols)}")
            st.info("Please upload data with columns matching your DAG variables, or go back to Step 1 to modify your DAG.")
            return

        if extra_cols:
            st.warning(f"⚠️ Extra columns in data (will be ignored): {', '.join(extra_cols)}")
            # Keep only DAG columns
            data = data[list(required_cols)]

        st.session_state.data = data

        # Show data preview
        st.success(f"✅ Data loaded successfully! Shape: {data.shape}")

        with st.expander("📊 Data Preview", expanded=True):
            st.dataframe(data.head(20), use_container_width=True)

        with st.expander("📈 Basic Statistics"):
            st.dataframe(data.describe(), use_container_width=True)

        # Button to proceed
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Back to DAG", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        with col3:
            if st.button("➡️ Proceed to Specify Interactions", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()


def step_3_specify_interactions():
    """Step 3: Specify which variables to use for interaction terms"""
    st.header("Step 3: Specify Interactions of Interest")

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
            if st.button("⬅️ Back to Upload Data", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
        with col3:
            if len(selected_vars) >= 2 and st.button("➡️ Proceed to Run Analysis", type="primary", use_container_width=True):
                st.session_state.step = 4
                st.rerun()

    else:
        st.warning("⚠️ Please select at least 2 variables to create interaction terms.")

        # Navigation buttons (without proceed)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Back to Upload Data", use_container_width=True):
                st.session_state.step = 2
                st.rerun()


def step_4_run_analysis():
    """Step 4: Run DML analysis"""
    st.header("Step 4: Run DML Analysis")

    if st.session_state.treatment is None or st.session_state.outcome is None:
        st.warning("⚠️ Please build your DAG with treatment and outcome (Step 1)")
        return

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data (Step 2)")
        return

    st.markdown("""
    Ready to estimate the causal effect! The analysis will:
    1. Identify confounders from the DAG
    2. Use Double Machine Learning (DoubleML) with LASSO
    3. Estimate main effect (treatment → outcome)
    4. Estimate all interaction effects
    5. Provide confidence intervals and p-values with sensitivity analysis
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
                estimator = DMLEstimatorDoubleML(
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
                    st.session_state.step = 5
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)

    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back to Interactions", use_container_width=True):
            st.session_state.step = 3
            st.rerun()


def step_5_view_results():
    """Step 5: View results and visualizations"""
    st.header("Step 5: Results")

    if st.session_state.results is None:
        st.warning("⚠️ Please run the analysis first (Step 4)")
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
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Coefficient", f"{result['coefficient']:.4f}")
                    with col2:
                        st.metric("Std Error", f"{result['se']:.4f}")
                    with col3:
                        st.metric("95% CI", f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
                    with col4:
                        st.metric("RV %", f"{result.get('rv_percent', 0):.2f}%")

                    st.markdown(f"""
                    - **Variables:** {', '.join(result['variables'])}
                    - **Order:** {result['order']}-way interaction
                    - **P-value:** {result['p_value']:.4f}
                    - **Interpretation:** This interaction term has a statistically significant effect on the outcome.
                      The coefficient represents the additional effect when these variables interact.
                    - **Robustness (RV%):** Percentage of variation explained by unobserved confounders needed to nullify this effect.
                    """)

        if non_significant:
            with st.expander(f"📊 Non-Significant Interaction Terms ({len(non_significant)})"):
                # Create a dataframe for easy viewing
                import pandas as pd
                df = pd.DataFrame([{
                    'Term': r['term'],
                    'Variables': ' × '.join(r['variables']),
                    'Order': r['order'],
                    'Coefficient': f"{r['coefficient']:.4f}",
                    'Std Error': f"{r['se']:.4f}",
                    'P-value': f"{r['p_value']:.4f}",
                    'CI Lower': f"{r['ci_lower']:.3f}",
                    'CI Upper': f"{r['ci_upper']:.3f}"
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
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("🔄 Start New Analysis", use_container_width=True):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
