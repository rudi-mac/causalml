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
if 'two_way_interaction_variables' not in st.session_state:
    st.session_state.two_way_interaction_variables = []
if 'three_way_interaction_variables' not in st.session_state:
    st.session_state.three_way_interaction_variables = []
if 'results' not in st.session_state:
    st.session_state.results = None
if 'categorical_drop_values' not in st.session_state:
    st.session_state.categorical_drop_values = {}  # Dict of categorical_var: value_to_drop
if 'categorical_drop_confirmed' not in st.session_state:
    st.session_state.categorical_drop_confirmed = False
if 'mediators' not in st.session_state:
    st.session_state.mediators = set()  # Set of mediator variables (on causal path, excluded from interactions)

def main():
    """Main application flow"""

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


def get_workflow_diagram_html():
    """Generate interactive HTML workflow diagram with hover tooltips"""
    html = """
    <style>
        .workflow-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            position: relative;
        }

        .workflow-box {
            background: white;
            border: 3px solid #333;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            position: relative;
            transition: all 0.3s ease;
            z-index: 1;
        }

        .workflow-box:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            cursor: pointer;
            z-index: 100;
        }

        .workflow-main-section {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border: 3px solid #333;
            border-radius: 20px;
            padding: 30px 20px;
            margin: 15px 0;
            position: relative;
        }

        .workflow-step {
            background: linear-gradient(135deg, #4a5f3a 0%, #5d7647 100%);
            color: white;
            border-radius: 12px;
            padding: 18px;
            margin: 12px 0;
            font-size: 17px;
            font-weight: bold;
            position: relative;
            transition: all 0.3s ease;
            cursor: pointer;
            z-index: 1;
        }

        .workflow-step:hover {
            transform: translateX(5px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            background: linear-gradient(135deg, #5d7647 0%, #6d8657 100%);
            z-index: 100;
        }

        .arrow-container {
            text-align: center;
            font-size: 24px;
            color: #666;
            margin: 5px 0;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }

        .arrow-down {
            font-size: 30px;
        }

        .arrow-up {
            font-size: 30px;
        }

        .tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.95);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: normal;
            width: 400px;
            z-index: 10000;
            pointer-events: none;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease, visibility 0.3s ease;
            line-height: 1.6;
            text-align: left;
            box-shadow: 0 6px 20px rgba(0,0,0,0.5);
            border: 2px solid rgba(255,255,255,0.2);
            left: 110%;
            top: 0;
            white-space: normal;
        }

        .workflow-step:hover .tooltip,
        .workflow-box:hover .tooltip {
            opacity: 1;
            visibility: visible;
        }

        /* Position tooltip to the left if it would overflow */
        @media (max-width: 1200px) {
            .tooltip {
                left: auto;
                right: 110%;
            }
        }

        @media (max-width: 768px) {
            .workflow-container {
                padding: 10px;
            }
            .workflow-box {
                font-size: 16px;
                padding: 15px;
            }
            .workflow-step {
                font-size: 15px;
                padding: 15px;
            }
            .tooltip {
                width: 90vw;
                font-size: 12px;
                padding: 12px 15px;
                left: 50%;
                right: auto;
                transform: translateX(-50%);
                top: 100%;
                margin-top: 10px;
            }
            .arrow-container {
                font-size: 20px;
            }
        }
    </style>

    <div class="workflow-container">
        <!-- Top Box -->
        <div class="workflow-box">
            Identify Phenomenon & Target Theory
            <div class="tooltip">
                Before starting with Step 1, it is crucial to narrow the scope of inquiry to a specific organizational phenomenon and establish a related theoretical framework to be elaborated.
            </div>
        </div>

        <div class="arrow-container">
            <span class="arrow-down">↓</span>
        </div>

        <!-- Main Section -->
        <div class="workflow-main-section">
            <div class="workflow-step">
                1. Encode Existing Knowledge into DAG
                <div class="tooltip">
                    In Step 1, researchers encode the existing state of the literature with respect to causal relationships into a DAG.
                </div>
            </div>

            <div class="arrow-container">
                <span class="arrow-down">↓</span>
                <span class="arrow-up">↑</span>
            </div>

            <div class="workflow-step">
                2. Collect and Pre-Process Data
                <div class="tooltip">
                    In Step 2, relevant data is collected and pre-processed, including, for example, the construction of dummy variables for categorical measures. Where feasible, researchers should document measurement choices and missing-data handling to preserve transparency and replicability.
                </div>
            </div>

            <div class="arrow-container">
                <span class="arrow-down">↓</span>
                <span class="arrow-up">↑</span>
            </div>

            <div class="workflow-step">
                3. Specify Learners (e.g., LASSO)
                <div class="tooltip">
                    Step 3 involves selecting appropriate ML estimators—such as LASSO regression—to separate the effect of the variable of interest from those of the control variables.
                </div>
            </div>

            <div class="arrow-container">
                <span class="arrow-down">↓</span>
                <span class="arrow-up">↑</span>
            </div>

            <div class="workflow-step">
                4. Specify Interactions of Interest
                <div class="tooltip">
                    In Step 4, interaction terms are specified based on their relevance in the context of the investigated phenomenon and target theory.
                </div>
            </div>

            <div class="arrow-container">
                <span class="arrow-down">↓</span>
                <span class="arrow-up">↑</span>
            </div>

            <div class="workflow-step">
                5. Fit Treatment & Outcome Models
                <div class="tooltip">
                    In Step 5, the treatment and outcome models are trained, which involves parameter tuning and cross validation to ensure robust and accurate estimation of treatment effects.
                </div>
            </div>

            <div class="arrow-container">
                <span class="arrow-down">↓</span>
                <span class="arrow-up">↑</span>
            </div>

            <div class="workflow-step">
                6. Perform Sensitivity Analysis
                <div class="tooltip">
                    Step 6 uses sensitivity analysis to evaluate the robustness of the causal estimates by examining the impact of unobserved confounders, ensuring confidence in the derived conclusions.
                </div>
            </div>

            <div class="arrow-container">
                <span class="arrow-down">↓</span>
                <span class="arrow-up">↑</span>
            </div>

            <div class="workflow-step">
                7. Select Robust Interactions
                <div class="tooltip">
                    Step 7 involves selecting the interaction terms that robustly appear significant. Taken together, these steps yield a disciplined pipeline that expands discovery while maintaining identification discipline and inferential validity.
                </div>
            </div>
        </div>

        <div class="arrow-container">
            <span class="arrow-down">↓</span>
        </div>

        <!-- Bottom Box -->
        <div class="workflow-box">
            Formulate Implications for Theory and Practice
            <div class="tooltip">
                After completing all seven steps, researchers can formulate implications for both theory and practice based on the robust findings from the graph-based DML analysis.
            </div>
        </div>
    </div>
    """
    return html

def step_0_workflow_overview():
    """Step 0: Workflow explanation and overview"""

    # Display logo at the very top with fixed width (1/3 of typical screen width ~400px)
    try:
        st.image("Logo.png", width=400)
    except Exception:
        pass  # If logo not found, continue without it

    # Description below logo
    st.markdown("""
    This tool enables you to discover **significant interaction effects** using **Graph-Based Double Machine Learning (DML)**.
    Define causal structures, select interactions of interest, and robustly estimate heterogeneous treatment effects.
    """)

    st.markdown("---")

    st.markdown("""
    ### What is Graph-Based Double Machine Learning?

    **Graph-Based Double Machine Learning** is a powerful methodology that combines:
    - **Directed Acyclic Graphs (DAGs)** to represent causal assumptions
    - **Double Machine Learning (DML)** to handle high-dimensional settings
    - **Interaction term exploration** to discover heterogeneous effects

    This approach enables you to **robustly explore and estimate interaction effects** without
    the limitations of traditional regression that requires pre-selecting a small subset of interactions.
    """)

    st.markdown("---")

    # Interactive workflow diagram
    st.subheader("🔄 Visual Workflow")
    st.markdown("*Hover over each box to see detailed explanations*")

    # Display interactive HTML workflow diagram
    import streamlit.components.v1 as components
    components.html(get_workflow_diagram_html(), height=1200, scrolling=False)

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

            st.markdown("---")

            # Now ask user to define data types, treatment, and outcome BEFORE visualization
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

            st.markdown("---")

            # Visualize DAG with treatment and outcome highlighted
            st.markdown("#### DAG Visualization (Interactive - Drag nodes to reposition)")
            from components.dag_editor import DAGEditor
            dag_editor = DAGEditor(nodes)
            dag_editor._visualize_dag(dag, treatment=treatment, outcome=outcome)

            st.markdown("---")

            # Save configuration button
            if st.button("✅ Confirm DAG Configuration", type="primary", use_container_width=True):
                st.session_state.dag = dag
                st.session_state.treatment = treatment
                st.session_state.outcome = outcome
                st.session_state.dag_variables = variables_config
                st.session_state.column_types = {var: config['type'] for var, config in variables_config.items()}
                st.session_state.dag_confirmed = True
                st.rerun()

            # Show summary and navigation after confirmation
            if st.session_state.get('dag_confirmed', False) and st.session_state.dag is not None:
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
        # Validate that data has all required columns with strict (case-sensitive) matching
        required_cols = set(st.session_state.dag_variables.keys())
        data_cols = set(data.columns)

        # Strict matching: exact name including case
        matched_cols = required_cols & data_cols
        missing_cols = required_cols - data_cols
        extra_cols = data_cols - required_cols

        # Display column matching overview
        st.subheader("📋 Column Matching Overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Matched Columns", len(matched_cols))
        with col2:
            st.metric("❌ Missing DAG Nodes", len(missing_cols))
        with col3:
            st.metric("⚠️ Extra Columns", len(extra_cols))

        # Show detailed matching results
        with st.expander("📊 Detailed Matching Results", expanded=True):
            if matched_cols:
                st.markdown("**✅ Matched columns (CSV ↔ DAG nodes):**")
                for col in sorted(matched_cols):
                    st.markdown(f"- `{col}`")

            if missing_cols:
                st.markdown("**❌ Missing DAG nodes (not found in CSV):**")
                for col in sorted(missing_cols):
                    st.markdown(f"- `{col}`")

            if extra_cols:
                st.markdown("**⚠️ Extra columns (in CSV but not in DAG):**")
                for col in sorted(extra_cols):
                    st.markdown(f"- `{col}`")

        if missing_cols:
            st.error(f"❌ Cannot proceed: {len(missing_cols)} DAG node(s) missing from uploaded CSV.")
            st.info("💡 Please upload a CSV with columns that exactly match your DAG variable names (case-sensitive), or go back to Step 1 to modify your DAG.")
            return

        # Keep only DAG columns
        data = data[list(matched_cols)]
        st.session_state.data = data

        # Show data preview
        st.success(f"✅ Data processed successfully! Shape: {data.shape} | Columns used: {len(matched_cols)}")

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

    if st.session_state.treatment is None:
        st.warning("⚠️ Please specify a treatment variable first (Step 1)")
        return

    treatment = st.session_state.treatment

    st.markdown(f"""
    Select variables to construct **interaction terms** with the **treatment variable: `{treatment}`**.

    ⚠️ **Important:** The treatment variable is **always included** in all interaction terms.

    - **Two-way interactions**: {treatment} × *selected variable* (e.g., {treatment} × Age, {treatment} × Gender)
    - **Three-way interactions**: {treatment} × *two-way variable* × *three-way variable* (e.g., {treatment} × Age × Gender)

    ### How It Works
    1. Select variables for **two-way interactions** (each creates: {treatment} × variable)
    2. Select variables for **three-way interactions** (each creates: {treatment} × two_way_var × three_way_var)

    ### Categorical Variables
    If a variable is categorical with multiple categories (e.g., Race: A, B, C), one interaction term
    will be created per category (e.g., {treatment} × Race_A, {treatment} × Race_B, {treatment} × Race_C).

    **Tip:** Include variables that are root nodes in your DAG (confounders, not mediators).
    """)

    # Get all variables from DAG (exclude treatment and outcome)
    all_variables = [v for v in list(st.session_state.dag.nodes())
                     if v != st.session_state.treatment and v != st.session_state.outcome]

    if not all_variables:
        st.error("❌ No variables available for interactions (only treatment and outcome in DAG)")
        return

    # Identify mediators: nodes that have incoming edges AND outgoing edges to outcome
    # These should NOT be included in interactions as they are on the causal path
    mediators = set()
    for node in all_variables:
        # Check if node has incoming edges
        has_incoming = len(list(st.session_state.dag.predecessors(node))) > 0
        # Check if node has outgoing edge to outcome
        has_outgoing_to_outcome = st.session_state.outcome in st.session_state.dag.successors(node)

        if has_incoming and has_outgoing_to_outcome:
            mediators.add(node)

    # Store mediators in session state for use in DML estimator
    st.session_state.mediators = mediators

    # Remove mediators from available variables
    all_variables = [v for v in all_variables if v not in mediators]

    if not all_variables:
        st.error("❌ No variables available for interactions (all non-treatment/outcome variables are mediators)")
        return

    # Identify root nodes (variables with no predecessors) - these are good candidates
    root_nodes = [node for node in all_variables if st.session_state.dag.in_degree(node) == 0]

    if mediators:
        st.warning(f"⚠️ **Mediators excluded from interaction selection** (on causal path): {', '.join(mediators)}")
    st.info(f"💡 **Root nodes in your DAG** (good candidates for interactions): {', '.join(root_nodes) if root_nodes else 'None'}")

    st.subheader("Select Interaction Variables")

    # Two-way interaction variables
    st.markdown("#### 1️⃣ Two-Way Interactions")
    st.markdown(f"Each selected variable will create one interaction: **{treatment} × variable**")

    two_way_vars = st.multiselect(
        "Select variables for two-way interactions:",
        options=all_variables,
        default=st.session_state.two_way_interaction_variables if st.session_state.two_way_interaction_variables else None,
        help=f"Each variable creates a two-way interaction with {treatment}",
        key="two_way_select"
    )

    if two_way_vars:
        st.session_state.two_way_interaction_variables = two_way_vars

    # Three-way interaction variables
    st.markdown("#### 2️⃣ Three-Way Interactions")
    st.markdown(f"""
    Each selected variable will be combined with **each two-way interaction variable** to create:
    **{treatment} × two_way_var × three_way_var**

    Example: If you select Age for two-way and Gender for three-way, this creates:
    - {treatment} × Age × Gender
    """)

    three_way_vars = st.multiselect(
        "Select variables for three-way interactions:",
        options=all_variables,
        default=st.session_state.three_way_interaction_variables if st.session_state.three_way_interaction_variables else None,
        help=f"Each variable will be combined with two-way variables to create three-way interactions with {treatment}",
        key="three_way_select"
    )

    if three_way_vars:
        st.session_state.three_way_interaction_variables = three_way_vars

    # Calculate interaction counts
    n_two_way = len(two_way_vars)
    n_three_way = len(two_way_vars) * len(three_way_vars) if (two_way_vars and three_way_vars) else 0

    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Two-Way Variables", len(two_way_vars))
    with col2:
        st.metric("Three-Way Variables", len(three_way_vars))
    with col3:
        st.metric("Total Interactions", n_two_way + n_three_way)

    # Show preview of interactions
    if two_way_vars or three_way_vars:
        with st.expander("📋 Preview of Interaction Terms", expanded=True):
            # Helper function to get categorical values from data
            def get_categorical_info(var):
                var_type = st.session_state.dag_variables.get(var, {}).get('type', 'unknown')
                if var_type in ['categorical', 'ordinal'] and st.session_state.data is not None and var in st.session_state.data.columns:
                    unique_values = st.session_state.data[var].unique()
                    n_values = len(unique_values)
                    if n_values <= 5:  # Show values if 5 or fewer
                        return var_type, f"Values: {', '.join(map(str, unique_values))}"
                    else:
                        return var_type, f"{n_values} categories (one dummy per category)"
                elif var_type in ['categorical', 'ordinal']:
                    return var_type, "Multiple categories"
                return var_type, None

            if two_way_vars:
                st.markdown("**Two-way interactions:**")
                st.markdown(f"*Each interaction combines {treatment} with one of the selected variables*")
                for var in two_way_vars:
                    var_type, cat_info = get_categorical_info(var)
                    treatment_type, treatment_info = get_categorical_info(treatment)

                    interaction_str = f"- {treatment} × {var}"
                    if var_type in ['categorical', 'ordinal'] or treatment_type in ['categorical', 'ordinal']:
                        interaction_str += " → "
                        if treatment_type in ['categorical', 'ordinal'] and var_type in ['categorical', 'ordinal']:
                            interaction_str += "Multiple terms (one per combination of categories)"
                        elif treatment_type in ['categorical', 'ordinal']:
                            interaction_str += f"Multiple terms (one per {treatment} category)"
                        else:
                            interaction_str += f"Multiple terms (one per {var} category)"
                        if cat_info:
                            interaction_str += f"\n  - {var}: {cat_info}"
                        if treatment_info and treatment_type in ['categorical', 'ordinal']:
                            interaction_str += f"\n  - {treatment}: {treatment_info}"
                    st.markdown(interaction_str)

            if two_way_vars and three_way_vars:
                st.markdown("**Three-way interactions:**")
                st.markdown(f"*Each interaction combines {treatment} with two selected variables*")
                for two_way_var in two_way_vars:
                    for three_way_var in three_way_vars:
                        two_way_type, two_way_info = get_categorical_info(two_way_var)
                        three_way_type, three_way_info = get_categorical_info(three_way_var)
                        treatment_type, treatment_info = get_categorical_info(treatment)

                        interaction_str = f"- {treatment} × {two_way_var} × {three_way_var}"
                        has_categorical = any(t in ['categorical', 'ordinal'] for t in [treatment_type, two_way_type, three_way_type])

                        if has_categorical:
                            interaction_str += " → Multiple terms (one per category combination)"
                        st.markdown(interaction_str)
            elif not two_way_vars:
                st.markdown("**Three-way interactions:** *Select two-way variables first*")

        # Warning for too many interactions
        total_interactions = n_two_way + n_three_way
        if total_interactions > 100:
            st.warning(f"⚠️ You've selected {total_interactions} base interaction terms. This may take a long time to compute. Consider selecting fewer variables.")
        elif total_interactions > 50:
            st.info(f"ℹ️ You've selected {total_interactions} base interaction terms. Analysis may take several minutes.")

    # Determine if we can proceed
    can_proceed = len(two_way_vars) > 0

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back to Upload Data", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col3:
        if can_proceed:
            if st.button("➡️ Proceed to Run Analysis", type="primary", use_container_width=True):
                st.session_state.step = 4
                st.rerun()
        else:
            st.button("➡️ Proceed to Run Analysis", type="primary", use_container_width=True, disabled=True)
            st.warning("⚠️ Please select at least one variable for two-way interactions to proceed.")


def step_4_run_analysis():
    """Step 4: Run DML analysis"""
    st.header("Step 4: Run DML Analysis")

    if st.session_state.treatment is None or st.session_state.outcome is None:
        st.warning("⚠️ Please build your DAG with treatment and outcome (Step 1)")
        return

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data (Step 2)")
        return

    # Identify categorical variables (including ordinal)
    categorical_vars = {var: dtype for var, dtype in st.session_state.column_types.items()
                       if dtype in ['categorical', 'ordinal'] and var in st.session_state.data.columns}

    # Step 4a: Ask user which value to drop for each categorical variable
    if categorical_vars and not st.session_state.categorical_drop_confirmed:
        st.markdown("### 📋 Step 4a: Select Reference Categories")
        st.markdown("""
        Before constructing interaction terms, you need to specify which category value to drop
        (reference category) for each categorical variable to avoid the **dummy variable trap**.

        The reference category will be the baseline against which other categories are compared.
        """)

        st.info("""
        **Important:** Dropping one category per categorical variable is necessary to:
        - Avoid perfect multicollinearity (dummy variable trap)
        - Properly identify interaction effects between different categorical variables
        - Ensure the model is mathematically identifiable
        """)

        with st.form("categorical_drop_form"):
            st.markdown("#### Select which category value to drop for each categorical variable:")

            # Create selection for each categorical variable
            drop_selections = {}
            for var_name, var_type in categorical_vars.items():
                st.markdown(f"**{var_name}** ({var_type})")

                # Get unique values from the data
                unique_values = sorted(st.session_state.data[var_name].dropna().unique())

                # Use existing selection if available, otherwise default to first value
                default_idx = 0
                if var_name in st.session_state.categorical_drop_values:
                    try:
                        default_idx = unique_values.index(st.session_state.categorical_drop_values[var_name])
                    except ValueError:
                        default_idx = 0

                selected_value = st.selectbox(
                    f"Drop this value from {var_name}:",
                    options=unique_values,
                    index=default_idx,
                    key=f"drop_{var_name}",
                    help=f"Select the reference category for {var_name}. This category will be omitted from the model."
                )

                drop_selections[var_name] = selected_value

                # Show info about how many categories will remain
                st.caption(f"  → Will create {len(unique_values) - 1} dummy variables (dropping '{selected_value}')")

            st.markdown("---")
            submitted = st.form_submit_button("✅ Confirm Reference Categories", type="primary", use_container_width=True)

            if submitted:
                # Store selections in session state
                st.session_state.categorical_drop_values = drop_selections
                st.session_state.categorical_drop_confirmed = True
                st.success("✅ Reference categories confirmed! Proceeding with interaction construction...")
                st.rerun()

        # Show back button
        if st.button("⬅️ Back to Interactions", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

        return  # Don't show the rest of Step 4 until user confirms

    # If no categorical variables, automatically confirm
    if not categorical_vars:
        st.session_state.categorical_drop_confirmed = True

    # Show a summary of the selected drop values if they exist
    if categorical_vars and st.session_state.categorical_drop_values:
        with st.expander("📋 Selected Reference Categories", expanded=False):
            st.markdown("The following category values will be dropped (used as reference):")
            for var, val in st.session_state.categorical_drop_values.items():
                st.markdown(f"- **{var}**: dropping `{val}`")

            # Add button to change selections
            if st.button("🔄 Change Reference Categories"):
                st.session_state.categorical_drop_confirmed = False
                st.rerun()

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

    # Preview treatment variables that will be estimated
    st.markdown("---")

    try:
        # Create estimator preview (without running the full analysis)
        preview_estimator = DMLEstimatorDoubleML(
            data=st.session_state.data,
            dag=st.session_state.dag,
            treatment=st.session_state.treatment,
            outcome=st.session_state.outcome,
            column_types=st.session_state.column_types,
            two_way_interaction_variables=st.session_state.two_way_interaction_variables,
            three_way_interaction_variables=st.session_state.three_way_interaction_variables,
            categorical_drop_values=st.session_state.categorical_drop_values,
            mediators=st.session_state.mediators
        )

        # Preprocess data to get the final dataframe shape
        preview_data = preview_estimator._preprocess_data()
        preview_data = preview_estimator._construct_interaction_terms(preview_data)

        # Display dataframe shape (as requested by user)
        st.markdown("### 📊 DoubleML Dataframe Shape")
        st.markdown("""
        This shows the shape (rows, columns) of the dataframe that will be used in the DoubleML data object,
        including all 2-way and 3-way interactions.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Rows", f"{preview_data.shape[0]:,}")
        with col2:
            st.metric("Number of Columns", f"{preview_data.shape[1]:,}")

        st.info(f"""
        **Dataframe shape:** {preview_data.shape[0]:,} rows × {preview_data.shape[1]:,} columns

        This dataframe includes:
        - Original variables (after preprocessing and dummification)
        - All possible 2-way interactions (filtered by category)
        - All possible 3-way interactions (filtered by category)
        """)

        st.markdown("---")
        st.markdown("### 📋 Treatment Variables to be Estimated")
        st.markdown("""
        The following treatment effects will be estimated by the DML model.
        This list **exactly matches** what will be passed to the DoubleML object via the `d_cols` parameter.
        """)

        preview = preview_estimator.preview_treatment_variables()

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Variables", len(preview['treatment_variables']))
        with col2:
            st.metric("Main Treatment", len(preview['main_treatment_vars']))
        with col3:
            st.metric("2-way Interactions", len(preview['two_way_interaction_vars']))
        with col4:
            st.metric("3-way Interactions", len(preview['three_way_interaction_vars']))

        # Display organized list
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Main Treatment Effects")
            if preview['main_treatment_vars']:
                for var in preview['main_treatment_vars']:
                    st.markdown(f"- `{var}`")
            else:
                st.markdown("*None*")

        with col2:
            st.markdown("#### 2-way Interactions")
            if preview['two_way_interaction_vars']:
                for var in preview['two_way_interaction_vars']:
                    st.markdown(f"- `{var}`")
            else:
                st.markdown("*None*")

        with col3:
            st.markdown("#### 3-way Interactions")
            if preview['three_way_interaction_vars']:
                # Show first 10, then expandable for rest
                preview_vars = preview['three_way_interaction_vars'][:10]
                for var in preview_vars:
                    st.markdown(f"- `{var}`")
                if len(preview['three_way_interaction_vars']) > 10:
                    with st.expander(f"Show {len(preview['three_way_interaction_vars']) - 10} more..."):
                        for var in preview['three_way_interaction_vars'][10:]:
                            st.markdown(f"- `{var}`")
            else:
                st.markdown("*None*")

        st.info(f"""
        **Note:** This list shows all {len(preview['treatment_variables'])} treatment variables that will be estimated.
        These are the exact variables passed to DoubleML via `d_cols`.
        """)

    except Exception as e:
        st.warning(f"⚠️ Could not preview treatment variables: {str(e)}")
        st.markdown("The preview will be generated when you run the analysis.")

    st.markdown("---")

    # Run analysis button
    if st.button("🚀 Run DML Analysis", type="primary", use_container_width=True):
        try:
            # Create progress bar and status message placeholders
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Define progress callback
            def update_progress(current, total, message):
                progress_value = int((current / total) * 100)
                progress_bar.progress(progress_value)
                status_text.text(f"⏳ {message}")

            # Run analysis with progress tracking
            estimator = DMLEstimatorDoubleML(
                data=st.session_state.data,
                dag=st.session_state.dag,
                treatment=st.session_state.treatment,
                outcome=st.session_state.outcome,
                column_types=st.session_state.column_types,
                two_way_interaction_variables=st.session_state.two_way_interaction_variables,
                three_way_interaction_variables=st.session_state.three_way_interaction_variables,
                categorical_drop_values=st.session_state.categorical_drop_values,
                mediators=st.session_state.mediators
            )

            results = estimator.estimate_ate(
                discrete_treatment=discrete_treatment,
                n_splits=n_splits,
                progress_callback=update_progress
            )

            st.session_state.results = results

            # Clear progress indicators and show success
            progress_bar.empty()
            status_text.empty()
            st.success("✅ Analysis complete!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.exception(e)

    # Display results directly if available (following Sample_Analysis.ipynb)
    if st.session_state.results is not None:
        results = st.session_state.results

        st.markdown("---")
        st.markdown("### 📊 DML Estimation Results")
        st.markdown("""
        Comprehensive results table showing all treatment effects.
        Click column headers to sort. Use filters below to refine the view.
        """)

        # Display the comprehensive results dataframe
        if results.get('detailed_results_df') is not None:
            detailed_df = results['detailed_results_df']

            # Display metrics for summary
            total_treatments = len(detailed_df)
            sig_1pct = detailed_df['sig_1pct'].sum()
            sig_5pct = detailed_df['sig_5pct'].sum()
            sig_10pct = detailed_df['sig_10pct'].sum()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Treatments", total_treatments)
            with col2:
                st.metric("Sig. at 1%", sig_1pct)
            with col3:
                st.metric("Sig. at 5%", sig_5pct)
            with col4:
                st.metric("Sig. at 10%", sig_10pct)

            # Add filtering options
            st.markdown("#### Filters")
            filter_col1, filter_col2, filter_col3 = st.columns(3)

            with filter_col1:
                significance_filter = st.selectbox(
                    "Significance Level",
                    options=["All", "p < 0.01", "p < 0.05", "p < 0.10", "Not Significant"],
                    index=0,
                    key="step4_sig_filter"
                )

            with filter_col2:
                sort_column = st.selectbox(
                    "Sort By",
                    options=["p_value", "coefficient", "std_error", "t_statistic", "treatment"],
                    index=0,
                    key="step4_sort_col"
                )

            with filter_col3:
                sort_order = st.radio(
                    "Sort Order",
                    options=["Ascending", "Descending"],
                    index=0 if sort_column == "p_value" else 1,
                    horizontal=True,
                    key="step4_sort_order"
                )

            # Apply filters
            filtered_df = detailed_df.copy()

            if significance_filter == "p < 0.01":
                filtered_df = filtered_df[filtered_df['sig_1pct']]
            elif significance_filter == "p < 0.05":
                filtered_df = filtered_df[filtered_df['sig_5pct']]
            elif significance_filter == "p < 0.10":
                filtered_df = filtered_df[filtered_df['sig_10pct']]
            elif significance_filter == "Not Significant":
                filtered_df = filtered_df[~filtered_df['sig_10pct']]

            # Apply sorting
            ascending = (sort_order == "Ascending")
            filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending)

            # Reset index for display
            filtered_df = filtered_df.reset_index(drop=True)

            # Display the detailed dataframe
            st.markdown(f"#### Results Table ({len(filtered_df)} of {len(detailed_df)} rows)")

            # Create a formatted display dataframe
            display_df = filtered_df.copy()

            # Add significance stars column first
            def add_stars(row):
                if row['sig_1pct']:
                    return '***'
                elif row['sig_5pct']:
                    return '**'
                elif row['sig_10pct']:
                    return '*'
                else:
                    return ''

            display_df.insert(0, 'sig', filtered_df.apply(add_stars, axis=1))

            # Reorder columns for better readability
            column_order = ['sig', 'treatment', 'coefficient', 'std_error', 't_statistic', 'p_value',
                           'ci_lower_95', 'ci_upper_95', 'sig_1pct', 'sig_5pct', 'sig_10pct']
            display_df = display_df[column_order]

            # Rename columns for better display
            display_df = display_df.rename(columns={
                'sig': 'Sig.',
                'treatment': 'Treatment',
                'coefficient': 'Coefficient',
                'std_error': 'Std Error',
                't_statistic': 't-Statistic',
                'p_value': 'P-Value',
                'ci_lower_95': 'CI Lower (95%)',
                'ci_upper_95': 'CI Upper (95%)',
                'sig_1pct': 'p<0.01',
                'sig_5pct': 'p<0.05',
                'sig_10pct': 'p<0.10'
            })

            # Display with interactive dataframe
            st.dataframe(
                display_df,
                use_container_width=True,
                height=500,
                column_config={
                    "Sig.": st.column_config.TextColumn(width="small"),
                    "Treatment": st.column_config.TextColumn(width="large"),
                    "Coefficient": st.column_config.NumberColumn(format="%.6f"),
                    "Std Error": st.column_config.NumberColumn(format="%.6f"),
                    "t-Statistic": st.column_config.NumberColumn(format="%.4f"),
                    "P-Value": st.column_config.NumberColumn(format="%.6f"),
                    "CI Lower (95%)": st.column_config.NumberColumn(format="%.6f"),
                    "CI Upper (95%)": st.column_config.NumberColumn(format="%.6f"),
                    "p<0.01": st.column_config.CheckboxColumn(),
                    "p<0.05": st.column_config.CheckboxColumn(),
                    "p<0.10": st.column_config.CheckboxColumn()
                }
            )

            st.caption("Significance levels: *** p<0.01, ** p<0.05, * p<0.10")

            # Add download button for filtered results
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Results (CSV)",
                data=csv_data,
                file_name="dml_results_filtered.csv",
                mime="text/csv",
                key="step4_download"
            )

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

    # Display comprehensive results dataframe
    if results.get('detailed_results_df') is not None:
        st.markdown("---")
        st.markdown("### 📊 Comprehensive Results Table")
        st.markdown("""
        All treatment effects including main effect and interaction terms.
        Click column headers to sort. Use filters below to refine the view.
        """)

        detailed_df = results['detailed_results_df']

        # Display metrics for summary
        total_treatments = len(detailed_df)
        sig_1pct = detailed_df['sig_1pct'].sum()
        sig_5pct = detailed_df['sig_5pct'].sum()
        sig_10pct = detailed_df['sig_10pct'].sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Treatments", total_treatments)
        with col2:
            st.metric("Sig. at 1%", sig_1pct)
        with col3:
            st.metric("Sig. at 5%", sig_5pct)
        with col4:
            st.metric("Sig. at 10%", sig_10pct)

        # Add filtering options
        st.markdown("#### Filters")
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            significance_filter = st.selectbox(
                "Significance Level",
                options=["All", "p < 0.01", "p < 0.05", "p < 0.10", "Not Significant"],
                index=0
            )

        with filter_col2:
            sort_column = st.selectbox(
                "Sort By",
                options=["p_value", "coefficient", "std_error", "t_statistic", "treatment"],
                index=0
            )

        with filter_col3:
            sort_order = st.radio(
                "Sort Order",
                options=["Ascending", "Descending"],
                index=0 if sort_column == "p_value" else 1,
                horizontal=True
            )

        # Apply filters
        filtered_df = detailed_df.copy()

        if significance_filter == "p < 0.01":
            filtered_df = filtered_df[filtered_df['sig_1pct']]
        elif significance_filter == "p < 0.05":
            filtered_df = filtered_df[filtered_df['sig_5pct']]
        elif significance_filter == "p < 0.10":
            filtered_df = filtered_df[filtered_df['sig_10pct']]
        elif significance_filter == "Not Significant":
            filtered_df = filtered_df[~filtered_df['sig_10pct']]

        # Apply sorting
        ascending = (sort_order == "Ascending")
        filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending)

        # Reset index for display
        filtered_df = filtered_df.reset_index(drop=True)

        # Display the detailed dataframe
        st.markdown(f"#### Results Table ({len(filtered_df)} of {len(detailed_df)} rows)")

        # Create a formatted display dataframe
        display_df = filtered_df.copy()

        # Add significance stars column first
        def add_stars(row):
            if row['sig_1pct']:
                return '***'
            elif row['sig_5pct']:
                return '**'
            elif row['sig_10pct']:
                return '*'
            else:
                return ''

        display_df.insert(0, 'sig', filtered_df.apply(add_stars, axis=1))

        # Reorder columns for better readability
        column_order = ['sig', 'treatment', 'coefficient', 'std_error', 't_statistic', 'p_value',
                       'ci_lower_95', 'ci_upper_95', 'sig_1pct', 'sig_5pct', 'sig_10pct']
        display_df = display_df[column_order]

        # Rename columns for better display
        display_df = display_df.rename(columns={
            'sig': 'Sig.',
            'treatment': 'Treatment',
            'coefficient': 'Coefficient',
            'std_error': 'Std Error',
            't_statistic': 't-Statistic',
            'p_value': 'P-Value',
            'ci_lower_95': 'CI Lower (95%)',
            'ci_upper_95': 'CI Upper (95%)',
            'sig_1pct': 'p<0.01',
            'sig_5pct': 'p<0.05',
            'sig_10pct': 'p<0.10'
        })

        # Display with interactive dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            height=500,
            column_config={
                "Sig.": st.column_config.TextColumn(width="small"),
                "Treatment": st.column_config.TextColumn(width="large"),
                "Coefficient": st.column_config.NumberColumn(format="%.6f"),
                "Std Error": st.column_config.NumberColumn(format="%.6f"),
                "t-Statistic": st.column_config.NumberColumn(format="%.4f"),
                "P-Value": st.column_config.NumberColumn(format="%.6f"),
                "CI Lower (95%)": st.column_config.NumberColumn(format="%.6f"),
                "CI Upper (95%)": st.column_config.NumberColumn(format="%.6f"),
                "p<0.01": st.column_config.CheckboxColumn(),
                "p<0.05": st.column_config.CheckboxColumn(),
                "p<0.10": st.column_config.CheckboxColumn()
            }
        )

        st.caption("Significance levels: *** p<0.01, ** p<0.05, * p<0.10")

        # Add download button for filtered results
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Results (CSV)",
            data=csv_data,
            file_name="dml_results_filtered.csv",
            mime="text/csv"
        )

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
                        st.metric("Coefficient", f"{result['coefficient']:.4f}")
                    with col2:
                        st.metric("Std Error", f"{result['se']:.4f}")
                    with col3:
                        st.metric("95% CI", f"[{result.get('ci_lower_95', result.get('ci_lower', 0)):.3f}, {result.get('ci_upper_95', result.get('ci_upper', 0)):.3f}]")

                    st.markdown(f"""
                    - **Variables:** {', '.join(result['variables'])}
                    - **Order:** {result['order']}-way interaction
                    - **P-value:** {result['p_value']:.4f}
                    - **Interpretation:** This interaction term has a statistically significant effect on the outcome.
                      The coefficient represents the additional effect when these variables interact.
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
                    'CI Lower': f"{r.get('ci_lower_95', r.get('ci_lower', 0)):.3f}",
                    'CI Upper': f"{r.get('ci_upper_95', r.get('ci_upper', 0)):.3f}"
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
