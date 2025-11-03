"""
DAG Editor Component
Interactive editor for creating causal DAGs
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64


class DAGEditor:
    """Component for creating and editing causal DAGs"""

    def __init__(self, variables, interactive=False):
        """
        Initialize DAG editor

        Args:
            variables (list): List of variable names (nodes)
            interactive (bool): If True, allow adding/removing variables dynamically
        """
        self.variables = variables
        self.interactive = interactive
        self.dag = nx.DiGraph()
        # Add all variables as nodes if provided
        if variables:
            self.dag.add_nodes_from(variables)

    def create_dag(self):
        """
        Display interface for creating a DAG

        Returns:
            nx.DiGraph: The created DAG
        """
        st.markdown("### Build Your Causal DAG")

        # Initialize edges in session state if not exists
        if 'dag_edges' not in st.session_state:
            st.session_state.dag_edges = []

        # Method selection
        method = st.radio(
            "DAG Creation Method:",
            ["Interactive Edge Addition", "Text Format (graphviz)", "Upload Graph"],
            horizontal=False
        )

        if method == "Interactive Edge Addition":
            dag = self._interactive_editor()
        elif method == "Text Format (graphviz)":
            dag = self._text_format_editor()
        else:
            dag = self._upload_editor()

        return dag

    def _interactive_editor(self):
        """
        Interactive edge-by-edge editor

        Returns:
            nx.DiGraph: The created DAG
        """
        st.markdown("#### Add Causal Relationships")
        st.markdown("For each edge, specify: **Cause → Effect**")

        # Edge addition form
        with st.form("add_edge_form", clear_on_submit=True):
            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                from_node = st.selectbox(
                    "From (Cause)",
                    options=self.variables,
                    key="from_node"
                )

            with col2:
                st.markdown("<div style='text-align: center; padding-top: 28px;'>→</div>", unsafe_allow_html=True)

            with col3:
                to_node = st.selectbox(
                    "To (Effect)",
                    options=self.variables,
                    key="to_node"
                )

            submitted = st.form_submit_button("➕ Add Edge")

            if submitted and from_node != to_node:
                edge = (from_node, to_node)
                if edge not in st.session_state.dag_edges:
                    st.session_state.dag_edges.append(edge)
                    st.success(f"✅ Added: {from_node} → {to_node}")
                else:
                    st.warning(f"⚠️ Edge already exists: {from_node} → {to_node}")

        # Show current edges
        if st.session_state.dag_edges:
            st.markdown("#### Current Edges")

            # Display edges in a table
            edges_df = [[i+1, e[0], "→", e[1]] for i, e in enumerate(st.session_state.dag_edges)]

            for i, (idx, from_n, arrow, to_n) in enumerate(edges_df):
                col1, col2, col3 = st.columns([1, 4, 1])
                with col1:
                    st.text(f"{idx}.")
                with col2:
                    st.text(f"{from_n} → {to_n}")
                with col3:
                    if st.button("🗑️", key=f"del_{i}"):
                        st.session_state.dag_edges.pop(i)
                        st.rerun()

            st.markdown("---")

            # Build DAG from edges
            self.dag.add_edges_from(st.session_state.dag_edges)

            # Visualize
            st.markdown("#### DAG Visualization")
            fig = self._visualize_dag(self.dag)
            st.pyplot(fig)

            # Clear all button
            if st.button("🗑️ Clear All Edges", type="secondary"):
                st.session_state.dag_edges = []
                st.rerun()

        else:
            st.info("👆 Add edges above to build your causal graph")

        return self.dag if st.session_state.dag_edges else None

    def _text_format_editor(self):
        """
        Text-based editor using graphviz format

        Returns:
            nx.DiGraph: The created DAG
        """
        st.markdown("#### Enter DAG in DOT Format")

        example = f"""digraph {{
    {self.variables[0]} -> {self.variables[1]};
    {self.variables[0]} -> {self.variables[-1]};
}}"""

        with st.expander("📖 Format Guide & Example"):
            st.markdown("""
            Use graphviz DOT format:
            ```
            digraph {
                Variable1 -> Variable2;
                Variable1 -> Variable3;
                Variable2 -> Variable3;
            }
            ```

            **Rules:**
            - Start with `digraph {` and end with `}`
            - Each edge: `source -> target;`
            - Use exact variable names from your data
            - Separate statements with semicolons
            """)
            st.code(example, language="dot")

        graph_text = st.text_area(
            "Enter your DAG:",
            height=200,
            placeholder=example
        )

        if graph_text:
            try:
                # Parse graphviz format
                import re

                # Extract edges from graphviz format
                edge_pattern = r'(\w+)\s*->\s*(\w+)'
                edges = re.findall(edge_pattern, graph_text)

                if edges:
                    # Validate that all nodes are in variables
                    all_nodes = set()
                    for from_n, to_n in edges:
                        all_nodes.add(from_n)
                        all_nodes.add(to_n)

                    invalid_nodes = all_nodes - set(self.variables)
                    if invalid_nodes:
                        st.error(f"❌ Invalid node names: {invalid_nodes}")
                        st.info(f"Valid variables: {', '.join(self.variables)}")
                        return None

                    # Build DAG
                    self.dag.add_edges_from(edges)

                    st.success(f"✅ Parsed {len(edges)} edges successfully!")

                    # Visualize
                    st.markdown("#### DAG Visualization")
                    fig = self._visualize_dag(self.dag)
                    st.pyplot(fig)

                    return self.dag
                else:
                    st.warning("⚠️ No edges found. Check format.")
                    return None

            except Exception as e:
                st.error(f"❌ Error parsing DAG: {str(e)}")
                return None

        return None

    def _upload_editor(self):
        """
        Upload DAG from file

        Returns:
            nx.DiGraph: The created DAG
        """
        st.markdown("#### Upload DAG File")

        st.info("""
        Upload a DAG in one of these formats:
        - **Edge list** (.txt): Each line contains `source,target`
        - **GraphML** (.graphml): Standard graph format
        - **DOT** (.dot): Graphviz format
        """)

        uploaded_file = st.file_uploader(
            "Choose a graph file",
            type=['txt', 'graphml', 'dot', 'gml']
        )

        if uploaded_file:
            try:
                file_type = uploaded_file.name.split('.')[-1].lower()

                if file_type == 'txt':
                    # Edge list format
                    content = uploaded_file.read().decode('utf-8')
                    edges = []
                    for line in content.strip().split('\n'):
                        parts = line.strip().split(',')
                        if len(parts) == 2:
                            edges.append((parts[0].strip(), parts[1].strip()))

                    self.dag.add_edges_from(edges)

                elif file_type == 'graphml':
                    # GraphML format
                    self.dag = nx.read_graphml(uploaded_file)

                elif file_type in ['dot', 'gml']:
                    st.error(f"❌ {file_type.upper()} format not yet supported. Use edge list (.txt) or graphml.")
                    return None

                st.success(f"✅ Loaded graph with {self.dag.number_of_edges()} edges")

                # Visualize
                fig = self._visualize_dag(self.dag)
                st.pyplot(fig)

                return self.dag

            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                return None

        return None

    def _visualize_dag(self, dag):
        """
        Visualize the DAG using matplotlib

        Args:
            dag (nx.DiGraph): The DAG to visualize

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Use hierarchical layout if possible
        try:
            pos = nx.spring_layout(dag, k=2, iterations=50)
        except:
            pos = nx.circular_layout(dag)

        # Draw nodes
        nx.draw_networkx_nodes(
            dag, pos,
            node_color='lightblue',
            node_size=2000,
            alpha=0.9,
            ax=ax
        )

        # Draw edges
        nx.draw_networkx_edges(
            dag, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            width=2,
            ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(
            dag, pos,
            font_size=10,
            font_weight='bold',
            ax=ax
        )

        ax.set_title("Causal DAG", fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        return fig

    def create_dag_with_variables(self):
        """
        Interactive DAG creation with variable definition
        Allows user to define treatment, outcome, and additional variables with data types

        Returns:
            tuple: (dag, treatment, outcome, variables_config)
                - dag: NetworkX DiGraph
                - treatment: str, name of treatment variable
                - outcome: str, name of outcome variable
                - variables_config: dict of {var_name: {'type': 'continuous/binary/categorical/ordinal'}}
        """
        st.markdown("### Step 1: Define Treatment and Outcome")

        # Initialize variables in session state
        if 'dag_builder_vars' not in st.session_state:
            st.session_state.dag_builder_vars = {}
        if 'dag_builder_edges' not in st.session_state:
            st.session_state.dag_builder_edges = []
        if 'dag_builder_treatment' not in st.session_state:
            st.session_state.dag_builder_treatment = None
        if 'dag_builder_outcome' not in st.session_state:
            st.session_state.dag_builder_outcome = None

        # Treatment and Outcome definition
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Treatment Variable")
            treatment_name = st.text_input("Treatment variable name:",
                                          value=st.session_state.dag_builder_treatment or "",
                                          key="treatment_name_input",
                                          placeholder="e.g., Education, Drug, Training")
            treatment_type = st.selectbox("Treatment data type:",
                                         ['binary', 'categorical', 'continuous', 'ordinal'],
                                         key="treatment_type_select")

            if treatment_name and st.button("✅ Set Treatment", key="set_treatment"):
                st.session_state.dag_builder_treatment = treatment_name
                st.session_state.dag_builder_vars[treatment_name] = {'type': treatment_type}
                st.success(f"✅ Treatment set: {treatment_name} ({treatment_type})")
                st.rerun()

        with col2:
            st.subheader("📊 Outcome Variable")
            outcome_name = st.text_input("Outcome variable name:",
                                        value=st.session_state.dag_builder_outcome or "",
                                        key="outcome_name_input",
                                        placeholder="e.g., Salary, Health, Performance")
            outcome_type = st.selectbox("Outcome data type:",
                                       ['continuous', 'binary', 'categorical', 'ordinal'],
                                       key="outcome_type_select")

            if outcome_name and st.button("✅ Set Outcome", key="set_outcome"):
                st.session_state.dag_builder_outcome = outcome_name
                st.session_state.dag_builder_vars[outcome_name] = {'type': outcome_type}
                st.success(f"✅ Outcome set: {outcome_name} ({outcome_type})")
                st.rerun()

        # Show current treatment and outcome
        if st.session_state.dag_builder_treatment and st.session_state.dag_builder_outcome:
            st.success(f"✅ Treatment: **{st.session_state.dag_builder_treatment}** ({st.session_state.dag_builder_vars[st.session_state.dag_builder_treatment]['type']}) → Outcome: **{st.session_state.dag_builder_outcome}** ({st.session_state.dag_builder_vars[st.session_state.dag_builder_outcome]['type']})")

            # Step 2: Add additional variables
            st.markdown("---")
            st.markdown("### Step 2: Add Additional Variables (Covariates)")

            with st.form("add_variable_form", clear_on_submit=True):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    new_var_name = st.text_input("Variable name:", key="new_var_name", placeholder="e.g., Age, Gender, Region")

                with col2:
                    new_var_type = st.selectbox("Data type:", ['continuous', 'binary', 'categorical', 'ordinal'], key="new_var_type")

                with col3:
                    st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
                    add_var = st.form_submit_button("➕ Add Variable")

                if add_var and new_var_name:
                    if new_var_name not in st.session_state.dag_builder_vars:
                        st.session_state.dag_builder_vars[new_var_name] = {'type': new_var_type}
                        st.success(f"✅ Added: {new_var_name} ({new_var_type})")
                        st.rerun()
                    else:
                        st.warning(f"⚠️ Variable '{new_var_name}' already exists!")

            # Show current variables
            if len(st.session_state.dag_builder_vars) > 2:
                st.markdown("#### Current Variables")
                for var_name, var_config in st.session_state.dag_builder_vars.items():
                    if var_name not in [st.session_state.dag_builder_treatment, st.session_state.dag_builder_outcome]:
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            st.text(f"📌 {var_name}")
                        with col2:
                            st.caption(f"Type: {var_config['type']}")
                        with col3:
                            if st.button("🗑️", key=f"del_var_{var_name}"):
                                # Remove variable and any edges connected to it
                                st.session_state.dag_builder_vars.pop(var_name)
                                st.session_state.dag_builder_edges = [
                                    e for e in st.session_state.dag_builder_edges
                                    if e[0] != var_name and e[1] != var_name
                                ]
                                st.rerun()

            # Step 3: Define causal relationships (edges)
            st.markdown("---")
            st.markdown("### Step 3: Define Causal Relationships (Edges)")

            if len(st.session_state.dag_builder_vars) >= 2:
                with st.form("add_edge_form", clear_on_submit=True):
                    col1, col2, col3 = st.columns([2, 1, 2])

                    var_list = list(st.session_state.dag_builder_vars.keys())

                    with col1:
                        from_node = st.selectbox("From (Cause)", options=var_list, key="from_node_select")

                    with col2:
                        st.markdown("<div style='text-align: center; padding-top: 28px;'>→</div>", unsafe_allow_html=True)

                    with col3:
                        to_node = st.selectbox("To (Effect)", options=var_list, key="to_node_select")

                    submitted = st.form_submit_button("➕ Add Edge")

                    if submitted and from_node != to_node:
                        edge = (from_node, to_node)
                        if edge not in st.session_state.dag_builder_edges:
                            st.session_state.dag_builder_edges.append(edge)
                            st.success(f"✅ Added: {from_node} → {to_node}")
                            st.rerun()
                        else:
                            st.warning(f"⚠️ Edge already exists: {from_node} → {to_node}")

                # Show current edges
                if st.session_state.dag_builder_edges:
                    st.markdown("#### Current Edges")
                    for i, (from_n, to_n) in enumerate(st.session_state.dag_builder_edges):
                        col1, col2, col3 = st.columns([1, 4, 1])
                        with col1:
                            st.text(f"{i+1}.")
                        with col2:
                            st.text(f"{from_n} → {to_n}")
                        with col3:
                            if st.button("🗑️", key=f"del_edge_{i}"):
                                st.session_state.dag_builder_edges.pop(i)
                                st.rerun()

                # Build and visualize DAG
                if st.session_state.dag_builder_vars:
                    st.markdown("---")
                    st.markdown("### Step 4: DAG Visualization")

                    # Build DAG
                    self.dag = nx.DiGraph()
                    self.dag.add_nodes_from(st.session_state.dag_builder_vars.keys())
                    self.dag.add_edges_from(st.session_state.dag_builder_edges)

                    # Visualize with interactive layout
                    fig = self._visualize_dag_interactive(
                        self.dag,
                        treatment=st.session_state.dag_builder_treatment,
                        outcome=st.session_state.dag_builder_outcome
                    )
                    st.pyplot(fig)

                    # Data type guide
                    with st.expander("ℹ️ Data Type Guide"):
                        st.markdown("""
                        **Continuous**: Numeric variables that can take any value (e.g., age, income, temperature)

                        **Binary**: Variables with exactly two values (e.g., yes/no, 0/1, treatment/control)

                        **Categorical**: Variables with multiple discrete categories (e.g., region, occupation, color)

                        **Ordinal**: Categorical variables with a natural order (e.g., education level, satisfaction rating)

                        Correct type specification helps the DML algorithm choose appropriate preprocessing and models.
                        """)

                    # Return DAG if complete
                    return (
                        self.dag,
                        st.session_state.dag_builder_treatment,
                        st.session_state.dag_builder_outcome,
                        st.session_state.dag_builder_vars
                    )

        return None, None, None, None

    def _visualize_dag_interactive(self, dag, treatment=None, outcome=None):
        """
        Visualize the DAG with special highlighting for treatment and outcome

        Args:
            dag (nx.DiGraph): The DAG to visualize
            treatment (str): Treatment variable name
            outcome (str): Outcome variable name

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        # Use hierarchical layout
        try:
            # Try to use a hierarchical layout if treatment->outcome path exists
            if treatment and outcome and dag.has_node(treatment) and dag.has_node(outcome):
                pos = nx.spring_layout(dag, k=3, iterations=100, seed=42)
            else:
                pos = nx.spring_layout(dag, k=2, iterations=50, seed=42)
        except:
            pos = nx.circular_layout(dag)

        # Prepare node colors
        node_colors = []
        for node in dag.nodes():
            if node == treatment:
                node_colors.append('lightgreen')
            elif node == outcome:
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightblue')

        # Draw nodes
        nx.draw_networkx_nodes(
            dag, pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.9,
            ax=ax
        )

        # Draw edges
        nx.draw_networkx_edges(
            dag, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=25,
            arrowstyle='->',
            width=2.5,
            ax=ax,
            connectionstyle='arc3,rad=0.1'
        )

        # Draw labels
        nx.draw_networkx_labels(
            dag, pos,
            font_size=11,
            font_weight='bold',
            ax=ax
        )

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', label='Treatment'),
            Patch(facecolor='lightcoral', label='Outcome'),
            Patch(facecolor='lightblue', label='Covariate')
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        ax.set_title("Causal DAG (Drag nodes in your mind - layout is auto-generated)",
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        return fig

    def export_dag(self, dag, format='edgelist'):
        """
        Export DAG to various formats

        Args:
            dag (nx.DiGraph): The DAG to export
            format (str): Export format ('edgelist', 'graphml', 'dot')

        Returns:
            str: Exported graph data
        """
        if format == 'edgelist':
            edges = [f"{u},{v}" for u, v in dag.edges()]
            return "\n".join(edges)

        elif format == 'graphml':
            from io import StringIO
            buffer = StringIO()
            nx.write_graphml(dag, buffer)
            return buffer.getvalue()

        elif format == 'dot':
            return nx.nx_pydot.to_pydot(dag).to_string()

        return ""
