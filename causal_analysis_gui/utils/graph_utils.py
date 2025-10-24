"""
Graph Utilities
Functions for DAG validation and manipulation
"""

import networkx as nx


class GraphValidator:
    """Validator for causal DAGs"""

    def validate_dag(self, graph):
        """
        Validate that a graph is a valid DAG

        Args:
            graph (nx.DiGraph): The graph to validate

        Returns:
            tuple: (is_valid: bool, message: str)
        """
        if graph is None:
            return False, "Graph is None"

        if not isinstance(graph, nx.DiGraph):
            return False, "Graph must be a directed graph (DiGraph)"

        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            try:
                cycle = nx.find_cycle(graph)
                cycle_str = " -> ".join([str(u) for u, v in cycle])
                return False, f"Graph contains cycle: {cycle_str}"
            except:
                return False, "Graph contains cycles"

        # Check for self-loops
        if graph.number_of_selfloops() > 0:
            return False, "Graph contains self-loops"

        # Check for isolated nodes (warning, not error)
        isolated = list(nx.isolates(graph))
        if isolated:
            # This is okay, just a warning
            pass

        # Check if graph is connected (undirected version)
        if graph.number_of_nodes() > 1:
            undirected = graph.to_undirected()
            if not nx.is_connected(undirected):
                # This is okay for causal graphs - some variables may not be connected
                pass

        return True, "Valid DAG"

    def check_identification(self, graph, treatment, outcome):
        """
        Check if causal effect is identifiable using backdoor criterion

        Args:
            graph (nx.DiGraph): The causal DAG
            treatment (str): Treatment variable
            outcome (str): Outcome variable

        Returns:
            tuple: (is_identifiable: bool, adjustment_set: list, message: str)
        """
        if not graph.has_node(treatment):
            return False, [], f"Treatment '{treatment}' not in graph"

        if not graph.has_node(outcome):
            return False, [], f"Outcome '{outcome}' not in graph"

        # Find all paths from treatment to outcome
        try:
            paths = list(nx.all_simple_paths(graph, treatment, outcome))
        except nx.NetworkXNoPath:
            return False, [], "No path from treatment to outcome - effect not identifiable"

        # Find confounders (common causes)
        treatment_ancestors = nx.ancestors(graph, treatment)
        outcome_ancestors = nx.ancestors(graph, outcome)
        confounders = treatment_ancestors.intersection(outcome_ancestors)

        # Backdoor adjustment set
        adjustment_set = list(confounders)

        if not adjustment_set:
            # Check if there are any backdoor paths
            # A backdoor path is a path from treatment to outcome that starts with an edge into treatment
            has_backdoor = False
            for node in graph.predecessors(treatment):
                if nx.has_path(graph, node, outcome):
                    has_backdoor = True
                    break

            if has_backdoor:
                return False, [], "Backdoor paths exist but no valid adjustment set found"

        return True, adjustment_set, "Effect is identifiable"

    def get_graph_properties(self, graph):
        """
        Get properties of the graph

        Args:
            graph (nx.DiGraph): The graph

        Returns:
            dict: Graph properties
        """
        properties = {
            'n_nodes': graph.number_of_nodes(),
            'n_edges': graph.number_of_edges(),
            'is_dag': nx.is_directed_acyclic_graph(graph),
            'density': nx.density(graph),
            'is_weakly_connected': nx.is_weakly_connected(graph) if graph.number_of_nodes() > 0 else False,
        }

        return properties

    def find_instrumental_variables(self, graph, treatment, outcome):
        """
        Find potential instrumental variables

        Args:
            graph (nx.DiGraph): The causal DAG
            treatment (str): Treatment variable
            outcome (str): Outcome variable

        Returns:
            list: Potential instrumental variables
        """
        instruments = []

        for node in graph.nodes():
            if node == treatment or node == outcome:
                continue

            # An instrument must:
            # 1. Affect treatment
            # 2. Not directly affect outcome (only through treatment)
            # 3. Not share common causes with outcome

            if not graph.has_edge(node, treatment):
                continue

            if graph.has_edge(node, outcome):
                continue

            # Check for common causes with outcome
            node_ancestors = nx.ancestors(graph, node)
            outcome_ancestors = nx.ancestors(graph, outcome)
            common_ancestors = node_ancestors.intersection(outcome_ancestors)

            if not common_ancestors:
                instruments.append(node)

        return instruments

    def find_mediators(self, graph, treatment, outcome):
        """
        Find mediating variables between treatment and outcome

        Args:
            graph (nx.DiGraph): The causal DAG
            treatment (str): Treatment variable
            outcome (str): Outcome variable

        Returns:
            list: Mediator variables
        """
        mediators = []

        # A mediator is on a path from treatment to outcome
        try:
            for path in nx.all_simple_paths(graph, treatment, outcome):
                # All intermediate nodes are mediators
                mediators.extend(path[1:-1])
        except nx.NetworkXNoPath:
            pass

        return list(set(mediators))

    def suggest_dag_structure(self, variables, variable_types):
        """
        Suggest a default DAG structure based on variable types and names

        Args:
            variables (list): List of variable names
            variable_types (dict): Variable type specifications

        Returns:
            nx.DiGraph: Suggested DAG
        """
        dag = nx.DiGraph()
        dag.add_nodes_from(variables)

        # Simple heuristics for suggesting edges
        # This is a placeholder - in practice, domain knowledge is required

        # Common patterns:
        # - Demographics -> Socioeconomic -> Outcomes
        # - Time-varying variables in order

        demographic_keywords = ['age', 'gender', 'sex', 'race', 'ethnicity', 'born']
        socioeconomic_keywords = ['education', 'income', 'employment', 'occupation']
        outcome_keywords = ['outcome', 'result', 'salary', 'wage', 'health', 'score']

        demographic_vars = [v for v in variables if any(k in v.lower() for k in demographic_keywords)]
        socioeconomic_vars = [v for v in variables if any(k in v.lower() for k in socioeconomic_keywords)]
        outcome_vars = [v for v in variables if any(k in v.lower() for k in outcome_keywords)]

        # Demographics -> Socioeconomic
        for demo in demographic_vars:
            for socio in socioeconomic_vars:
                dag.add_edge(demo, socio)

        # Demographics -> Outcomes
        for demo in demographic_vars:
            for outcome in outcome_vars:
                dag.add_edge(demo, outcome)

        # Socioeconomic -> Outcomes
        for socio in socioeconomic_vars:
            for outcome in outcome_vars:
                dag.add_edge(socio, outcome)

        return dag
