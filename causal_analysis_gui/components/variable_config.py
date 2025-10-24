"""
Variable Configurator Component
Allows users to specify data types for each variable
"""

import streamlit as st
import pandas as pd


class VariableConfigurator:
    """Component for configuring variable data types"""

    def __init__(self, data):
        """
        Initialize configurator

        Args:
            data (pd.DataFrame): The loaded dataset
        """
        self.data = data
        self.column_types = {}

    def configure_types(self):
        """
        Display interface for configuring data types

        Returns:
            dict: Dictionary mapping column names to data types
        """
        st.markdown("### Variable Type Configuration")

        # Auto-detect types
        if st.button("🔍 Auto-detect types"):
            self.column_types = self._auto_detect_types()
            st.success("✅ Types auto-detected! Review and adjust if needed.")

        # Create a table for type selection
        st.markdown("#### Configure each variable:")

        # Use columns for better layout
        for col in self.data.columns:
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 3])

                with col1:
                    st.markdown(f"**{col}**")

                with col2:
                    # Auto-detect initial value
                    if col not in self.column_types:
                        self.column_types[col] = self._detect_type(col)

                    var_type = st.selectbox(
                        "Type",
                        options=['continuous', 'binary', 'categorical', 'ordinal'],
                        index=['continuous', 'binary', 'categorical', 'ordinal'].index(self.column_types[col]),
                        key=f"type_{col}",
                        label_visibility="collapsed"
                    )
                    self.column_types[col] = var_type

                with col3:
                    # Show sample values
                    sample = self.data[col].dropna().head(3).tolist()
                    st.caption(f"Sample: {sample}")

                st.markdown("---")

        # Type explanation
        with st.expander("ℹ️ Data Type Guide"):
            st.markdown("""
            **Continuous**: Numeric variables that can take any value (e.g., age, income, temperature)

            **Binary**: Variables with exactly two values (e.g., yes/no, 0/1, treatment/control)

            **Categorical**: Variables with multiple discrete categories (e.g., region, occupation, color)

            **Ordinal**: Categorical variables with a natural order (e.g., education level, satisfaction rating)

            Correct type specification helps the DML algorithm choose appropriate preprocessing and models.
            """)

        return self.column_types

    def _detect_type(self, column):
        """
        Auto-detect the likely type of a column

        Args:
            column (str): Column name

        Returns:
            str: Detected type
        """
        col_data = self.data[column]
        n_unique = col_data.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(col_data)

        # Binary detection
        if n_unique == 2:
            return 'binary'

        # Categorical vs continuous
        if is_numeric:
            # If numeric with few unique values, likely categorical
            if n_unique < 10:
                return 'categorical'
            else:
                return 'continuous'
        else:
            # Non-numeric is categorical
            return 'categorical'

    def _auto_detect_types(self):
        """
        Auto-detect types for all columns

        Returns:
            dict: Dictionary of column types
        """
        detected_types = {}
        for col in self.data.columns:
            detected_types[col] = self._detect_type(col)
        return detected_types

    def get_encoded_data(self):
        """
        Get data with appropriate encoding based on types

        Returns:
            pd.DataFrame: Encoded data
        """
        encoded_data = self.data.copy()

        for col, dtype in self.column_types.items():
            if dtype in ['binary', 'categorical', 'ordinal']:
                # Use label encoding for categorical variables
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))

        return encoded_data
