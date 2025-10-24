"""
Data Loader Component
Handles CSV file upload and initial data validation
"""

import streamlit as st
import pandas as pd
import io


class DataLoader:
    """Component for loading and validating CSV data"""

    def __init__(self):
        self.data = None

    def load_data(self):
        """
        Display file uploader and load CSV data

        Returns:
            pd.DataFrame: Loaded data or None
        """
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with your data. First row should contain column names."
        )

        if uploaded_file is not None:
            try:
                # Read CSV
                self.data = pd.read_csv(uploaded_file)

                # Validate
                if self.data.empty:
                    st.error("❌ The uploaded file is empty!")
                    return None

                if len(self.data.columns) < 2:
                    st.error("❌ Data must have at least 2 columns (treatment and outcome)!")
                    return None

                # Check for missing values
                missing_count = self.data.isnull().sum().sum()
                if missing_count > 0:
                    st.warning(f"⚠️ Data contains {missing_count} missing values. Consider cleaning your data.")

                    # Show missing values per column
                    missing_per_col = self.data.isnull().sum()
                    missing_cols = missing_per_col[missing_per_col > 0]

                    if len(missing_cols) > 0:
                        with st.expander("View missing values by column"):
                            st.dataframe(missing_cols.to_frame(name='Missing Count'))

                        # Option to drop missing values
                        if st.checkbox("Drop rows with missing values?"):
                            original_shape = self.data.shape
                            self.data = self.data.dropna()
                            st.info(f"Dropped {original_shape[0] - self.data.shape[0]} rows. New shape: {self.data.shape}")

                return self.data

            except pd.errors.EmptyDataError:
                st.error("❌ The file is empty or not a valid CSV!")
                return None
            except pd.errors.ParserError as e:
                st.error(f"❌ Error parsing CSV file: {str(e)}")
                return None
            except Exception as e:
                st.error(f"❌ Unexpected error loading file: {str(e)}")
                return None

        return None

    def get_column_info(self):
        """
        Get information about columns

        Returns:
            dict: Column information including detected types
        """
        if self.data is None:
            return {}

        column_info = {}
        for col in self.data.columns:
            dtype = self.data[col].dtype
            n_unique = self.data[col].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(dtype)

            # Infer likely type
            if n_unique == 2:
                likely_type = 'binary'
            elif not is_numeric or n_unique < 10:
                likely_type = 'categorical'
            else:
                likely_type = 'continuous'

            column_info[col] = {
                'dtype': str(dtype),
                'n_unique': n_unique,
                'likely_type': likely_type,
                'sample_values': self.data[col].dropna().head(3).tolist()
            }

        return column_info
