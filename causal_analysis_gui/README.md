# Interactive Causal Analysis Tool

A web-based GUI application for performing **Double Machine Learning (DML)** causal inference on your own data. Built with Streamlit, DoWhy, and EconML.

## Overview

This tool allows researchers and data scientists to:
- Upload custom datasets (CSV format)
- Interactively build causal DAGs (Directed Acyclic Graphs)
- Specify treatment and outcome variables
- Estimate causal effects using state-of-the-art Double Machine Learning
- Visualize results with confidence intervals

Based on the methodology from "Supplementary Material DML" notebook (gender wage gap analysis), this tool generalizes the workflow to any causal analysis problem.

## Features

- **Interactive DAG Editor**: Multiple methods to create causal graphs
  - Point-and-click edge addition
  - Text-based graphviz format
  - File upload support

- **Smart Data Type Detection**: Automatically detects variable types
  - Continuous, binary, categorical, ordinal
  - Customizable type specifications

- **Rigorous Causal Inference**: Uses Double Machine Learning
  - DoWhy framework for causal identification
  - EconML for effect estimation
  - LASSO and LightGBM for nuisance parameter estimation
  - Cross-fitting for bias reduction

- **Comprehensive Results**:
  - Average Treatment Effect (ATE)
  - Confidence intervals
  - Statistical significance tests
  - Exportable results (CSV)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd causal_analysis_gui
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Workflow

The tool guides you through a 6-step workflow:

#### Step 1: Upload Data
- Upload a CSV file with your dataset
- Preview the data and check for missing values
- Each column will become a node in the causal DAG

**Requirements:**
- CSV format with headers
- At least 2 columns (treatment and outcome)
- Preferably no missing values

#### Step 2: Configure Data Types
- Specify the type of each variable:
  - **Continuous**: Numeric variables (e.g., age, income, temperature)
  - **Binary**: Two-value variables (e.g., treatment/control, yes/no)
  - **Categorical**: Multiple discrete categories (e.g., region, occupation)
  - **Ordinal**: Ordered categories (e.g., education level)

- Use the "Auto-detect types" button for automatic inference
- Review and adjust types as needed

#### Step 3: Build Causal DAG
- Create a Directed Acyclic Graph representing your causal assumptions
- Choose from three methods:
  1. **Interactive**: Click to add edges one by one
  2. **Text Format**: Use graphviz DOT syntax
  3. **Upload**: Load a pre-existing graph file

**Important:**
- Draw edges from causes to effects (A → B means "A causes B")
- The graph must be acyclic (no loops)
- Include all relevant confounders

#### Step 4: Specify Variables
- Select the **treatment** variable (intervention of interest)
- Select the **outcome** variable (effect to measure)
- Review identified confounders from the DAG

#### Step 5: Run DML Analysis
- Configure advanced settings (optional):
  - Discrete vs continuous treatment
  - Cross-validation folds
- Click "Run DML Analysis" to estimate causal effects
- Wait for computation (may take 1-5 minutes depending on data size)

#### Step 6: View Results
- Review the estimated Average Treatment Effect (ATE)
- Check confidence intervals and statistical significance
- Interpret the results with provided guidance
- Download results as CSV

## Example

Here's a simple example using synthetic data:

### Data (example.csv)
```csv
Age,Education,Income,Treatment,Outcome
25,12,30000,0,50
30,16,50000,1,75
35,12,35000,0,55
...
```

### DAG Structure
```
Age → Education
Age → Income
Education → Income
Education → Treatment
Income → Treatment
Treatment → Outcome
Age → Outcome
Education → Outcome
```

### Interpretation
If the estimated ATE is **20.5** with 95% CI [15.2, 25.8]:
- The treatment causes an average increase of 20.5 units in the outcome
- We are 95% confident the true effect is between 15.2 and 25.8
- The effect is statistically significant (if p < 0.05)

## Methodology

### Double Machine Learning (DML)

DML combines causal inference with machine learning:

1. **Causal Identification**: Use DAG to identify confounders via backdoor criterion
2. **Nuisance Estimation**: Use ML models (LASSO/LightGBM) to predict:
   - Outcome given confounders: E[Y|X]
   - Treatment given confounders: E[T|X]
3. **Residualization**: Compute residuals
   - Y_res = Y - E[Y|X]
   - T_res = T - E[T|X]
4. **Effect Estimation**: Regress Y_res on T_res
5. **Cross-fitting**: Use cross-validation to avoid overfitting

### Why DML?

- **Flexible**: Works with any ML model for nuisance parameters
- **Robust**: Less sensitive to model misspecification
- **Efficient**: Achieves optimal convergence rates
- **Valid inference**: Provides valid confidence intervals

## Project Structure

```
causal_analysis_gui/
├── app.py                      # Main Streamlit application
├── components/
│   ├── __init__.py
│   ├── data_loader.py         # CSV upload and validation
│   ├── dag_editor.py          # Interactive DAG creation
│   ├── variable_config.py     # Data type configuration
│   └── dml_estimator.py       # DML estimation engine
├── utils/
│   ├── __init__.py
│   └── graph_utils.py         # DAG validation and manipulation
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Dependencies

Core libraries:
- **Streamlit**: Web GUI framework
- **DoWhy**: Causal inference framework
- **EconML**: Econometric machine learning
- **NetworkX**: Graph analysis
- **LightGBM**: Gradient boosting models
- **scikit-learn**: Machine learning utilities

See `requirements.txt` for complete list with versions.

## Troubleshooting

### Installation Issues

**Problem**: `pip install` fails for DoWhy or EconML
- **Solution**: Try installing separately:
  ```bash
  pip install dowhy==0.11.1
  pip install econml==0.15.0
  ```

**Problem**: Missing system dependencies for graphviz
- **Solution**: Install graphviz system package:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install graphviz libgraphviz-dev

  # macOS
  brew install graphviz

  # Windows
  # Download from https://graphviz.org/download/
  ```

### Runtime Issues

**Problem**: "Graph contains cycles" error
- **Solution**: Check your DAG for circular dependencies. Use the visualization to identify loops.

**Problem**: DML estimation fails
- **Solution**:
  - Ensure sufficient sample size (n > 100 recommended)
  - Check for missing values in treatment/outcome
  - Verify data types are correctly specified

**Problem**: Very large confidence intervals
- **Solution**:
  - Increase sample size
  - Check for strong confounding
  - Consider different model specifications

## Limitations

- Assumes no unobserved confounding (given the DAG)
- Requires correct DAG specification (domain knowledge essential)
- Large datasets (n > 100,000) may be slow
- Currently supports only single treatment and outcome

## Future Enhancements

Potential improvements:
- [ ] Multiple treatment variables
- [ ] Time-series/panel data support
- [ ] Heterogeneous treatment effects (CATE) visualization
- [ ] Sensitivity analysis tools
- [ ] Causal discovery algorithms
- [ ] Integration with other estimators (IV, RDD, etc.)

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{causal_analysis_gui,
  title = {Interactive Causal Analysis Tool},
  author = {[Your Name]},
  year = {2024},
  note = {Based on DoWhy and EconML frameworks}
}
```

## References

- **DoWhy**: Sharma, A., & Kiciman, E. (2020). DoWhy: An end-to-end library for causal inference. arXiv preprint arXiv:2011.04216.
- **EconML**: Battocchi, K., et al. (2019). EconML: A Python package for ML-based heterogeneous treatment effects estimation.
- **DML**: Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal.

## License

This project is provided as-is for educational and research purposes.

## Support

For issues, questions, or contributions:
- Open an issue on the repository
- Check the DoWhy documentation: https://www.pywhy.org/dowhy/
- Check the EconML documentation: https://econml.azurewebsites.net/

## Acknowledgments

This tool was developed based on the "Supplementary Material DML" notebook analyzing the gender wage gap. It generalizes that methodology to any causal analysis problem.

Special thanks to the DoWhy and EconML development teams for their excellent causal inference frameworks.
