# Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Navigate to the tool directory
cd causal_analysis_gui

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the application
./run.sh
# OR
streamlit run app.py
```

The app will open at http://localhost:8501

## Your First Analysis (10 minutes)

### Option A: Use the Example Data

1. **Upload Data**: Upload `example_data.csv` (included)
2. **Configure Types**: Click "Auto-detect types"
3. **Build DAG**: Add these edges:
   - Age → Education_Years
   - Age → Income
   - Gender → Education_Years
   - Gender → Income
   - Education_Years → Treatment
   - Experience → Treatment
   - Experience → Income
   - Region → Income
   - Treatment → Income

4. **Specify Variables**:
   - Treatment: `Treatment`
   - Outcome: `Income`

5. **Run Analysis**: Click "Run DML Analysis"
6. **View Results**: Check the estimated treatment effect!

### Option B: Use Your Own Data

1. Prepare a CSV file with:
   - At least 100 rows
   - Columns for treatment and outcome
   - Potential confounders

2. Follow the 6-step workflow in the app

## Common Workflows

### Simple A/B Test Analysis
```
Randomization → Treatment → Outcome
Background vars → Outcome
```

### Observational Study
```
Demographics → Treatment
Demographics → Outcome
Socioeconomic → Treatment
Socioeconomic → Outcome
Treatment → Outcome
```

### Mediation Analysis
```
Treatment → Mediator → Outcome
Treatment → Outcome
Confounders → Treatment
Confounders → Mediator
Confounders → Outcome
```

## Tips

1. **Start Simple**: Begin with a basic DAG and add complexity
2. **Domain Knowledge**: Use your subject expertise to build the DAG
3. **Check Assumptions**: Validate that your DAG captures all confounders
4. **Sample Size**: Aim for n > 100 for reliable estimates
5. **Missing Data**: Handle missing values before analysis

## Troubleshooting

**App won't start?**
- Check that all dependencies are installed: `pip list | grep -E "streamlit|dowhy|econml"`
- Try: `pip install --upgrade streamlit`

**Analysis fails?**
- Check for missing values in treatment/outcome
- Ensure sample size > 50
- Verify DAG is acyclic (no loops)

**Need help?**
- See the full README.md for detailed documentation
- Check DoWhy docs: https://www.pywhy.org/dowhy/
- Check EconML docs: https://econml.azurewebsites.net/

## Next Steps

- Read the full README.md for methodology details
- Explore advanced settings in Step 5
- Try sensitivity analyses
- Export and share your results

Happy causal inference!
