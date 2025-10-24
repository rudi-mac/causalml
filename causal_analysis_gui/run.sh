#!/bin/bash
# Launch script for Interactive Causal Analysis Tool

echo "Starting Interactive Causal Analysis Tool..."
echo "================================================"
echo ""
echo "The application will open in your default browser."
echo "Press Ctrl+C to stop the server."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "Error: Streamlit is not installed."
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

# Run streamlit app
streamlit run app.py
