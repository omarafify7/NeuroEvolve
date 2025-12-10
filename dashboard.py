import streamlit as st
import pandas as pd
import json
import time
import os

st.set_page_config(page_title="NeuroEvolve Dashboard", layout="wide")

st.title("🧬 NeuroEvolve: Real-Time Training Dashboard")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Fitness Evolution")
    chart_placeholder = st.empty()

with col2:
    st.subheader("Current Status")
    metrics_placeholder = st.empty()
    st.subheader("Best Architecture")
    arch_placeholder = st.empty()

def load_data():
    if os.path.exists('evolution_history.csv'):
        return pd.read_csv('evolution_history.csv')
    return pd.DataFrame(columns=['generation', 'best_fitness', 'avg_fitness'])

def load_best_genome():
    if os.path.exists('best_genome_golden.json'):
        with open('best_genome_golden.json', 'r') as f:
            return json.load(f)
    return None

# Auto-refresh loop
while True:
    df = load_data()
    
    if not df.empty:
        # Update Chart
        with chart_placeholder:
            st.line_chart(df.set_index('generation')[['best_fitness', 'avg_fitness']])
            
        # Update Metrics
        latest = df.iloc[-1]
        with metrics_placeholder:
            # st.metric("Generation", int(latest['generation']))
            # st.metric("Avg Fitness", f"{latest['avg_fitness']:.4f}")
            st.metric("Best Fitness", f"{latest['best_fitness']:.4f}")
            
    # Update Architecture
    genes = load_best_genome()
    if genes:
        with arch_placeholder:
            st.json(genes)
    else:
        with arch_placeholder:
            st.info("No best model saved yet.")
            
    time.sleep(2) # Refresh every 2 seconds
