import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO
import os
import numpy as np

@st.cache_data
def load_data():
    """Load data with fallback options and error handling"""
    local_path = "/Users/aryangupta/Desktop/Final_LLM_dataframe.csv"
    drive_url = "https://drive.google.com/uc?id=17sviDKgkOMu6CYmhSorvFQ24skC2j0Kl&export=download"
    
    if os.path.exists(local_path):
        try:
            return pd.read_csv(local_path)
        except Exception as e:
            st.warning(f"Error loading local file: {e}")
    
    try:
        with st.spinner("Loading data from Google Drive..."):
            response = requests.get(drive_url, timeout=30)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
    except Exception as e:
        st.error(f"Error downloading or processing CSV: {e}")
        return None

df = load_data()
if df is None:
    st.stop()

# ---------------- Data Processing ----------------
required_columns = [
    "Prompt_ID", "Prompt_Text_x", "Output", "Output_ref1", "Output_ref2",
    "Category", "Latency_sec", "Tokens", "Tokens_per_sec",
    "Embedding_avg", "BLEU_avg", "ROUGE1_avg", "ROUGEL_avg", "G-Eval Avg", "Model_Name"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

df = df[required_columns].rename(columns={
    "Prompt_Text_x": "Prompt_Text",
    "G-Eval Avg": "G_Eval_avg"
})
df["G_Eval_avg"] = df["G_Eval_avg"] / 10.0

METRICS = ["Embedding_avg", "BLEU_avg", "ROUGE1_avg", "ROUGEL_avg", "G_Eval_avg"]
METRIC_LABELS = {
    "Embedding_avg": "Embedding Avg",
    "BLEU_avg": "BLEU Avg",
    "ROUGE1_avg": "ROUGE-1 Avg",
    "ROUGEL_avg": "ROUGE-L Avg",
    "G_Eval_avg": "G-Eval Avg (0–10)"
}

# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide", page_title="LLM Evaluation Dashboard")
st.title("📊 LLM Evaluation Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Prompts", len(df))
col2.metric("Models", df["Model_Name"].nunique())
col3.metric("Categories", df["Category"].nunique())
col4.metric("Avg Latency", f"{df['Latency_sec'].mean():.2f}s")

page = st.sidebar.radio("Select Page", ["Leaderboard", "Compare Models", "Per-Model Heatmaps", "Raw Data"])

# ---------------- Leaderboard ----------------
if page == "Leaderboard":
    st.header("🏆 Leaderboard by Average Scores")
    metric_to_sort = st.selectbox(
        "Select Metric to Sort",
        options=METRICS,
        index=1,
        format_func=lambda m: METRIC_LABELS[m]
    )

    leaderboard = df.groupby("Model_Name")[METRICS].mean().sort_values(metric_to_sort, ascending=False)

    tab1, tab2 = st.tabs(["📋 Table View", "📊 Bar Chart"])
    with tab1:
        st.dataframe(leaderboard.style.highlight_max(axis=0), use_container_width=True)
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        leaderboard[metric_to_sort].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(f"Model Ranking by {METRIC_LABELS[metric_to_sort]}")
        ax.set_ylabel(METRIC_LABELS[metric_to_sort])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

# ---------------- Compare Models ----------------
elif page == "Compare Models":
    st.header("🔍 Compare Multiple Models Across Categories")

    models = df["Model_Name"].unique()
    categories = df["Category"].unique()
    selected_models = st.multiselect("Select Models", models, default=models[:2] if len(models) >= 2 else models)
    selected_categories = st.multiselect("Select Categories", categories, default=categories[:3] if len(categories) >= 3 else categories)
    selected_metrics = st.multiselect("Select Metrics", METRICS, default=["BLEU_avg"], format_func=lambda m: METRIC_LABELS[m])

    if selected_models and selected_categories and selected_metrics:
        df_comp = df[(df["Model_Name"].isin(selected_models)) & (df["Category"].isin(selected_categories))]
        if not df_comp.empty:
            for metric in selected_metrics:
                with st.expander(f"📊 {METRIC_LABELS[metric]} Comparison", expanded=True):
                    heatmap_data = df_comp.pivot_table(
                        index="Category",
                        columns="Model_Name",
                        values=metric,
                        aggfunc="mean"
                    ).clip(0,1)
                    if not heatmap_data.empty:
                        fig, ax = plt.subplots(figsize=(max(6, len(selected_models)*1.5), max(3, len(selected_categories)*0.6)))
                        cax = ax.matshow(heatmap_data.values, cmap="YlGnBu", vmin=0, vmax=1)
                        fig.colorbar(cax, ax=ax, shrink=0.6)
                        ax.set_xticks(np.arange(len(heatmap_data.columns)))
                        ax.set_xticklabels(heatmap_data.columns, rotation=30, ha="right")
                        ax.set_yticks(np.arange(len(heatmap_data.index)))
                        ax.set_yticklabels(heatmap_data.index)
                        ax.set_title(f"{METRIC_LABELS[metric]} Across Models", pad=12)
                        st.pyplot(fig)
                    else:
                        st.warning("No data available for this metric combination.")
        else:
            st.warning("No data available for selected models and categories.")

# ---------------- Per-Model Heatmaps ----------------
elif page == "Per-Model Heatmaps":
    st.header("🔥 Per-Model Metric Heatmaps by Category")

    models = df["Model_Name"].unique()
    categories = df["Category"].unique()
    selected_models = st.multiselect("Select Models", models, default=models[:1])
    selected_categories = st.multiselect("Select Categories", categories, default=categories[:3] if len(categories) >= 3 else categories)

    if selected_models and selected_categories:
        df_heat = df[(df["Model_Name"].isin(selected_models)) & (df["Category"].isin(selected_categories))]
        if not df_heat.empty:
            for model in selected_models:
                st.subheader(f"Model: {model}")
                df_model = df_heat[df_heat["Model_Name"] == model]
                if not df_model.empty:
                    tabs = st.tabs([METRIC_LABELS[m] for m in METRICS])
                    for tab, metric in zip(tabs, METRICS):
                        with tab:
                            heatmap_data = df_model.pivot_table(
                                index="Category",
                                values=metric,
                                aggfunc="mean"
                            ).clip(0,1)
                            if not heatmap_data.empty:
                                fig, ax = plt.subplots(figsize=(4, max(3, len(selected_categories)*0.6)))
                                cax = ax.matshow(heatmap_data.values, cmap="viridis", vmin=0, vmax=1)
                                fig.colorbar(cax, ax=ax)
                                ax.set_xticks([])
                                ax.set_yticks(np.arange(len(heatmap_data.index)))
                                ax.set_yticklabels(heatmap_data.index)
                                ax.set_title(f"{METRIC_LABELS[metric]} Heatmap", pad=10)
                                st.pyplot(fig)
                            else:
                                st.warning(f"No data available for {metric} metric.")
        else:
            st.warning("No data available for selected models and categories.")

# ---------------- Raw Data ----------------
elif page == "Raw Data":
    st.header("📄 Raw Data")
    col1, col2 = st.columns(2)
    with col1:
        model_filter = st.multiselect("Filter by Model", df["Model_Name"].unique(), default=df["Model_Name"].unique())
    with col2:
        category_filter = st.multiselect("Filter by Category", df["Category"].unique(), default=df["Category"].unique())
    filtered_df = df[(df["Model_Name"].isin(model_filter)) & (df["Category"].isin(category_filter))]
    st.dataframe(filtered_df, use_container_width=True)
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered CSV", csv, file_name="llm_eval_results_filtered.csv", mime="text/csv")
    st.subheader("Data Statistics")
    st.write(f"Showing {len(filtered_df)} rows out of {len(df)} total rows")
    if not filtered_df.empty:
        st.write("Numeric columns summary:")
        st.dataframe(filtered_df[METRICS + ["Latency_sec", "Tokens", "Tokens_per_sec"]].describe())