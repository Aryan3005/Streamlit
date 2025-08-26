import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import StringIO
import os

# ---------------- Load Data with Error Handling ----------------
@st.cache_data
def load_data():
    """Load data with fallback options and error handling"""
    local_path = "/Users/aryangupta/Desktop/Final_LLM_dataframe.csv"
    drive_url = "https://drive.google.com/uc?id=17sviDKgkOMu6CYmhSorvFQ24skC2j0Kl&export=download"
    
    # Try local file first
    if os.path.exists(local_path):
        try:
            return pd.read_csv(local_path)
        except Exception as e:
            st.warning(f"Error loading local file: {e}")
    
    # Use requests for remote URL (handles SSL better)
    try:
        with st.spinner("Loading data from Google Drive..."):
            response = requests.get(drive_url, timeout=30)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
    except requests.RequestException as e:
        st.error(f"Error downloading CSV: {e}")
        st.error("Please check your internet connection or file permissions.")
        return None
    except Exception as e:
        st.error(f"Error processing CSV data: {e}")
        return None

# Load the data
df = load_data()

if df is None:
    st.error("❌ Failed to load data. Please check the file source.")
    st.stop()

# ---------------- Data Processing ----------------
try:
    # Keep only necessary columns
    required_columns = [
        "Prompt_ID", "Prompt_Text_x", "Output", "Output_ref1", "Output_ref2",
        "Category", "Latency_sec", "Tokens", "Tokens_per_sec",
        "Embedding_avg", "BLEU_avg", "ROUGE1_avg", "ROUGEL_avg", "G-Eval Avg", "Model_Name"
    ]
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"❌ Missing columns in dataset: {missing_columns}")
        st.info("Available columns: " + ", ".join(df.columns.tolist()))
        st.stop()
    
    df = df[required_columns].rename(columns={
        "Prompt_Text_x": "Prompt_Text",
        "G-Eval Avg": "G_Eval_avg"
    })

    # 🔥 Normalize G-Eval Avg (convert 0–10 → 0–1 scale)
    df["G_Eval_avg"] = df["G_Eval_avg"] / 10.0

except Exception as e:
    st.error(f"❌ Error processing data: {e}")
    st.stop()

# Metrics list (normalized internally)
METRICS = ["Embedding_avg", "BLEU_avg", "ROUGE1_avg", "ROUGEL_avg", "G_Eval_avg"]

# Mapping for pretty labels in UI
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

# Show data summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Prompts", len(df))
with col2:
    st.metric("Models", df["Model_Name"].nunique())
with col3:
    st.metric("Categories", df["Category"].nunique())
with col4:
    st.metric("Avg Latency", f"{df['Latency_sec'].mean():.2f}s")

# Sidebar Navigation
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
    selected_metrics = st.multiselect(
        "Select Metrics", METRICS, default=["BLEU_avg"],
        format_func=lambda m: METRIC_LABELS[m]
    )

    if selected_models and selected_categories and selected_metrics:
        df_comp = df[(df["Model_Name"].isin(selected_models)) & 
                     (df["Category"].isin(selected_categories))]

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
                        fig, ax = plt.subplots(
                            figsize=(max(6, len(selected_models)*1.5), max(3, len(selected_categories)*0.6))
                        )
                        sns.heatmap(
                            heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu",
                            linewidths=0.3, linecolor='gray', vmin=0, vmax=1,
                            cbar_kws={"shrink": 0.6}, ax=ax
                        )
                        ax.set_title(f"{METRIC_LABELS[metric]} Across Models", fontsize=12, pad=12)
                        plt.xticks(rotation=30, ha="right")
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("No data available for this metric combination.")
        else:
            st.warning("⚠️ No data available for selected models and categories.")
    else:
        st.info("ℹ️ Please select at least one model, category, and metric.")

# ---------------- Per-Model Heatmaps ----------------
elif page == "Per-Model Heatmaps":
    st.header("🔥 Per-Model Metric Heatmaps by Category")

    models = df["Model_Name"].unique()
    categories = df["Category"].unique()

    selected_models = st.multiselect("Select Models", models, default=models[:1])
    selected_categories = st.multiselect("Select Categories", categories, default=categories[:3] if len(categories) >= 3 else categories)

    if selected_models and selected_categories:
        df_heat = df[(df["Model_Name"].isin(selected_models)) & 
                     (df["Category"].isin(selected_categories))]
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
                                fig, ax = plt.subplots(
                                    figsize=(4, max(3, len(selected_categories)*0.6))
                                )
                                sns.heatmap(
                                    heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                                    linewidths=0.5, linecolor='gray', vmin=0, vmax=1, ax=ax
                                )
                                ax.set_title(f"{METRIC_LABELS[metric]} Heatmap", fontsize=12, pad=10)
                                plt.yticks(rotation=0)
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.warning(f"No data available for {metric} metric.")
                else:
                    st.warning(f"No data available for model {model}.")
        else:
            st.warning("No data available for selected models and categories.")
    else:
        st.info("Please select at least one model and category.")

# ---------------- Raw Data ----------------
elif page == "Raw Data":
    st.header("📄 Raw Data")
    
    # Add filters
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        model_filter = st.multiselect("Filter by Model", df["Model_Name"].unique(), default=df["Model_Name"].unique())
    with col2:
        category_filter = st.multiselect("Filter by Category", df["Category"].unique(), default=df["Category"].unique())
    
    # Apply filters
    filtered_df = df[(df["Model_Name"].isin(model_filter)) & (df["Category"].isin(category_filter))]
    
    st.dataframe(filtered_df, use_container_width=True)

    # CSV Download
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Filtered CSV",
        data=csv,
        file_name="llm_eval_results_filtered.csv",
        mime="text/csv"
    )
    
    # Show statistics
    st.subheader("Data Statistics")
    st.write(f"Showing {len(filtered_df)} rows out of {len(df)} total rows")
    
    if not filtered_df.empty:
        st.write("Numeric columns summary:")
        st.dataframe(filtered_df[METRICS + ["Latency_sec", "Tokens", "Tokens_per_sec"]].describe())