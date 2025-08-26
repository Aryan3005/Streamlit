import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
import os
import numpy as np

# Configure page settings
st.set_page_config(
    layout="wide", 
    page_title="LLM Evaluation Dashboard",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.3rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #3498db;
    }
    
    .section-header h2 {
        color: #2c3e50;
        margin: 0;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #5a6fd8, #6a4190);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(45deg, #667eea, #764ba2);
    }
    
    /* Success/Error message styling */
    .success-message {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .error-message {
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

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
            st.warning(f"⚠️ Error loading local file: {e}")
    
    # Use requests for remote URL (handles SSL better)
    try:
        with st.spinner("🔄 Loading data from Google Drive..."):
            response = requests.get(drive_url, timeout=30)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
    except requests.RequestException as e:
        st.error(f"❌ Error downloading CSV: {e}")
        st.error("Please check your internet connection or file permissions.")
        return None
    except Exception as e:
        st.error(f"❌ Error processing CSV data: {e}")
        return None

# Load the data
df = load_data()

if df is None:
    st.markdown('<div class="error-message">❌ Failed to load data. Please check the file source.</div>', unsafe_allow_html=True)
    st.stop()

# Data Processing
try:
    required_columns = [
        "Prompt_ID", "Prompt_Text_x", "Output", "Output_ref1", "Output_ref2",
        "Category", "Latency_sec", "Tokens", "Tokens_per_sec",
        "Embedding_avg", "BLEU_avg", "ROUGE1_avg", "ROUGEL_avg", "G-Eval Avg", "Model_Name"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"❌ Missing columns in dataset: {missing_columns}")
        st.info("Available columns: " + ", ".join(df.columns.tolist()))
        st.stop()
    
    df = df[required_columns].rename(columns={
        "Prompt_Text_x": "Prompt_Text",
        "G-Eval Avg": "G_Eval_avg"
    })
    
    df["G_Eval_avg"] = df["G_Eval_avg"] / 10.0

except Exception as e:
    st.error(f"❌ Error processing data: {e}")
    st.stop()

METRICS = ["Embedding_avg", "BLEU_avg", "ROUGE1_avg", "ROUGEL_avg", "G_Eval_avg"]
METRIC_LABELS = {
    "Embedding_avg": "Embedding Similarity",
    "BLEU_avg": "BLEU Score",
    "ROUGE1_avg": "ROUGE-1 Score",
    "ROUGEL_avg": "ROUGE-L Score",
    "G_Eval_avg": "G-Eval Score"
}

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 LLM Evaluation Dashboard</h1>
    <p>Comprehensive analysis and comparison of Large Language Model performance metrics</p>
</div>
""", unsafe_allow_html=True)

# Key Metrics Overview
st.markdown('<div class="section-header"><h2>📊 Dataset Overview</h2></div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Total Prompts</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df["Model_Name"].nunique()}</div>
        <div class="metric-label">Models</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df["Category"].nunique()}</div>
        <div class="metric-label">Categories</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df['Latency_sec'].mean():.2f}s</div>
        <div class="metric-label">Avg Latency</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df['Tokens_per_sec'].mean():.0f}</div>
        <div class="metric-label">Tokens/Sec</div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🎛️ Navigation")
    page = st.radio(
        "Select Analysis View",
        ["🏆 Leaderboard", "🔍 Model Comparison", "🔥 Performance Heatmaps", "📊 Advanced Analytics", "📄 Data Explorer"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🎨 Theme Options")
    color_theme = st.selectbox("Chart Color Theme", ["viridis", "plasma", "inferno", "magma", "cividis"])

# Leaderboard Page
if page == "🏆 Leaderboard":
    st.markdown('<div class="section-header"><h2>🏆 Model Performance Leaderboard</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        metric_to_sort = st.selectbox(
            "🎯 Ranking Metric",
            options=METRICS,
            index=1,
            format_func=lambda m: METRIC_LABELS[m]
        )
    
    with col2:
        show_all_metrics = st.checkbox("Show All Metrics", value=True)

    leaderboard = df.groupby("Model_Name")[METRICS].mean().sort_values(metric_to_sort, ascending=False)
    
    tab1, tab2, tab3 = st.tabs(["📋 Rankings Table", "📊 Performance Chart", "🎯 Radar Comparison"])
    
    with tab1:
        if show_all_metrics:
            # Style the dataframe
            styled_df = leaderboard.style.format("{:.4f}").background_gradient(
                cmap='RdYlGn', subset=METRICS, vmin=0, vmax=1
            ).set_properties(**{
                'font-family': 'Inter, sans-serif',
                'font-size': '12px'
            })
            st.dataframe(styled_df, use_container_width=True, height=400)
        else:
            single_metric_df = leaderboard[[metric_to_sort]].style.format("{:.4f}").background_gradient(
                cmap='RdYlGn', vmin=0, vmax=1
            )
            st.dataframe(single_metric_df, use_container_width=True, height=400)
    
    with tab2:
        fig = px.bar(
            x=leaderboard[metric_to_sort].values,
            y=leaderboard.index,
            orientation='h',
            title=f"Model Performance: {METRIC_LABELS[metric_to_sort]}",
            labels={'x': METRIC_LABELS[metric_to_sort], 'y': 'Model'},
            color=leaderboard[metric_to_sort].values,
            color_continuous_scale=color_theme
        )
        fig.update_layout(
            height=600,
            showlegend=False,
            title_font_size=16,
            font_family="Inter"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Radar chart for top 5 models
        top_5_models = leaderboard.head(5)
        
        fig = go.Figure()
        
        for model in top_5_models.index:
            fig.add_trace(go.Scatterpolar(
                r=top_5_models.loc[model, METRICS].values,
                theta=[METRIC_LABELS[m] for m in METRICS],
                fill='toself',
                name=model,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Top 5 Models - Multi-Metric Comparison",
            height=600,
            font_family="Inter"
        )
        st.plotly_chart(fig, use_container_width=True)

# Model Comparison Page
elif page == "🔍 Model Comparison":
    st.markdown('<div class="section-header"><h2>🔍 Advanced Model Comparison</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_models = st.multiselect(
            "🤖 Select Models",
            df["Model_Name"].unique(),
            default=df["Model_Name"].unique()[:3] if len(df["Model_Name"].unique()) >= 3 else df["Model_Name"].unique()
        )
    
    with col2:
        selected_categories = st.multiselect(
            "📂 Select Categories",
            df["Category"].unique(),
            default=df["Category"].unique()[:5] if len(df["Category"].unique()) >= 5 else df["Category"].unique()
        )
    
    with col3:
        selected_metrics = st.multiselect(
            "📏 Select Metrics",
            METRICS,
            default=METRICS,
            format_func=lambda m: METRIC_LABELS[m]
        )

    if selected_models and selected_categories and selected_metrics:
        df_comp = df[
            (df["Model_Name"].isin(selected_models)) & 
            (df["Category"].isin(selected_categories))
        ]
        
        if not df_comp.empty:
            # Create subplots for each metric
            for metric in selected_metrics:
                with st.expander(f"📊 {METRIC_LABELS[metric]} Analysis", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        heatmap_data = df_comp.pivot_table(
                            index="Category",
                            columns="Model_Name",
                            values=metric,
                            aggfunc="mean"
                        ).clip(0, 1)
                        
                        if not heatmap_data.empty:
                            fig = px.imshow(
                                heatmap_data,
                                title=f"{METRIC_LABELS[metric]} Heatmap",
                                color_continuous_scale=color_theme,
                                aspect="auto"
                            )
                            fig.update_layout(height=400, font_family="Inter")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Summary statistics
                        summary_stats = df_comp.groupby("Model_Name")[metric].agg(['mean', 'std', 'min', 'max']).round(4)
                        st.subheader("📈 Statistics")
                        st.dataframe(summary_stats, use_container_width=True)
        else:
            st.warning("⚠️ No data available for selected combination.")

# Performance Heatmaps Page
elif page == "🔥 Performance Heatmaps":
    st.markdown('<div class="section-header"><h2>🔥 Detailed Performance Heatmaps</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_models = st.multiselect(
            "🤖 Select Models",
            df["Model_Name"].unique(),
            default=df["Model_Name"].unique()[:2]
        )
    
    with col2:
        selected_categories = st.multiselect(
            "📂 Select Categories",
            df["Category"].unique(),
            default=df["Category"].unique()[:6] if len(df["Category"].unique()) >= 6 else df["Category"].unique()
        )

    if selected_models and selected_categories:
        df_heat = df[
            (df["Model_Name"].isin(selected_models)) & 
            (df["Category"].isin(selected_categories))
        ]
        
        if not df_heat.empty:
            for model in selected_models:
                st.markdown(f"### 🤖 {model}")
                df_model = df_heat[df_heat["Model_Name"] == model]
                
                if not df_model.empty:
                    # Create a comprehensive heatmap for all metrics
                    heatmap_data = []
                    for metric in METRICS:
                        metric_data = df_model.pivot_table(
                            index="Category",
                            values=metric,
                            aggfunc="mean"
                        )
                        heatmap_data.append(metric_data)
                    
                    combined_data = pd.concat(heatmap_data, axis=1, keys=[METRIC_LABELS[m] for m in METRICS])
                    
                    if not combined_data.empty:
                        fig = px.imshow(
                            combined_data.T,
                            title=f"Complete Performance Profile: {model}",
                            color_continuous_scale=color_theme,
                            aspect="auto"
                        )
                        fig.update_layout(height=500, font_family="Inter")
                        st.plotly_chart(fig, use_container_width=True)

# Advanced Analytics Page
elif page == "📊 Advanced Analytics":
    st.markdown('<div class="section-header"><h2>📊 Advanced Performance Analytics</h2></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔄 Correlation Analysis", "📈 Distribution Analysis", "⚡ Performance vs Speed"])
    
    with tab1:
        st.subheader("🔄 Metric Correlation Matrix")
        correlation_matrix = df[METRICS].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Metric Correlation Heatmap",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1
        )
        fig.update_layout(height=500, font_family="Inter")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Insights:**")
        high_corr_pairs = []
        for i in range(len(METRICS)):
            for j in range(i+1, len(METRICS)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append(f"• {METRIC_LABELS[METRICS[i]]} ↔ {METRIC_LABELS[METRICS[j]]}: {corr_val:.3f}")
        
        if high_corr_pairs:
            st.markdown("High correlations (|r| > 0.7):")
            for pair in high_corr_pairs:
                st.markdown(pair)
        else:
            st.markdown("No strong correlations (|r| > 0.7) found between metrics.")
    
    with tab2:
        st.subheader("📈 Performance Distribution Analysis")
        
        selected_metric = st.selectbox(
            "Select Metric for Distribution Analysis",
            METRICS,
            format_func=lambda m: METRIC_LABELS[m]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot
            fig = px.box(
                df.melt(id_vars=['Model_Name'], value_vars=[selected_metric], var_name='Metric', value_name='Score'),
                x='Model_Name',
                y='Score',
                title=f"{METRIC_LABELS[selected_metric]} Distribution by Model",
                color='Model_Name'
            )
            fig.update_layout(showlegend=False, font_family="Inter")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Violin plot
            fig = px.violin(
                df.melt(id_vars=['Model_Name'], value_vars=[selected_metric], var_name='Metric', value_name='Score'),
                x='Model_Name',
                y='Score',
                title=f"{METRIC_LABELS[selected_metric]} Density Distribution",
                color='Model_Name'
            )
            fig.update_layout(showlegend=False, font_family="Inter")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("⚡ Performance vs Speed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            quality_metric = st.selectbox(
                "Quality Metric (Y-axis)",
                METRICS,
                index=1,
                format_func=lambda m: METRIC_LABELS[m]
            )
        
        with col2:
            speed_metric = st.selectbox(
                "Speed Metric (X-axis)",
                ["Latency_sec", "Tokens_per_sec"],
                format_func=lambda m: "Latency (seconds)" if m == "Latency_sec" else "Tokens per Second"
            )
        
        # Scatter plot with model performance vs speed
        avg_performance = df.groupby("Model_Name").agg({
            quality_metric: 'mean',
            speed_metric: 'mean',
            'Tokens': 'mean'
        }).reset_index()
        
        fig = px.scatter(
            avg_performance,
            x=speed_metric,
            y=quality_metric,
            size='Tokens',
            hover_name='Model_Name',
            title=f"{METRIC_LABELS[quality_metric]} vs {speed_metric.replace('_', ' ').title()}",
            color=quality_metric,
            color_continuous_scale=color_theme
        )
        fig.update_layout(height=500, font_family="Inter")
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance efficiency score
        st.subheader("🏅 Efficiency Rankings")
        if speed_metric == "Latency_sec":
            avg_performance['efficiency'] = avg_performance[quality_metric] / (avg_performance[speed_metric] + 1e-6)
            efficiency_label = f"{METRIC_LABELS[quality_metric]} per Second"
        else:
            avg_performance['efficiency'] = avg_performance[quality_metric] * avg_performance[speed_metric]
            efficiency_label = f"Quality × Speed Score"
        
        efficiency_ranking = avg_performance.sort_values('efficiency', ascending=False)[['Model_Name', 'efficiency']]
        efficiency_ranking.columns = ['Model', efficiency_label]
        st.dataframe(efficiency_ranking.style.format({efficiency_label: "{:.4f}"}), use_container_width=True)

# Data Explorer Page
elif page == "📄 Data Explorer":
    st.markdown('<div class="section-header"><h2>📄 Interactive Data Explorer</h2></div>', unsafe_allow_html=True)
    
    # Advanced Filters
    st.subheader("🔍 Data Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_filter = st.multiselect(
            "🤖 Filter by Model",
            df["Model_Name"].unique(),
            default=df["Model_Name"].unique()
        )
    
    with col2:
        category_filter = st.multiselect(
            "📂 Filter by Category",
            df["Category"].unique(),
            default=df["Category"].unique()
        )
    
    with col3:
        # Performance threshold filter
        quality_threshold = st.slider(
            "📊 Minimum BLEU Score",
            min_value=float(df["BLEU_avg"].min()),
            max_value=float(df["BLEU_avg"].max()),
            value=float(df["BLEU_avg"].min()),
            step=0.01
        )
    
    # Apply filters
    filtered_df = df[
        (df["Model_Name"].isin(model_filter)) & 
        (df["Category"].isin(category_filter)) &
        (df["BLEU_avg"] >= quality_threshold)
    ]
    
    # Display filtered results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 Filtered Results ({len(filtered_df):,} rows)")
        st.dataframe(
            filtered_df.style.format({
                col: "{:.4f}" for col in METRICS
            }).format({
                "Latency_sec": "{:.3f}",
                "Tokens_per_sec": "{:.1f}"
            }),
            use_container_width=True,
            height=500
        )
    
    with col2:
        st.subheader("📈 Summary Statistics")
        if not filtered_df.empty:
            summary_stats = filtered_df[METRICS + ["Latency_sec", "Tokens_per_sec"]].describe().round(4)
            st.dataframe(summary_stats, use_container_width=True)
            
            # Download options
            st.subheader("⬇️ Export Options")
            
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"llm_eval_filtered_{len(filtered_df)}_rows.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            json_data = filtered_df.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"llm_eval_filtered_{len(filtered_df)}_rows.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.warning("⚠️ No data matches the current filters.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #7f8c8d; padding: 2rem;">
        <p>🚀 <strong>LLM Evaluation Dashboard</strong> | Built with Streamlit & Plotly</p>
        <p>📊 Analyzing performance across {models} models and {categories} categories</p>
    </div>
    """.format(
        models=df["Model_Name"].nunique(),
        categories=df["Category"].nunique()
    ),
    unsafe_allow_html=True
)