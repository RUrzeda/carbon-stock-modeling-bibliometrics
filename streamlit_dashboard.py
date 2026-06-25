#!/usr/bin/env python3
"""
Comprehensive Streamlit Dashboard for Bibliometric Analysis
Carbon Stock Modeling Research with Remote Sensing and AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from bibliometric_analysis import BibliometricAnalyzer
from classic_indicators import render_classic_indicators_tab, render_networks_tab, render_methodology_tab

st.set_page_config(
    page_title="Carbon Stock Modeling Research - Bibliometric Analysis",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E8B57;
    }
    .metric-card-label {
        font-size: 0.875rem;
        color: #555;
        text-align: center;
    }
    .metric-card-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #111;
        text-align: center;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4682B4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_base_data():
    """Load and perform initial cleaning on the data. This is cached."""
    analyzer = BibliometricAnalyzer('DF_COMBINADO_LIMPO.csv')
    base_df = analyzer.load_and_clean_data()
    return analyzer, base_df

def create_temporal_chart(temporal_df, selected_years):
    """Create temporal evolution chart"""
    filtered_df = temporal_df[temporal_df['Year'].isin(selected_years)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['Year'],
        y=filtered_df['Total'],
        mode='lines+markers',
        name='Total Publications',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df['Year'],
        y=filtered_df['Brazilian'],
        mode='lines+markers',
        name='Brazilian Studies',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df['Year'],
        y=filtered_df['International'],
        mode='lines+markers',
        name='International Studies',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Temporal Evolution of Publications',
        xaxis_title='Year',
        yaxis_title='Number of Publications',
        template='plotly_white',
        height=500
    )
    return fig

def create_ai_techniques_chart(ai_df, top_n):
    """Create AI techniques chart"""
    filtered_df = ai_df.head(top_n)
    
    fig = px.bar(
        filtered_df,
        x='Frequency',
        y='Technique',
        orientation='h',
        title=f'Top {top_n} AI Techniques in Carbon/Biomass Estimation',
        color='Frequency',
        color_continuous_scale='reds'
    )
    fig.update_layout(
        template='plotly_white',
        height=max(400, top_n * 30),
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig

def create_data_types_chart(data_df, top_n):
    """Create data types chart"""
    filtered_df = data_df.head(top_n)
    
    fig = px.bar(
        filtered_df,
        x='Frequency',
        y='Data_Type',
        orientation='h',
        title=f'Top {top_n} Data Types for Carbon/Biomass Estimation',
        color='Frequency',
        color_continuous_scale='greens'
    )
    fig.update_layout(
        template='plotly_white',
        height=max(400, top_n * 30),
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig

def create_trends_chart(trends_df, selected_items, chart_type):
    """Create trends chart"""
    fig = go.Figure()
    colors = px.colors.qualitative.Set3
    
    for i, item in enumerate(selected_items):
        if item in trends_df.columns:
            fig.add_trace(go.Scatter(
                x=trends_df['Year'],
                y=trends_df[item],
                mode='lines+markers',
                name=item,
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title=f'{chart_type} Trends Over Time',
        xaxis_title='Year',
        yaxis_title='Frequency',
        template='plotly_white',
        height=600
    )
    return fig

def create_geographic_chart(geo_df, top_n):
    """Create geographic distribution chart"""
    filtered_df = geo_df.head(top_n)
    
    fig = px.bar(
        filtered_df,
        x='Publications',
        y='Country',
        orientation='h',
        title=f'Top {top_n} Countries by Publication Count',
        color='Publications',
        color_continuous_scale='blues'
    )
    fig.update_layout(
        template='plotly_white',
        height=max(400, top_n * 25),
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig



def main():
    analyzer, base_df = load_base_data()

    st.sidebar.title("📊 Analysis Filters")

    min_year_data, max_year_data = int(base_df['PY'].min()), int(base_df['PY'].max())
    doc_types_available = sorted(base_df['DT'].dropna().unique().tolist())

    if st.sidebar.button("🔄 Reset filters", help="Reset year range, study type, document type and top-N to defaults"):
        for key in ("year_range", "study_type", "doc_types", "top_n"):
            st.session_state.pop(key, None)
        st.rerun()

    min_year, max_year = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year_data,
        max_value=max_year_data,
        value=(min_year_data, max_year_data),
        help="Filter data by publication year range",
        key="year_range",
    )

    study_type = st.sidebar.radio(
        "Select Study Type",
        ["All", "Brazilian", "International"],
        index=0,
        help="Filter by study origin",
        key="study_type",
    )

    selected_doc_types = st.sidebar.multiselect(
        "Document Type",
        options=doc_types_available,
        default=doc_types_available,
        help="Filter by document type (Article, Review, Conference Paper, etc.)",
        key="doc_types",
    )

    top_n = st.sidebar.slider(
        "Number of items to display",
        min_value=5, max_value=20, value=10,
        help="Select number of top items to display in charts",
        key="top_n",
    )

    df_filtered_by_user = base_df[base_df['PY'].between(min_year, max_year)]
    if study_type == "Brazilian":
        df_filtered_by_user = df_filtered_by_user[df_filtered_by_user['is_brazilian']]
    elif study_type == "International":
        df_filtered_by_user = df_filtered_by_user[~df_filtered_by_user['is_brazilian']]
    df_filtered_by_user = df_filtered_by_user[df_filtered_by_user['DT'].isin(selected_doc_types)]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{len(df_filtered_by_user):,} documents** match these filters")

    if not df_filtered_by_user.empty:
        results = analyzer.run_analysis(df_filtered_by_user.copy())
    else:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        return

    st.markdown('<h1 class="main-header">🌳 Carbon Stock Modeling Research</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Bibliometric Analysis: Multisensor Modeling with Remote Sensing and AI</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📈 Overview", "👥 Authors & Journals", "🔬 AI Techniques",
        "📡 Data Types", "🚁 Drone Analysis", "🗺️ Geographic Distribution", "🌱 Area Focus",
        "📚 Classic Indicators", "🕸️ Networks", "🧪 Methodology"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">📊 Research Overview</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        metric_cards = [
            ("Total Studies", f"{len(df_filtered_by_user):,}"),
            ("Brazilian Studies", f"{df_filtered_by_user['is_brazilian'].sum():,}"),
            ("International Studies", f"{(~df_filtered_by_user['is_brazilian']).sum():,}"),
            ("Drone/UAV Studies", f"{results['drone_count']:,}"),
        ]
        for col, (label, value) in zip((col1, col2, col3, col4), metric_cards):
            with col:
                # Single markdown call: a div opened/closed across separate
                # st.markdown calls never actually wraps st.metric (each is a
                # sibling block), leaving an empty styled rectangle behind.
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-card-label">{label}</div>'
                    f'<div class="metric-card-value">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        
        st.plotly_chart(create_temporal_chart(results['temporal_df'], list(range(min_year, max_year + 1))), use_container_width=True)
        st.markdown("<h2 class='sub-header'>☁️ Word Cloud & Key Terms</h2>", unsafe_allow_html=True)
        wordcloud_fig = results.get('wordcloud_fig')
        if wordcloud_fig:
            col1, col2, col3 = st.columns([0.2, 1, 0.2])
            with col2:
                st.pyplot(wordcloud_fig)
        else:
            st.warning("No keyword data available for the selected period.")
        st.markdown('<div class="insight-box" style="margin-top: 2rem;">', unsafe_allow_html=True)
        st.markdown("**🔑 Featured Terms in the Cloud:**")
        top_words_df = results.get('top_words_df')
        if top_words_df is not None and not top_words_df.empty:
            for i, row in top_words_df.head(3).iterrows():
                st.markdown(f"• **{row['Term'].title()}**: {row['Frequency']} mentions")
            with st.expander("See the list of the 20 most frequent terms"):
                st.dataframe(top_words_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab2:
        st.markdown('<h2 class="sub-header">👥 Authors & Journals Analysis</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Top Authors")
            fig_authors = px.bar(results['top_authors'].head(top_n), x='Publications', y='Author', orientation='h', color='Publications', color_continuous_scale='viridis')
            fig_authors.update_layout(height=max(400, top_n * 25), yaxis={'categoryorder': 'total ascending'}, template='plotly_white')
            st.plotly_chart(fig_authors, use_container_width=True)
            if 'top_brazil_authors' in results and not results['top_brazil_authors'].empty:
                st.markdown("### Top Brazilian Authors")
                fig_brazil_authors = px.bar(results['top_brazil_authors'].head(10), x='Publications', y='Author', orientation='h', color='Publications', color_continuous_scale='oranges')
                fig_brazil_authors.update_layout(height=300, yaxis={'categoryorder': 'total ascending'}, template='plotly_white')
                st.plotly_chart(fig_brazil_authors, use_container_width=True)
        with col2:
            st.markdown("### Top Journals")
            fig_journals = px.bar(results['top_journals'].head(top_n), x='Publications', y='Journal', orientation='h', color='Publications', color_continuous_scale='plasma')
            fig_journals.update_layout(height=max(400, top_n * 25), yaxis={'categoryorder': 'total ascending'}, template='plotly_white')
            st.plotly_chart(fig_journals, use_container_width=True)

        st.markdown("---")
        st.markdown("### Most Cited Documents")
        most_cited_df = results.get('most_cited_df')
        if most_cited_df is not None and not most_cited_df.empty:
            fig_cited = px.bar(
                most_cited_df.head(top_n), x='TC', y='TI_short', orientation='h',
                color='TC', color_continuous_scale='magma', hover_name='TI',
                labels={'TC': 'Citations', 'TI_short': ''},
            )
            fig_cited.update_layout(
                height=max(400, top_n * 30), yaxis={'categoryorder': 'total ascending'},
                template='plotly_white', xaxis_title='Citations', yaxis_title='',
            )
            st.plotly_chart(fig_cited, use_container_width=True)
        else:
            st.warning("No citation data available for the selected filters.")

        st.markdown("---")
        st.markdown("### Most Cited Brazilian Documents")
        most_cited_brazil_df = results.get('most_cited_brazil_df')
        if most_cited_brazil_df is not None and not most_cited_brazil_df.empty:
            fig_cited_br = px.bar(
                most_cited_brazil_df.head(top_n), x='TC', y='TI_short', orientation='h',
                color='TC', color_continuous_scale='oranges', hover_name='TI',
                labels={'TC': 'Citations', 'TI_short': ''},
            )
            fig_cited_br.update_layout(
                height=max(400, top_n * 30), yaxis={'categoryorder': 'total ascending'},
                template='plotly_white', xaxis_title='Citations', yaxis_title='',
            )
            st.plotly_chart(fig_cited_br, use_container_width=True)
        else:
            st.warning("No Brazilian citation data available for the selected filters.")

        st.markdown("---")
        st.markdown("### Most Frequent Keywords")
        st.dataframe(results['top_keywords'].head(20), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("<h2 class='sub-header'>🤖 Most Used AI Techniques</h2>", unsafe_allow_html=True)
        ai_techniques_df = results.get('ai_techniques_df')
        if ai_techniques_df is not None and not ai_techniques_df.empty:
            fig_ai_freq = px.bar(ai_techniques_df.head(top_n), x='Count', y='Technique', orientation='h', title='Frequency of AI Techniques', color='Count', color_continuous_scale=px.colors.sequential.Reds)
            fig_ai_freq.update_layout(template='plotly_white', height=max(400, top_n * 30), yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_ai_freq, use_container_width=True)
        else:
            st.warning("No AI technique data available.")
        st.markdown("---")
        st.markdown("<h2 class='sub-header'>📈 Trends of Top 7 AI Techniques</h2>", unsafe_allow_html=True)
        ai_trends_df = results.get('ai_trends_df')
        if ai_trends_df is not None and not ai_trends_df.empty:
            fig_ai_trends = px.line(ai_trends_df, x='PY', y='Count', color='AI_Techniques', title='Annual Usage of Top 7 AI Techniques', markers=True, labels={'PY': 'Year', 'Count': 'Publications', 'AI_Techniques': 'Technique'})
            fig_ai_trends.update_layout(template='plotly_white', height=500, legend_title_text='Technique')
            st.plotly_chart(fig_ai_trends, use_container_width=True)
        else:
            st.warning("No AI technique trend data available.")

        if ai_techniques_df is not None and not ai_techniques_df.empty:
            st.markdown('<div class="insight-box" style="margin-top: 2rem;">', unsafe_allow_html=True)
            st.markdown("**🤖 AI Techniques Insights:**")
            top_technique = ai_techniques_df.iloc[0]
            st.markdown(f"• **Dominant Method**: {top_technique['Technique']} is the most applied, featured in {top_technique['Count']} studies.")
            if len(ai_techniques_df) > 1:
                second_technique = ai_techniques_df.iloc[1]
                st.markdown(f"• **Key Contender**: {second_technique['Technique']} follows as a popular alternative.")
            if ai_trends_df is not None and not ai_trends_df.empty:
                latest_year = ai_trends_df['PY'].max()
                latest_trends = ai_trends_df[ai_trends_df['PY'] == latest_year]
                if not latest_trends.empty:
                    top_recent_technique = latest_trends.loc[latest_trends['Count'].idxmax()]
                    st.markdown(f"• **Recent Trend**: **{top_recent_technique['AI_Techniques']}** shows strong recent usage, peaking in {int(latest_year)}.")
            st.markdown('</div>', unsafe_allow_html=True)
            
    with tab4:
        st.markdown('<h2 class="sub-header">📡 Data Types Analysis</h2>', unsafe_allow_html=True)
        st.plotly_chart(create_data_types_chart(results['data_types_df'], top_n), use_container_width=True)
        st.markdown("### Data Types Trends Over Time")
        available_data_types = results['data_types_df']['Data_Type'].tolist()
        selected_data_types = st.multiselect("Select data types to compare trends:", available_data_types, default=available_data_types[:5])
        if selected_data_types:
            st.plotly_chart(create_trends_chart(results['data_trends_df'], selected_data_types, "Data Types"), use_container_width=True)
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**🔍 Data Types Insights:**")
        top_3_data = results['data_types_df'].head(3)
        for i, row in top_3_data.iterrows():
            st.markdown(f"• **{row['Data_Type']}**: {row['Frequency']} studies")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<h2 class="sub-header">🚁 Drone/UAV Analysis</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            drone_years = list(range(min_year, max_year + 1))
            drone_counts = [results['drone_by_year'].get(year, 0) for year in drone_years]
            fig_drone = go.Figure(go.Scatter(x=drone_years, y=drone_counts, mode='lines+markers', name='Drone/UAV Studies', line=dict(color='#ff6b6b', width=4), marker=dict(size=10), fill='tonexty'))
            fig_drone.update_layout(title='Drone/UAV Usage Over Time', xaxis_title='Year', yaxis_title='Number of Studies', template='plotly_white', height=400)
            st.plotly_chart(fig_drone, use_container_width=True)
        with col2:
            st.plotly_chart(px.bar(results['promising_drone_data'].head(top_n), x='Frequency', y='Data_Type', orientation='h', title='Most Promising Drone Data Types', color='Frequency', color_continuous_scale='oranges').update_layout(height=400, yaxis={'categoryorder': 'total ascending'}, template='plotly_white'), use_container_width=True)
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**🔍 Drone/UAV Insights:**")
        st.markdown(f"• {results['drone_count']} studies ({(results['drone_count']/len(df_filtered_by_user)*100 if len(df_filtered_by_user) > 0 else 0):.1f}%) use drone/UAV technology")
        if drone_counts:
            st.markdown(f"• Peak drone usage: {max(drone_counts)} studies in {drone_years[drone_counts.index(max(drone_counts))]}")
        if 'promising_drone_data' in results and not results['promising_drone_data'].empty:
            top_drone_data = results['promising_drone_data'].iloc[0]
            st.markdown(f"• Most promising drone data: **{top_drone_data['Data_Type']}** ({top_drone_data['Frequency']} studies)")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab6:
        st.markdown('<h2 class="sub-header">🗺️ Geographic Distribution</h2>', unsafe_allow_html=True)
        st.plotly_chart(create_geographic_chart(results['geographic_df'], top_n), use_container_width=True)

        # A pie chart makes a ~96/4 split nearly impossible to read (Brazil's
        # slice all but disappears). A horizontal bar with the percentage as
        # the bar label stays legible at any ratio.
        total_n = len(df_filtered_by_user)
        brazil_n = int(df_filtered_by_user['is_brazilian'].sum())
        intl_n = total_n - brazil_n
        brazil_pct = round(100 * brazil_n / total_n, 1) if total_n else 0.0
        intl_pct = round(100 * intl_n / total_n, 1) if total_n else 0.0
        comparison_df = pd.DataFrame({
            'Study Type': ['Brazilian', 'International'],
            'Count': [brazil_n, intl_n],
            'Label': [f"{brazil_n:,} ({brazil_pct}%)", f"{intl_n:,} ({intl_pct}%)"],
        })
        fig_comparison = px.bar(
            comparison_df, x='Count', y='Study Type', orientation='h', text='Label',
            color='Study Type', color_discrete_sequence=['#ff7f0e', '#1f77b4'],
        )
        fig_comparison.update_traces(textposition='outside')
        fig_comparison.update_layout(
            title='Brazilian vs International Studies', template='plotly_white',
            height=300, showlegend=False, xaxis_title='Number of studies', yaxis_title='',
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**🔍 Geographic Insights:**")
        top_3_countries = results['geographic_df'].head(3)
        for i, row in top_3_countries.iterrows():
            st.markdown(f"• **{row['Country']}**: {row['Publications']} publications")
        if 'BRAZIL' in results['geographic_df']['Country'].values:
            brazil_rank = results['geographic_df'][results['geographic_df']['Country'] == 'BRAZIL'].index[0] + 1
            st.markdown(f"• Brazil ranks #{brazil_rank} globally in this selection.")
        st.markdown(f"• Brazil represents **{brazil_pct}%** of studies in this selection "
                    f"({brazil_n:,} of {total_n:,}).")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab7:
        st.markdown("<h2 class='sub-header'>🌱 Area Focus Analysis</h2>", unsafe_allow_html=True)
        st.markdown("This section explores which technologies are applied to different environmental study areas.")
        st.markdown("---")
        st.markdown("<h3 class='sub-header' style='font-size:1.3rem;'>🚁 Drone Applications by Study Area</h3>", unsafe_allow_html=True)
        drone_areas_recent = results.get('drone_areas_recent')
        if drone_areas_recent is not None and not drone_areas_recent.empty:
            st.plotly_chart(px.bar(drone_areas_recent, x='Publications', y='Study Area', orientation='h', title='Drone Usage Across Study Areas in the Last 5 Years', color='Publications', color_continuous_scale=px.colors.sequential.Teal).update_layout(template='plotly_white', yaxis={'categoryorder': 'total ascending'}, height=500), use_container_width=True)
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**🚁 Drone Application Insights:**")
            top_area = drone_areas_recent.iloc[0]
            st.markdown(f"• **Top Application**: **{top_area['Study Area']}** is the leading area for drone research, with {top_area['Publications']} publications in the last 5 years.")
            if len(drone_areas_recent) > 1:
                second_area = drone_areas_recent.iloc[1]
                st.markdown(f"• **Runner-Up**: Research in **{second_area['Study Area']}** follows closely.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No data on recent drone study areas available.")
        
        drone_areas_evolution = results.get('drone_areas_evolution')
        if drone_areas_evolution is not None and not drone_areas_evolution.empty:
            st.plotly_chart(px.line(drone_areas_evolution, x='PY', y='Publications', color='Study Area', title='Evolution of Study Areas in Drone-Based Research', markers=True, labels={'PY': 'Publication Year', 'Publications': 'Number of Publications', 'Study Area': 'Study Area'}).update_layout(template='plotly_white', height=500), use_container_width=True)
        else:
            st.warning("No data on the evolution of drone study areas available.")

        st.markdown("---")
        st.markdown("<h3 class='sub-header' style='font-size:1.3rem;'>🔗 Co-occurrence Heatmaps</h3>", unsafe_allow_html=True)
        cooc_ia_area = results.get('cooc_ia_area')
        if cooc_ia_area is not None and not cooc_ia_area.empty:
            st.plotly_chart(px.imshow(cooc_ia_area, text_auto=True, aspect="auto", color_continuous_scale='Blues', labels=dict(x="Study Area", y="AI Technique", color="Publications"), title="Heatmap of AI Techniques Applied in Each Study Area").update_layout(height=600), use_container_width=True)
        else:
            st.warning("No co-occurrence data available for AI Techniques and Study Areas.")
        
        cooc_data_area = results.get('cooc_data_area')
        if cooc_data_area is not None and not cooc_data_area.empty:
            st.plotly_chart(px.imshow(cooc_data_area, text_auto=True, aspect="auto", color_continuous_scale='Greens', labels=dict(x="Study Area", y="Data Type", color="Publications"), title="Heatmap of Data Types Used in Each Study Area").update_layout(height=500), use_container_width=True)
        else:
            st.warning("No co-occurrence data available for Data Types and Study Areas.")
            
    with tab8:
        render_classic_indicators_tab()

    with tab9:
        render_networks_tab()

    with tab10:
        render_methodology_tab()

    # Footer
    st.markdown("---")
    st.markdown("**📊 Dashboard created for comprehensive bibliometric analysis of carbon stock modeling research (2015-2025)**")
    st.markdown("*Focus: Multisensor modeling with remote sensing and artificial intelligence*")

if __name__ == "__main__":
    main()