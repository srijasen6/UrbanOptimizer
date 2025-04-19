import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def render_dashboard(metrics):
    """
    Render a dashboard with city metrics using Plotly
    
    Args:
        metrics (dict): Dictionary containing metrics time series
    """
    st.subheader("City Performance Metrics")
    
    # Create summary KPIs if we have data
    if any(len(v) > 0 for v in metrics.values()):
        render_kpi_summary(metrics)
    
    # Create metric visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Traffic metrics
        st.write("##### Traffic Performance")
        if len(metrics.get('avg_wait_time', [])) > 0:
            fig = create_traffic_metrics_chart(metrics)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the traffic simulation to generate traffic metrics")
    
    with col2:
        # Energy metrics
        st.write("##### Energy Grid Performance")
        if len(metrics.get('energy_consumption', [])) > 0:
            fig = create_energy_metrics_chart(metrics)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the energy grid simulation to generate energy metrics")
    
    # Combined metrics over time if we have enough data
    if any(len(v) >= 5 for v in metrics.values()):
        st.write("##### City Performance Over Time")
        fig = create_combined_metrics_chart(metrics)
        st.plotly_chart(fig, use_container_width=True)
    
    # Optimization impact if we have enough data
    st.write("##### Optimization Impact Analysis")
    if any(len(v) >= 10 for v in metrics.values()):
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_optimization_comparison(metrics)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_correlation_heatmap(metrics)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Continue running the simulation to generate optimization impact data")

def render_kpi_summary(metrics):
    """Render KPI summary cards"""
    cols = st.columns(4)
    
    # Traffic wait time
    if len(metrics.get('avg_wait_time', [])) > 0:
        avg_wait = metrics['avg_wait_time'][-1]
        wait_delta = avg_wait - metrics['avg_wait_time'][0] if len(metrics['avg_wait_time']) > 1 else 0
        cols[0].metric("Average Wait Time", f"{avg_wait:.2f} s", f"{-wait_delta:.2f} s")
    else:
        cols[0].metric("Average Wait Time", "No data", "")
    
    # Traffic throughput
    if len(metrics.get('throughput', [])) > 0:
        throughput = metrics['throughput'][-1]
        throughput_delta = throughput - metrics['throughput'][0] if len(metrics['throughput']) > 1 else 0
        cols[1].metric("Traffic Throughput", f"{throughput:.0f} vehicles", f"{throughput_delta:.0f}")
    else:
        cols[1].metric("Traffic Throughput", "No data", "")
    
    # Energy consumption
    if len(metrics.get('energy_consumption', [])) > 0:
        consumption = metrics['energy_consumption'][-1]
        consumption_delta = consumption - metrics['energy_consumption'][0] if len(metrics['energy_consumption']) > 1 else 0
        cols[2].metric("Energy Consumption", f"{consumption:.1f} MW", f"{-consumption_delta:.1f} MW")
    else:
        cols[2].metric("Energy Consumption", "No data", "")
    
    # Grid stability
    if len(metrics.get('grid_stability', [])) > 0:
        stability = metrics['grid_stability'][-1] * 100
        stability_delta = (metrics['grid_stability'][-1] - metrics['grid_stability'][0]) * 100 if len(metrics['grid_stability']) > 1 else 0
        cols[3].metric("Grid Stability", f"{stability:.1f}%", f"{stability_delta:.1f}%")
    else:
        cols[3].metric("Grid Stability", "No data", "")

def create_traffic_metrics_chart(metrics):
    """Create a chart showing traffic metrics over time"""
    # Create a subplot with 2 y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add wait time trace
    if len(metrics.get('avg_wait_time', [])) > 0:
        x = list(range(len(metrics['avg_wait_time'])))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=metrics['avg_wait_time'],
                name="Average Wait Time",
                line=dict(color='red', width=2)
            ),
            secondary_y=False
        )
    
    # Add throughput trace
    if len(metrics.get('throughput', [])) > 0:
        x = list(range(len(metrics['throughput'])))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=metrics['throughput'],
                name="Throughput",
                line=dict(color='green', width=2)
            ),
            secondary_y=True
        )
    
    # Configure axes
    fig.update_layout(
        title_text="Traffic Flow Metrics",
        xaxis_title="Simulation Step",
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="Avg Wait Time (s)", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Throughput (vehicles)", secondary_y=True, showgrid=False)
    
    return fig

def create_energy_metrics_chart(metrics):
    """Create a chart showing energy metrics over time"""
    # Create a subplot with 2 y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add consumption trace
    if len(metrics.get('energy_consumption', [])) > 0:
        x = list(range(len(metrics['energy_consumption'])))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=metrics['energy_consumption'],
                name="Energy Consumption",
                line=dict(color='orange', width=2)
            ),
            secondary_y=False
        )
    
    # Add grid stability trace
    if len(metrics.get('grid_stability', [])) > 0:
        x = list(range(len(metrics['grid_stability'])))
        # Convert to percentage
        stability_pct = [val * 100 for val in metrics['grid_stability']]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=stability_pct,
                name="Grid Stability",
                line=dict(color='blue', width=2)
            ),
            secondary_y=True
        )
    
    # Configure axes
    fig.update_layout(
        title_text="Energy Grid Metrics",
        xaxis_title="Simulation Step",
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="Energy Consumption (MW)", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Grid Stability (%)", secondary_y=True, showgrid=False, range=[0, 100])
    
    return fig

def create_combined_metrics_chart(metrics):
    """Create a chart showing all metrics over time"""
    # Normalize all metrics to 0-100 scale for comparison
    normalized_metrics = {}
    
    for key, values in metrics.items():
        if len(values) > 0:
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:
                normalized = [(v - min_val) / (max_val - min_val) * 100 for v in values]
            else:
                normalized = [50 for _ in values]  # Default if all values are the same
            normalized_metrics[key] = normalized
    
    # Create the figure
    fig = go.Figure()
    
    # Add each metric
    colors = {
        'avg_wait_time': 'red',
        'throughput': 'green',
        'energy_consumption': 'orange',
        'grid_stability': 'blue'
    }
    
    labels = {
        'avg_wait_time': 'Average Wait Time',
        'throughput': 'Traffic Throughput',
        'energy_consumption': 'Energy Consumption',
        'grid_stability': 'Grid Stability'
    }
    
    for key, values in normalized_metrics.items():
        x = list(range(len(values)))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                name=labels.get(key, key),
                line=dict(color=colors.get(key, 'gray'), width=2)
            )
        )
    
    # Configure layout
    fig.update_layout(
        title_text="Normalized City Metrics Over Time",
        xaxis_title="Simulation Step",
        yaxis_title="Normalized Value (0-100)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_optimization_comparison(metrics):
    """Create a visualization comparing before and after optimization"""
    # Calculate before/after values
    before_after = {}
    
    for key, values in metrics.items():
        if len(values) >= 10:
            # Average of first 5 steps vs last 5 steps
            before = sum(values[:5]) / 5
            after = sum(values[-5:]) / 5
            
            # For wait time and energy consumption, lower is better
            if key in ['avg_wait_time', 'energy_consumption']:
                improvement = (before - after) / before * 100 if before > 0 else 0
                direction = 'decrease'
            else:
                improvement = (after - before) / before * 100 if before > 0 else 0
                direction = 'increase'
            
            before_after[key] = {
                'before': before,
                'after': after,
                'improvement': improvement,
                'direction': direction
            }
    
    # Create the figure
    labels = {
        'avg_wait_time': 'Wait Time',
        'throughput': 'Throughput',
        'energy_consumption': 'Energy',
        'grid_stability': 'Grid Stability'
    }
    
    # Prepare data for bar chart
    metrics_list = []
    before_values = []
    after_values = []
    
    for key, data in before_after.items():
        metrics_list.append(labels.get(key, key))
        before_values.append(data['before'])
        after_values.append(data['after'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=metrics_list,
        y=before_values,
        name='Before Optimization',
        marker_color='lightgray'
    ))
    
    fig.add_trace(go.Bar(
        x=metrics_list,
        y=after_values,
        name='After Optimization',
        marker_color='darkgreen'
    ))
    
    # Configure layout
    fig.update_layout(
        title_text="Optimization Impact",
        yaxis_title="Value",
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_correlation_heatmap(metrics):
    """Create a correlation heatmap between metrics"""
    # Create a DataFrame from metrics
    metric_data = {}
    
    # Ensure all metrics have the same length
    min_length = min(len(values) for values in metrics.values() if len(values) > 0)
    
    for key, values in metrics.items():
        if len(values) >= min_length:
            metric_data[key] = values[:min_length]
    
    if len(metric_data) >= 2:
        df = pd.DataFrame(metric_data)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create labels
        labels = {
            'avg_wait_time': 'Wait Time',
            'throughput': 'Throughput',
            'energy_consumption': 'Energy',
            'grid_stability': 'Stability'
        }
        
        # Rename columns and index
        corr_matrix = corr_matrix.rename(columns=labels)
        corr_matrix = corr_matrix.rename(index=labels)
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Metric Correlations"
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title=""
        )
        
        return fig
    
    # Default empty figure if not enough data
    fig = go.Figure()
    fig.update_layout(title_text="Not enough data for correlation analysis")
    return fig
