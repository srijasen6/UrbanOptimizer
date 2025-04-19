import streamlit as st
import pandas as pd
import numpy as np
import time

# Import project modules
from visualization.digital_twin import render_digital_twin
from visualization.dashboard import render_dashboard
from simulation.traffic_simulator import TrafficSimulator
from simulation.energy_grid import EnergyGrid
from agents.reinforcement_learning import TrafficRLAgent
from data.gis_loader import load_osm_data, generate_sample_city_grid
from utils.helpers import sidebar_config

# App title and configuration
st.set_page_config(
    page_title="AI Urban Planner",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar configuration
sidebar_config()

# Main title
st.title("AI Urban Planner: Smart City Optimization")

# Initialize session state for simulation variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.simulation_running = False
    st.session_state.simulation_speed = 1.0
    st.session_state.city_data = None
    st.session_state.traffic_simulator = None
    st.session_state.energy_grid = None
    st.session_state.rl_agent = None
    st.session_state.optimization_enabled = False
    st.session_state.current_step = 0
    st.session_state.metrics = {
        'avg_wait_time': [],
        'throughput': [],
        'energy_consumption': [],
        'grid_stability': []
    }

# Main layout with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Digital Twin", "Traffic Simulation", "Energy Grid", "Dashboard"])

with tab1:
    st.header("City Digital Twin")
    
    # City selection
    city_option = st.selectbox(
        "Select city data source",
        ["Generate simple grid", "Import from OpenStreetMap"]
    )
    
    if city_option == "Import from OpenStreetMap":
        city_name = st.text_input("Enter city name (e.g., 'Manhattan, New York City')")
        if st.button("Load City Data") and city_name:
            with st.spinner(f"Loading {city_name} data from OpenStreetMap..."):
                try:
                    city_data = load_osm_data(city_name)
                    st.session_state.city_data = city_data
                    st.success(f"Successfully loaded {city_name} data")
                except Exception as e:
                    st.error(f"Error loading city data: {str(e)}")
    else:
        cols = st.columns(3)
        grid_size = cols[0].slider("Grid size", 3, 15, 5)
        density = cols[1].slider("Building density", 0.1, 0.9, 0.5)
        if cols[2].button("Generate Grid"):
            with st.spinner("Generating city grid..."):
                st.session_state.city_data = generate_sample_city_grid(grid_size, density)
                st.success("Generated sample city grid")

    # Render digital twin if data is available
    if st.session_state.city_data is not None:
        render_digital_twin(st.session_state.city_data)
    else:
        st.info("Please load or generate city data to visualize the digital twin")

with tab2:
    st.header("Traffic Simulation")
    
    if st.session_state.city_data is None:
        st.warning("Please load or generate city data in the Digital Twin tab first")
    else:
        # Traffic parameters
        cols = st.columns(3)
        traffic_density = cols[0].slider("Traffic density", 0.1, 1.0, 0.5)
        peak_hour_factor = cols[1].slider("Peak hour factor", 0.5, 2.0, 1.0)
        emergency_vehicles = cols[2].slider("Emergency vehicles (per hour)", 0, 10, 2)
        
        # Initialize traffic simulator if not already initialized
        if st.session_state.traffic_simulator is None:
            st.session_state.traffic_simulator = TrafficSimulator(
                st.session_state.city_data,
                traffic_density,
                peak_hour_factor,
                emergency_vehicles
            )
        
        # Display traffic simulation control buttons
        cols = st.columns(4)
        start_stop = cols[0].button("Start/Stop Simulation")
        if start_stop:
            st.session_state.simulation_running = not st.session_state.simulation_running
        
        reset = cols[1].button("Reset Simulation")
        if reset:
            st.session_state.simulation_running = False
            st.session_state.current_step = 0
            st.session_state.traffic_simulator = TrafficSimulator(
                st.session_state.city_data,
                traffic_density,
                peak_hour_factor,
                emergency_vehicles
            )
            st.session_state.metrics = {
                'avg_wait_time': [],
                'throughput': [],
                'energy_consumption': [],
                'grid_stability': []
            }
        
        st.session_state.simulation_speed = cols[2].slider("Simulation Speed", 0.1, 2.0, 1.0)
        
        enable_rl = cols[3].checkbox("Enable RL Optimization", st.session_state.optimization_enabled)
        if enable_rl != st.session_state.optimization_enabled:
            st.session_state.optimization_enabled = enable_rl
            if enable_rl and st.session_state.rl_agent is None:
                st.session_state.rl_agent = TrafficRLAgent(st.session_state.traffic_simulator)
        
        # Visualize the traffic simulation
        traffic_state = st.session_state.traffic_simulator.get_current_state()
        st.plotly_chart(st.session_state.traffic_simulator.visualize_traffic(traffic_state))
        
        # Status information
        current_step = st.empty()
        current_step.text(f"Simulation step: {st.session_state.current_step}")
        
        # Run simulation steps if active
        if st.session_state.simulation_running:
            placeholder = st.empty()
            try:
                with placeholder.container():
                    traffic_metrics = st.session_state.traffic_simulator.step()
                    st.session_state.current_step += 1
                    
                    # Update metrics
                    st.session_state.metrics['avg_wait_time'].append(traffic_metrics['avg_wait_time'])
                    st.session_state.metrics['throughput'].append(traffic_metrics['throughput'])
                    
                    # Apply RL optimization if enabled
                    if st.session_state.optimization_enabled and st.session_state.rl_agent:
                        action = st.session_state.rl_agent.get_action(traffic_state)
                        st.session_state.traffic_simulator.apply_action(action)
                        st.write(f"RL Agent applied traffic light optimization")
                    
                    current_step.text(f"Simulation step: {st.session_state.current_step}")
                    
                    # Slow down based on simulation speed
                    time.sleep(1.0 / st.session_state.simulation_speed)
                    st.rerun()
            except Exception as e:
                st.error(f"Simulation error: {str(e)}")
                st.session_state.simulation_running = False

with tab3:
    st.header("Energy Grid Simulation")
    
    if st.session_state.city_data is None:
        st.warning("Please load or generate city data in the Digital Twin tab first")
    else:
        # Energy grid parameters
        cols = st.columns(3)
        renewable_percentage = cols[0].slider("Renewable Energy (%)", 0, 100, 30)
        consumption_factor = cols[1].slider("Consumption Factor", 0.5, 2.0, 1.0)
        grid_resilience = cols[2].slider("Grid Resilience", 0.1, 1.0, 0.7)
        
        # Initialize energy grid if not already initialized
        if st.session_state.energy_grid is None:
            st.session_state.energy_grid = EnergyGrid(
                st.session_state.city_data,
                renewable_percentage / 100,
                consumption_factor,
                grid_resilience
            )
        
        # Visualize the energy grid
        grid_state = st.session_state.energy_grid.get_current_state()
        st.plotly_chart(st.session_state.energy_grid.visualize_grid(grid_state))
        
        # Energy metrics
        if st.session_state.simulation_running and st.session_state.energy_grid:
            energy_metrics = st.session_state.energy_grid.step()
            
            # Update metrics
            st.session_state.metrics['energy_consumption'].append(energy_metrics['consumption'])
            st.session_state.metrics['grid_stability'].append(energy_metrics['stability'])
            
            # Display current metrics
            cols = st.columns(2)
            cols[0].metric("Current Consumption (MW)", f"{energy_metrics['consumption']:.2f}")
            cols[1].metric("Grid Stability (%)", f"{energy_metrics['stability'] * 100:.1f}")

with tab4:
    st.header("City Metrics Dashboard")
    
    render_dashboard(st.session_state.metrics)

# Run the app
if __name__ == "__main__":
    # This is handled by streamlit
    pass
