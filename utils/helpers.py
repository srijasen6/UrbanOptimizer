import streamlit as st
import pandas as pd
import numpy as np
import random
import datetime
import time

def sidebar_config():
    """Configure the sidebar with app information and options"""
    st.sidebar.title("AI Urban Planner")
    st.sidebar.write("A Smart City Optimization Platform")
    
    st.sidebar.markdown("---")
    
    # About this app
    with st.sidebar.expander("About this app", expanded=False):
        st.write("""
        This application demonstrates the concepts of a smart city optimization platform featuring:
        
        - Digital twin visualization of a city
        - Traffic flow simulation
        - Energy grid simulation
        - Reinforcement learning for city optimization
        - Performance dashboards
        
        The platform aims to help urban planners explore different scenarios and optimize city infrastructure.
        """)
    
    # Controls and settings
    st.sidebar.markdown("## Settings")
    
    simulation_speed = st.sidebar.slider(
        "Simulation Speed", 
        min_value=0.1, 
        max_value=2.0, 
        value=st.session_state.get('simulation_speed', 1.0),
        step=0.1
    )
    st.session_state.simulation_speed = simulation_speed
    
    # Environment factors
    st.sidebar.markdown("## Environment Factors")
    
    time_of_day = st.sidebar.select_slider(
        "Time of Day",
        options=["Early Morning", "Rush Hour", "Midday", "Evening", "Night"],
        value="Midday"
    )
    
    weather = st.sidebar.selectbox(
        "Weather Condition",
        options=["Clear", "Cloudy", "Rainy", "Snowy"],
        index=0
    )
    
    # Apply environment factors to simulations
    apply_environment_factors(time_of_day, weather)
    
    st.sidebar.markdown("---")
    
    # Display simulation statistics if available
    if 'current_step' in st.session_state and st.session_state.current_step > 0:
        st.sidebar.markdown("## Simulation Statistics")
        st.sidebar.write(f"Current step: {st.session_state.current_step}")
        
        if 'traffic_simulator' in st.session_state and st.session_state.traffic_simulator:
            active_vehicles = len(st.session_state.traffic_simulator.vehicles)
            st.sidebar.write(f"Active vehicles: {active_vehicles}")
        
        if 'energy_grid' in st.session_state and st.session_state.energy_grid:
            grid_status = "Stable" if st.session_state.energy_grid.grid_stability() > 0.7 else "Unstable"
            st.sidebar.write(f"Grid status: {grid_status}")
    
    st.sidebar.markdown("---")
    st.sidebar.info("This is a simplified model for demonstration purposes.")

def apply_environment_factors(time_of_day, weather):
    """Apply environmental factors to simulations"""
    # Time of day factors
    time_factors = {
        "Early Morning": {"traffic": 0.6, "energy": 0.7},
        "Rush Hour": {"traffic": 1.5, "energy": 1.2},
        "Midday": {"traffic": 1.0, "energy": 1.0},
        "Evening": {"traffic": 1.3, "energy": 1.3},
        "Night": {"traffic": 0.3, "energy": 0.8}
    }
    
    # Weather factors
    weather_factors = {
        "Clear": {"traffic": 1.0, "energy": 1.0, "renewable": 1.0},
        "Cloudy": {"traffic": 0.9, "energy": 1.1, "renewable": 0.6},
        "Rainy": {"traffic": 0.7, "energy": 1.2, "renewable": 0.3},
        "Snowy": {"traffic": 0.5, "energy": 1.4, "renewable": 0.2}
    }
    
    # Apply factors to traffic simulator if available
    if 'traffic_simulator' in st.session_state and st.session_state.traffic_simulator:
        traffic_factor = time_factors[time_of_day]["traffic"] * weather_factors[weather]["traffic"]
        st.session_state.traffic_simulator.peak_hour_factor = traffic_factor
    
    # Apply factors to energy grid if available
    if 'energy_grid' in st.session_state and st.session_state.energy_grid:
        energy_factor = time_factors[time_of_day]["energy"] * weather_factors[weather]["energy"]
        renewable_factor = weather_factors[weather]["renewable"]
        
        # This would update consumption and weather factors in the energy grid
        if hasattr(st.session_state.energy_grid, 'consumption_factor'):
            st.session_state.energy_grid.consumption_factor = energy_factor
        if hasattr(st.session_state.energy_grid, 'weather_factor'):
            st.session_state.energy_grid.weather_factor = renewable_factor

def generate_random_city_name():
    """Generate a random city name for demo purposes"""
    prefixes = ["New", "Old", "North", "South", "East", "West", "Central", "Upper", "Lower"]
    roots = ["York", "London", "Paris", "Berlin", "Tokyo", "Rome", "Athens", "Brook", "Ridge", 
             "Field", "Grove", "Valley", "Hill", "Lake", "Port", "Haven", "Ville", "Town"]
    suffixes = ["burg", "ton", "ville", "ford", "field", "land", "berg", "boro", "ham", "shire"]
    
    name_parts = []
    
    # 30% chance to add a prefix
    if random.random() < 0.3:
        name_parts.append(random.choice(prefixes))
    
    # Always add a root
    name_parts.append(random.choice(roots))
    
    # 40% chance to add a suffix
    if random.random() < 0.4:
        name_parts.append(random.choice(suffixes))
    
    return "".join(name_parts)

def format_time(timestamp):
    """Format a timestamp for display"""
    return datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
