import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import random

def render_digital_twin(city_data):
    """
    Render a 3D digital twin of the city using Plotly
    
    Args:
        city_data (dict): Dictionary containing city layout data
    """
    st.subheader("3D City Visualization")
    
    # Check what data we have
    grid_size = city_data.get('grid_size', 5)
    buildings = city_data.get('buildings', [])
    roads = city_data.get('roads', [])
    
    # Create a 3D visualization
    fig = go.Figure()
    
    # Add buildings if available
    if buildings:
        for building in buildings:
            add_building_to_figure(fig, building)
    else:
        # Generate simple buildings based on grid
        for x in range(grid_size):
            for y in range(grid_size):
                # Randomly place buildings with 40% probability
                if random.random() < 0.4:
                    height = random.randint(1, 5)
                    building = {
                        'position': (x, y),
                        'width': 0.8,
                        'length': 0.8,
                        'height': height,
                        'type': random.choice(['residential', 'commercial', 'industrial'])
                    }
                    add_building_to_figure(fig, building)
    
    # Add roads
    if roads:
        add_roads_to_figure(fig, roads)
    else:
        # Generate grid roads
        road_data = []
        for x in range(grid_size):
            for y in range(grid_size):
                # Horizontal roads
                if x < grid_size - 1:
                    road_data.append([(x, y), (x+1, y)])
                # Vertical roads
                if y < grid_size - 1:
                    road_data.append([(x, y), (x, y+1)])
        add_roads_to_figure(fig, road_data)
    
    # Add a base ground plane
    add_ground_plane(fig, grid_size)
    
    # Configure the layout
    fig.update_layout(
        title="City Digital Twin",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Height",
            aspectmode="data"
        ),
        scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600,
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

def add_building_to_figure(fig, building):
    """Add a 3D building to the figure"""
    # Extract building properties
    position = building.get('position', (0, 0))
    width = building.get('width', 0.8)
    length = building.get('length', 0.8)
    height = building.get('height', 1.0)
    building_type = building.get('type', 'residential')
    
    # Set color based on building type
    colors = {
        'residential': 'lightblue',
        'commercial': 'lightgreen',
        'industrial': 'lightgray',
        'public': 'lightyellow'
    }
    color = colors.get(building_type, 'lightblue')
    
    # Calculate vertices
    x, y = position
    vertices = [
        [x - width/2, y - length/2, 0],
        [x + width/2, y - length/2, 0],
        [x + width/2, y + length/2, 0],
        [x - width/2, y + length/2, 0],
        [x - width/2, y - length/2, height],
        [x + width/2, y - length/2, height],
        [x + width/2, y + length/2, height],
        [x - width/2, y + length/2, height]
    ]
    
    # Define faces for a cube
    i = [0, 0, 0, 0, 4, 4, 4, 4, 0, 1, 2, 3]
    j = [1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7]
    k = [2, 3, 0, 1, 6, 7, 4, 5, 1, 2, 3, 0]
    
    # Create the mesh3d trace
    fig.add_trace(go.Mesh3d(
        x=[v[0] for v in vertices],
        y=[v[1] for v in vertices],
        z=[v[2] for v in vertices],
        i=i, j=j, k=k,
        color=color,
        opacity=0.7,
        name=f"{building_type.capitalize()} Building"
    ))

def add_roads_to_figure(fig, roads):
    """Add roads to the figure"""
    for road in roads:
        if len(road) >= 2:  # Ensure road has start and end points
            start, end = road[0], road[1]
            
            # Add a line slightly above ground level
            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[0.01, 0.01],  # Slightly above ground
                mode='lines',
                line=dict(
                    color='gray',
                    width=6
                ),
                name='Road'
            ))

def add_ground_plane(fig, grid_size):
    """Add a ground plane to the figure"""
    # Create a grid of points for the ground
    x = np.linspace(0, grid_size-1, grid_size)
    y = np.linspace(0, grid_size-1, grid_size)
    z = np.zeros((grid_size, grid_size))
    
    # Add the surface
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, 'green'], [1, 'lightgreen']],
        showscale=False,
        opacity=0.6,
        name='Ground'
    ))

def create_heatmap(city_data, values, title):
    """Create a heatmap visualization for the city"""
    grid_size = city_data.get('grid_size', 5)
    
    # Create a grid for the heatmap
    grid = np.zeros((grid_size, grid_size))
    
    # Populate the grid with values
    for i, value in enumerate(values):
        x = i % grid_size
        y = i // grid_size
        if x < grid_size and y < grid_size:
            grid[y, x] = value
    
    # Create the heatmap figure
    fig = px.imshow(
        grid,
        labels=dict(x="X", y="Y", color=title),
        x=list(range(grid_size)),
        y=list(range(grid_size)),
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_city_profile(city_data):
    """Create a profile visualization of the city's composition"""
    # Extract or generate data
    building_types = ['Residential', 'Commercial', 'Industrial', 'Public']
    building_counts = []
    
    # Count buildings by type
    buildings = city_data.get('buildings', [])
    if buildings:
        type_counts = {'residential': 0, 'commercial': 0, 'industrial': 0, 'public': 0}
        for building in buildings:
            btype = building.get('type', 'residential')
            if btype in type_counts:
                type_counts[btype] += 1
        
        building_counts = [
            type_counts['residential'],
            type_counts['commercial'],
            type_counts['industrial'],
            type_counts['public']
        ]
    else:
        # Generate random counts for demonstration
        grid_size = city_data.get('grid_size', 5)
        total_buildings = int(grid_size * grid_size * 0.4)  # Assume 40% of grid has buildings
        
        # Distribution: 60% residential, 25% commercial, 10% industrial, 5% public
        building_counts = [
            int(total_buildings * 0.6),
            int(total_buildings * 0.25),
            int(total_buildings * 0.1),
            int(total_buildings * 0.05)
        ]
    
    # Create a pie chart
    fig = px.pie(
        values=building_counts,
        names=building_types,
        title="City Building Composition"
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig
