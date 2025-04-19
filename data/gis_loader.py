import streamlit as st
import numpy as np
import pandas as pd
import random
import time
from collections import defaultdict

def load_osm_data(city_name):
    """
    Load city data from OpenStreetMap
    
    Args:
        city_name (str): Name of the city to load
        
    Returns:
        dict: City data including roads, buildings, etc.
    """
    # In a real implementation, this would use osmnx or a similar library
    # to fetch actual OSM data. For this simplified version, we'll
    # generate a random city with a realistic structure.
    
    with st.spinner(f"Fetching OSM data for {city_name}..."):
        # Simulate a network request
        time.sleep(2)
        
        # Extract city name parts
        parts = city_name.split(',')
        district = parts[0].strip()
        
        # Generate a pseudo-random grid size based on the city name
        # This makes it consistent for the same city name
        seed = sum(ord(c) for c in city_name)
        random.seed(seed)
        
        grid_size = random.randint(8, 15)
        density = random.uniform(0.4, 0.8)
        
        # Generate the city grid
        city_data = generate_realistic_city(grid_size, density, district)
        
        return city_data

def generate_sample_city_grid(grid_size, density):
    """
    Generate a sample city grid with buildings and roads
    
    Args:
        grid_size (int): Size of the city grid
        density (float): Building density (0-1)
        
    Returns:
        dict: City data including roads, buildings, etc.
    """
    # Initialize city data
    city_data = {
        'grid_size': grid_size,
        'roads': [],
        'buildings': [],
        'intersections': []
    }
    
    # Generate roads - simple grid pattern
    for x in range(grid_size):
        for y in range(grid_size):
            # Horizontal roads
            if x < grid_size - 1:
                city_data['roads'].append([(x, y), (x+1, y)])
            # Vertical roads
            if y < grid_size - 1:
                city_data['roads'].append([(x, y), (x, y+1)])
    
    # Generate intersections
    for x in range(grid_size):
        for y in range(grid_size):
            city_data['intersections'].append((x, y))
    
    # Generate buildings
    building_types = ['residential', 'commercial', 'industrial', 'public']
    building_weights = [0.6, 0.25, 0.1, 0.05]
    
    for x in range(grid_size):
        for y in range(grid_size):
            # Randomly place buildings with probability based on density
            if random.random() < density:
                building_type = random.choices(building_types, weights=building_weights)[0]
                height = 0
                
                if building_type == 'residential':
                    height = random.uniform(1, 3)
                elif building_type == 'commercial':
                    height = random.uniform(2, 5)
                elif building_type == 'industrial':
                    height = random.uniform(1, 4)
                else:  # public
                    height = random.uniform(2, 4)
                
                building = {
                    'position': (x, y),
                    'width': random.uniform(0.5, 0.9),
                    'length': random.uniform(0.5, 0.9),
                    'height': height,
                    'type': building_type
                }
                city_data['buildings'].append(building)
    
    return city_data

def generate_realistic_city(grid_size, density, district_name):
    """
    Generate a more realistic city structure with districts and patterns
    
    Args:
        grid_size (int): Size of the city grid
        density (float): Overall building density
        district_name (str): Name of the district
        
    Returns:
        dict: City data with realistic urban patterns
    """
    # Initialize city data
    city_data = {
        'grid_size': grid_size,
        'name': district_name,
        'roads': [],
        'buildings': [],
        'intersections': []
    }
    
    # Create a density map that varies across the city
    # This will make some areas denser (downtown) and others sparser (suburbs)
    density_map = create_density_map(grid_size, density)
    
    # Generate a road network - start with a grid but add some irregularities
    generate_road_network(city_data, grid_size)
    
    # Generate buildings based on the density map and road network
    generate_buildings(city_data, density_map)
    
    # Identify districts/neighborhoods
    identify_districts(city_data, grid_size)
    
    return city_data

def create_density_map(grid_size, avg_density):
    """Create a density distribution map for the city"""
    # Create a base density map
    density_map = np.zeros((grid_size, grid_size))
    
    # Add a high-density center (downtown)
    center_x, center_y = grid_size // 2, grid_size // 2
    for x in range(grid_size):
        for y in range(grid_size):
            # Distance from center
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(2) * grid_size / 2
            
            # Density decreases with distance from center
            density_map[x, y] = avg_density * (1.5 - dist / max_dist)
    
    # Add some random variation
    variation = np.random.rand(grid_size, grid_size) * 0.3
    density_map = density_map + variation
    
    # Ensure density is between 0 and 1
    density_map = np.clip(density_map, 0.1, 0.95)
    
    return density_map

def generate_road_network(city_data, grid_size):
    """Generate a realistic road network"""
    # Start with a grid network
    for x in range(grid_size):
        for y in range(grid_size):
            # Horizontal roads
            if x < grid_size - 1:
                city_data['roads'].append([(x, y), (x+1, y)])
            # Vertical roads
            if y < grid_size - 1:
                city_data['roads'].append([(x, y), (x, y+1)])
    
    # Add some diagonal roads or irregularities
    num_diagonals = grid_size // 3
    for _ in range(num_diagonals):
        start_x = random.randint(0, grid_size-2)
        start_y = random.randint(0, grid_size-2)
        end_x = min(start_x + random.randint(1, 3), grid_size-1)
        end_y = min(start_y + random.randint(1, 3), grid_size-1)
        
        city_data['roads'].append([(start_x, start_y), (end_x, end_y)])
    
    # Generate intersections from roads
    intersections = set()
    for road in city_data['roads']:
        if len(road) >= 2:
            intersections.add(road[0])
            intersections.add(road[1])
    
    city_data['intersections'] = list(intersections)

def generate_buildings(city_data, density_map):
    """Generate buildings based on density map and road network"""
    grid_size = city_data['grid_size']
    
    # Building types and their relative frequencies
    building_types = ['residential', 'commercial', 'industrial', 'public']
    
    # For each cell in the grid
    for x in range(grid_size):
        for y in range(grid_size):
            # Randomly place buildings with probability based on density map
            if random.random() < density_map[x, y]:
                # Determine building type based on location
                type_weights = get_building_type_weights(x, y, grid_size)
                building_type = random.choices(building_types, weights=type_weights)[0]
                
                # Determine building height based on type and density
                height = determine_building_height(building_type, density_map[x, y])
                
                # Create building
                building = {
                    'position': (x, y),
                    'width': random.uniform(0.5, 0.9),
                    'length': random.uniform(0.5, 0.9),
                    'height': height,
                    'type': building_type
                }
                city_data['buildings'].append(building)

def get_building_type_weights(x, y, grid_size):
    """
    Determine building type weights based on location
    Downtown: More commercial
    Mid-city: Mixed
    Outskirts: More residential
    """
    center_x, center_y = grid_size // 2, grid_size // 2
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(2) * grid_size / 2
    
    normalized_dist = dist / max_dist
    
    if normalized_dist < 0.3:  # Downtown
        return [0.3, 0.5, 0.1, 0.1]  # More commercial
    elif normalized_dist < 0.7:  # Mid-city
        return [0.5, 0.3, 0.15, 0.05]  # Mixed
    else:  # Outskirts
        return [0.7, 0.1, 0.15, 0.05]  # More residential

def determine_building_height(building_type, density):
    """Determine building height based on type and density"""
    if building_type == 'residential':
        # Residential buildings: 1-5 stories
        base = 1 + 4 * density
        variation = random.uniform(-0.5, 0.5)
        return base + variation
    elif building_type == 'commercial':
        # Commercial buildings: 2-10 stories
        base = 2 + 8 * density
        variation = random.uniform(-1, 1)
        return base + variation
    elif building_type == 'industrial':
        # Industrial buildings: 1-4 stories
        base = 1 + 3 * density
        variation = random.uniform(-0.5, 0.5)
        return base + variation
    else:  # public
        # Public buildings: 2-6 stories
        base = 2 + 4 * density
        variation = random.uniform(-0.5, 0.5)
        return base + variation

def identify_districts(city_data, grid_size):
    """Identify and label districts/neighborhoods"""
    # This would create named districts based on building clusters
    # For simplicity, we'll just add a district field to the city data
    city_data['districts'] = [
        {'name': 'Downtown', 'center': (grid_size // 2, grid_size // 2), 'radius': grid_size // 4},
        {'name': 'Northern Suburb', 'center': (grid_size // 2, grid_size // 5), 'radius': grid_size // 5},
        {'name': 'Industrial Zone', 'center': (grid_size // 5, grid_size // 2), 'radius': grid_size // 6},
    ]
