import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
from collections import defaultdict

class PowerSource:
    """Class representing a power source in the energy grid"""
    def __init__(self, source_id, position, capacity, is_renewable=False):
        self.id = source_id
        self.position = position  # (x, y) coordinates
        self.capacity = capacity  # Maximum power output in MW
        self.is_renewable = is_renewable
        self.current_output = 0.0  # Current power output
        self.reliability = 0.95 if not is_renewable else 0.8  # Base reliability
        self.online = True
    
    def update_output(self, demand_factor, weather_factor=1.0):
        """Update power output based on demand and weather"""
        if not self.online:
            self.current_output = 0.0
            return
        
        if self.is_renewable:
            # Renewable sources are affected by weather
            max_output = self.capacity * weather_factor
            variation = random.uniform(0.7, 1.0)  # Renewable sources have more variability
            self.current_output = min(max_output * variation, self.capacity)
        else:
            # Non-renewable sources respond to demand
            target_output = self.capacity * demand_factor
            variation = random.uniform(0.9, 1.0)  # Less variability for non-renewable
            self.current_output = min(target_output * variation, self.capacity)
    
    def check_failure(self, grid_stress=0.0):
        """Check if the power source experiences a failure"""
        failure_chance = (1 - self.reliability) * (1 + grid_stress)
        if random.random() < failure_chance:
            self.online = False
            return True
        return False
    
    def repair(self):
        """Repair the power source"""
        self.online = True

class EnergyConsumer:
    """Class representing an energy consumer in the grid"""
    def __init__(self, consumer_id, position, base_consumption):
        self.id = consumer_id
        self.position = position  # (x, y) coordinates
        self.base_consumption = base_consumption  # Base energy consumption in MW
        self.current_consumption = base_consumption
        self.priority = random.uniform(0.5, 1.0)  # Priority level (for load shedding)
    
    def update_consumption(self, time_factor, temperature_factor=1.0):
        """Update consumption based on time of day and temperature"""
        # Time factor represents time of day effects (e.g., peak hours)
        # Temperature factor affects heating/cooling needs
        variation = random.uniform(0.8, 1.2)  # Random variation
        self.current_consumption = self.base_consumption * time_factor * temperature_factor * variation

class EnergyGrid:
    """Simulation of an energy grid for a city"""
    def __init__(self, city_data, renewable_percentage=0.3, consumption_factor=1.0, grid_resilience=0.7):
        self.city_data = city_data
        self.renewable_percentage = renewable_percentage
        self.consumption_factor = consumption_factor
        self.grid_resilience = grid_resilience  # 0.0 to 1.0, higher means more resilient
        
        # Initialize grid components
        self.power_sources = {}
        self.consumers = {}
        self.grid_lines = set()
        self.current_step = 0
        
        # Environmental factors
        self.temperature = 20.0  # Degrees Celsius
        self.weather_condition = 'clear'  # clear, cloudy, rainy, etc.
        self.weather_factor = 1.0  # Impact on renewable energy
        
        # Grid statistics
        self.stats = {
            'total_production': [],
            'total_consumption': [],
            'renewable_percentage': [],
            'grid_stability': [],
            'blackouts': []
        }
        
        self._initialize_from_city_data()
    
    def _initialize_from_city_data(self):
        """Initialize grid from city data"""
        # Extract grid size
        if 'grid_size' in self.city_data:
            grid_size = self.city_data['grid_size']
        else:
            # Estimate from roads data if available
            max_coord = 0
            for item in self.city_data.get('roads', []):
                if item and len(item) >= 2:
                    max_coord = max(max_coord, item[0][0], item[0][1], item[1][0], item[1][1])
            grid_size = max_coord + 1
        
        self.grid_size = grid_size
        
        # Create power sources based on grid size
        num_sources = max(1, grid_size // 2)
        
        # Determine how many are renewable
        num_renewable = int(num_sources * self.renewable_percentage)
        num_nonrenewable = num_sources - num_renewable
        
        # Place power sources at strategic locations
        for i in range(num_nonrenewable):
            position = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
            capacity = random.uniform(50, 200)  # MW
            source_id = f"power_{i}"
            self.power_sources[source_id] = PowerSource(source_id, position, capacity, False)
        
        for i in range(num_renewable):
            position = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
            capacity = random.uniform(20, 100)  # MW
            source_id = f"renewable_{i}"
            self.power_sources[source_id] = PowerSource(source_id, position, capacity, True)
        
        # Create consumers based on building locations or grid size
        buildings = self.city_data.get('buildings', [])
        if not buildings:
            # Create consumers based on grid size
            num_consumers = grid_size * grid_size // 3
            for i in range(num_consumers):
                position = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
                base_consumption = random.uniform(0.5, 5.0)  # MW
                consumer_id = f"consumer_{i}"
                self.consumers[consumer_id] = EnergyConsumer(consumer_id, position, base_consumption)
        else:
            # Create consumers at building locations
            for i, building in enumerate(buildings):
                position = building.get('position', (0, 0))
                size = building.get('size', 1)
                base_consumption = size * random.uniform(0.5, 2.0)  # MW
                consumer_id = f"building_{i}"
                self.consumers[consumer_id] = EnergyConsumer(consumer_id, position, base_consumption)
        
        # Create grid connections
        self._create_grid_connections()
    
    def _create_grid_connections(self):
        """Create connections between power sources and consumers"""
        # Create a simple grid network
        all_nodes = [(source.position, source_id) for source_id, source in self.power_sources.items()]
        all_nodes.extend([(consumer.position, consumer_id) for consumer_id, consumer in self.consumers.items()])
        
        # Connect each node to its nearest neighbors
        for (pos1, id1) in all_nodes:
            # Find nearest nodes
            nearest = sorted(
                [(((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2), id2) 
                 for (pos2, id2) in all_nodes if id2 != id1]
            )
            
            # Connect to nearest nodes
            for i in range(min(3, len(nearest))):
                self.grid_lines.add((id1, nearest[i][1]))
    
    def update_environmental_factors(self):
        """Update temperature and weather conditions"""
        # Simulate changing weather and temperature
        self.current_step += 1
        
        # Temperature varies throughout the day
        hour_of_day = (self.current_step % 24)
        self.temperature = 20 + 5 * np.sin(hour_of_day * np.pi / 12)
        
        # Weather conditions change less frequently
        if self.current_step % 12 == 0:
            weather_options = ['clear', 'partly_cloudy', 'cloudy', 'rainy']
            weights = [0.4, 0.3, 0.2, 0.1]
            self.weather_condition = random.choices(weather_options, weights=weights)[0]
            
            # Weather affects renewable energy production
            weather_factors = {
                'clear': 1.0,
                'partly_cloudy': 0.7,
                'cloudy': 0.4,
                'rainy': 0.2
            }
            self.weather_factor = weather_factors[self.weather_condition]
    
    def calculate_demand_factor(self):
        """Calculate demand factor based on time of day"""
        hour_of_day = (self.current_step % 24)
        
        # Morning peak: 7-9am, Evening peak: 6-9pm
        if 7 <= hour_of_day <= 9:
            return 1.2 * self.consumption_factor
        elif 18 <= hour_of_day <= 21:
            return 1.3 * self.consumption_factor
        elif 0 <= hour_of_day <= 5:
            return 0.6 * self.consumption_factor
        else:
            return 1.0 * self.consumption_factor
    
    def calculate_temperature_factor(self):
        """Calculate how temperature affects energy consumption"""
        # Higher temperatures increase AC usage
        # Lower temperatures increase heating usage
        if self.temperature > 25:
            return 1.0 + (self.temperature - 25) * 0.05
        elif self.temperature < 15:
            return 1.0 + (15 - self.temperature) * 0.04
        else:
            return 1.0
    
    def update_production(self):
        """Update energy production from all sources"""
        demand_factor = self.calculate_demand_factor()
        
        for source in self.power_sources.values():
            source.update_output(demand_factor, self.weather_factor)
            
            # Check for failures
            grid_stress = 1.0 - self.grid_stability()
            source.check_failure(grid_stress)
    
    def update_consumption(self):
        """Update energy consumption for all consumers"""
        time_factor = self.calculate_demand_factor()
        temperature_factor = self.calculate_temperature_factor()
        
        for consumer in self.consumers.values():
            consumer.update_consumption(time_factor, temperature_factor)
    
    def total_production(self):
        """Calculate total energy production"""
        return sum(source.current_output for source in self.power_sources.values())
    
    def total_consumption(self):
        """Calculate total energy consumption"""
        return sum(consumer.current_consumption for consumer in self.consumers.values())
    
    def renewable_ratio(self):
        """Calculate the ratio of renewable to total energy production"""
        total = self.total_production()
        if total == 0:
            return 0
        
        renewable = sum(source.current_output for source in self.power_sources.values() 
                        if source.is_renewable)
        return renewable / total
    
    def grid_stability(self):
        """Calculate grid stability based on production/consumption balance"""
        production = self.total_production()
        consumption = self.total_consumption()
        
        if production == 0:
            return 0  # No production means no stability
        
        # Closer to 1.0 means production meets consumption
        # Values less than 0.9 or greater than 1.1 indicate instability
        balance_ratio = production / max(consumption, 0.001)
        
        if 0.9 <= balance_ratio <= 1.1:
            # Good balance
            stability = self.grid_resilience * min(1.0, 2.0 - abs(balance_ratio - 1.0) * 5)
        else:
            # Poor balance
            stability = self.grid_resilience * max(0.0, 1.0 - abs(balance_ratio - 1.0))
        
        return stability
    
    def check_blackouts(self):
        """Check and simulate blackouts based on grid stability"""
        stability = self.grid_stability()
        blackout_threshold = 0.3  # Threshold below which blackouts occur
        
        if stability < blackout_threshold:
            # Calculate affected areas
            affected_percentage = (blackout_threshold - stability) / blackout_threshold
            affected_percentage = min(1.0, affected_percentage * 1.5)  # Amplify effect
            
            # Determine affected consumers
            for consumer in self.consumers.values():
                if random.random() < affected_percentage:
                    # This consumer is affected by the blackout
                    consumer.current_consumption = 0
            
            # Record the blackout
            self.stats['blackouts'].append({
                'step': self.current_step,
                'severity': affected_percentage,
                'affected_percentage': affected_percentage
            })
            
            return affected_percentage
        
        return 0  # No blackout
    
    def step(self):
        """Advance the simulation by one step"""
        self.current_step += 1
        
        # Update environmental factors
        self.update_environmental_factors()
        
        # Update energy production and consumption
        self.update_production()
        self.update_consumption()
        
        # Check for blackouts
        blackout_severity = self.check_blackouts()
        
        # Calculate metrics
        production = self.total_production()
        consumption = self.total_consumption()
        renewable = self.renewable_ratio()
        stability = self.grid_stability()
        
        # Record metrics
        self.stats['total_production'].append(production)
        self.stats['total_consumption'].append(consumption)
        self.stats['renewable_percentage'].append(renewable)
        self.stats['grid_stability'].append(stability)
        
        # Return metrics for this step
        return {
            'production': production,
            'consumption': consumption,
            'renewable': renewable,
            'stability': stability,
            'blackout': blackout_severity,
            'weather': self.weather_condition,
            'temperature': self.temperature
        }
    
    def get_current_state(self):
        """Get the current state of the energy grid for visualization"""
        state = {
            'power_sources': [
                {
                    'id': source.id,
                    'position': source.position,
                    'output': source.current_output,
                    'capacity': source.capacity,
                    'is_renewable': source.is_renewable,
                    'online': source.online
                }
                for source in self.power_sources.values()
            ],
            'consumers': [
                {
                    'id': consumer.id,
                    'position': consumer.position,
                    'consumption': consumer.current_consumption,
                    'priority': consumer.priority
                }
                for consumer in self.consumers.values()
            ],
            'grid_lines': list(self.grid_lines),
            'metrics': {
                'production': self.total_production(),
                'consumption': self.total_consumption(),
                'renewable_ratio': self.renewable_ratio(),
                'stability': self.grid_stability()
            },
            'environment': {
                'temperature': self.temperature,
                'weather': self.weather_condition,
                'weather_factor': self.weather_factor
            },
            'grid_size': self.grid_size
        }
        return state
    
    def visualize_grid(self, state):
        """Visualize the energy grid using Plotly"""
        power_sources = state['power_sources']
        consumers = state['consumers']
        grid_lines = state['grid_lines']
        grid_size = state['grid_size']
        
        # Create figure
        fig = go.Figure()
        
        # Add grid connections
        line_x = []
        line_y = []
        
        # Map IDs to positions
        id_to_pos = {}
        for source in power_sources:
            id_to_pos[source['id']] = source['position']
        for consumer in consumers:
            id_to_pos[consumer['id']] = consumer['position']
        
        # Draw grid lines
        for (id1, id2) in grid_lines:
            if id1 in id_to_pos and id2 in id_to_pos:
                pos1 = id_to_pos[id1]
                pos2 = id_to_pos[id2]
                line_x.extend([pos1[0], pos2[0], None])
                line_y.extend([pos1[1], pos2[1], None])
        
        fig.add_trace(go.Scatter(
            x=line_x, y=line_y,
            mode='lines',
            line=dict(color='lightgray', width=1),
            name='Grid Connections'
        ))
        
        # Add power sources
        source_x = []
        source_y = []
        source_text = []
        source_sizes = []
        source_colors = []
        
        for source in power_sources:
            source_x.append(source['position'][0])
            source_y.append(source['position'][1])
            source_text.append(
                f"ID: {source['id']}<br>"
                f"Output: {source['output']:.1f} MW<br>"
                f"Capacity: {source['capacity']:.1f} MW<br>"
                f"Type: {'Renewable' if source['is_renewable'] else 'Non-renewable'}<br>"
                f"Status: {'Online' if source['online'] else 'Offline'}"
            )
            
            # Size proportional to capacity
            source_sizes.append(10 + source['capacity'] / 10)
            
            # Color based on type and status
            if not source['online']:
                source_colors.append('gray')  # Offline
            elif source['is_renewable']:
                source_colors.append('green')  # Renewable
            else:
                source_colors.append('orange')  # Non-renewable
        
        fig.add_trace(go.Scatter(
            x=source_x, y=source_y,
            mode='markers',
            marker=dict(
                size=source_sizes,
                color=source_colors,
                symbol='circle'
            ),
            text=source_text,
            hoverinfo='text',
            name='Power Sources'
        ))
        
        # Add consumers
        consumer_x = []
        consumer_y = []
        consumer_text = []
        consumer_sizes = []
        
        for consumer in consumers:
            consumer_x.append(consumer['position'][0])
            consumer_y.append(consumer['position'][1])
            consumer_text.append(
                f"ID: {consumer['id']}<br>"
                f"Consumption: {consumer['consumption']:.2f} MW<br>"
                f"Priority: {consumer['priority']:.2f}"
            )
            
            # Size proportional to consumption
            consumer_sizes.append(5 + consumer['consumption'] * 2)
        
        fig.add_trace(go.Scatter(
            x=consumer_x, y=consumer_y,
            mode='markers',
            marker=dict(
                size=consumer_sizes,
                color='blue',
                symbol='square'
            ),
            text=consumer_text,
            hoverinfo='text',
            name='Consumers'
        ))
        
        # Configure layout
        fig.update_layout(
            title='Energy Grid Simulation',
            xaxis=dict(
                title='X',
                range=[-1, grid_size]
            ),
            yaxis=dict(
                title='Y',
                range=[-1, grid_size]
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=500
        )
        
        return fig
