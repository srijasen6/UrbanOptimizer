import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
from collections import defaultdict

class Vehicle:
    """Class representing a vehicle in the traffic simulation"""
    def __init__(self, vehicle_id, origin, destination, start_time, is_emergency=False):
        self.id = vehicle_id
        self.origin = origin  # (x, y) coordinates
        self.destination = destination  # (x, y) coordinates
        self.start_time = start_time
        self.current_position = origin
        self.route = []  # Will be calculated
        self.waiting_time = 0
        self.is_emergency = is_emergency
        self.completed = False
        self.speed = 1.0  # Base speed in grid units per step
    
    def update_position(self, new_position):
        self.current_position = new_position
        
    def increment_waiting(self):
        self.waiting_time += 1
    
    def mark_completed(self):
        self.completed = True

class TrafficLight:
    """Class representing a traffic light in the simulation"""
    def __init__(self, intersection_id, position):
        self.id = intersection_id
        self.position = position  # (x, y) coordinates
        self.state = 0  # 0: North-South green, 1: East-West green
        self.timer = 0
        self.default_cycle = 20  # Default cycle length in simulation steps
    
    def cycle(self):
        """Change the traffic light state"""
        self.state = (self.state + 1) % 2
        self.timer = 0
    
    def step(self):
        """Increment timer and cycle if needed"""
        self.timer += 1
        if self.timer >= self.default_cycle:
            self.cycle()
    
    def force_state(self, new_state):
        """Force the traffic light to a specific state (used by RL agent)"""
        if new_state != self.state:
            self.state = new_state
            self.timer = 0
    
    def is_green(self, direction):
        """Check if light is green for a specific direction"""
        if direction in ['north', 'south'] and self.state == 0:
            return True
        elif direction in ['east', 'west'] and self.state == 1:
            return True
        return False

class TrafficSimulator:
    """Traffic simulation for a city grid"""
    def __init__(self, city_data, traffic_density=0.5, peak_hour_factor=1.0, emergency_vehicle_rate=2):
        self.city_data = city_data
        self.traffic_density = traffic_density
        self.peak_hour_factor = peak_hour_factor
        self.emergency_vehicle_rate = emergency_vehicle_rate
        
        # Initialize simulation components
        self.vehicles = {}
        self.traffic_lights = {}
        self.roads = set()
        self.intersections = set()
        self.current_step = 0
        self.next_vehicle_id = 0
        
        # Statistics
        self.stats = {
            'avg_wait_time': [],
            'throughput': [],
            'emergency_response_time': [],
        }
        
        self._initialize_from_city_data()
    
    def _initialize_from_city_data(self):
        """Initialize roads, intersections, and traffic lights from city data"""
        # Extract grid size from city data
        if 'grid_size' in self.city_data:
            grid_size = self.city_data['grid_size']
        else:
            # Estimate from the roads data
            max_x = max(road[0][0] for road in self.city_data.get('roads', []) if road)
            max_y = max(road[0][1] for road in self.city_data.get('roads', []) if road)
            grid_size = max(max_x, max_y) + 1
        
        self.grid_size = grid_size
        
        # Generate road network if not present in city data
        if 'roads' not in self.city_data or not self.city_data['roads']:
            # Create a simple grid of roads
            roads = []
            for x in range(grid_size):
                for y in range(grid_size):
                    # Horizontal roads
                    if x < grid_size - 1:
                        roads.append([(x, y), (x+1, y)])
                    # Vertical roads
                    if y < grid_size - 1:
                        roads.append([(x, y), (x, y+1)])
            self.city_data['roads'] = roads
        
        # Initialize roads
        for road in self.city_data['roads']:
            if len(road) >= 2:  # Ensure road has start and end points
                start, end = road[0], road[1]
                self.roads.add((start, end))
                self.roads.add((end, start))  # Make roads bidirectional
        
        # Initialize intersections and traffic lights
        for x in range(grid_size):
            for y in range(grid_size):
                # Check if this point is an intersection (connected to multiple roads)
                incoming_roads = 0
                for (start, end) in self.roads:
                    if end == (x, y):
                        incoming_roads += 1
                
                if incoming_roads > 1:
                    self.intersections.add((x, y))
                    intersection_id = f"i_{x}_{y}"
                    self.traffic_lights[intersection_id] = TrafficLight(intersection_id, (x, y))
    
    def reset(self):
        """Reset the simulation"""
        self.vehicles = {}
        self.current_step = 0
        self.next_vehicle_id = 0
        
        # Reset all traffic lights
        for light in self.traffic_lights.values():
            light.state = 0
            light.timer = 0
        
        # Reset statistics
        self.stats = {
            'avg_wait_time': [],
            'throughput': [],
            'emergency_response_time': [],
        }
    
    def generate_vehicle(self):
        """Generate a new vehicle with random origin and destination"""
        # Get random road endpoints as origin and destination
        road_endpoints = list({point for road in self.roads for point in road})
        if len(road_endpoints) < 2:
            return None
        
        origin = random.choice(road_endpoints)
        destination = random.choice([p for p in road_endpoints if p != origin])
        
        # Determine if this is an emergency vehicle
        is_emergency = random.random() < (self.emergency_vehicle_rate / 100)
        
        # Create and return the vehicle
        vehicle = Vehicle(
            self.next_vehicle_id,
            origin,
            destination,
            self.current_step,
            is_emergency
        )
        self.next_vehicle_id += 1
        
        return vehicle
    
    def add_vehicles(self):
        """Add new vehicles based on traffic density and peak hour factor"""
        # Calculate how many vehicles to add this step
        base_rate = self.traffic_density * 0.1  # Base vehicle generation rate
        
        # Apply peak hour factor (could simulate morning/evening rush hours)
        time_of_day = (self.current_step % 240) / 240  # 0-1 representing time of day
        peak_factor = 1 + self.peak_hour_factor * np.sin(time_of_day * 2 * np.pi)
        
        rate = base_rate * peak_factor
        
        # Probabilistic vehicle generation
        if random.random() < rate:
            vehicle = self.generate_vehicle()
            if vehicle:
                self.vehicles[vehicle.id] = vehicle
    
    def calculate_shortest_path(self, origin, destination):
        """Calculate shortest path using A* algorithm"""
        # Simple implementation for grid-based movement
        # In a real system, this would use A* or Dijkstra's algorithm
        
        # For grid movement, we'll just do a simple greedy path
        path = [origin]
        current = origin
        
        # Simple greedy path (move in x direction, then y direction)
        while current != destination:
            x, y = current
            dest_x, dest_y = destination
            
            if x < dest_x:
                next_point = (x + 1, y)
            elif x > dest_x:
                next_point = (x - 1, y)
            elif y < dest_y:
                next_point = (x, y + 1)
            elif y > dest_y:
                next_point = (x, y - 1)
            else:
                break  # We've reached the destination
                
            # Check if this move is valid (there's a road)
            if (current, next_point) in self.roads:
                path.append(next_point)
                current = next_point
            else:
                # No direct road, try to find alternative
                alternatives = []
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    alt_next = (x + dx, y + dy)
                    if (current, alt_next) in self.roads:
                        alternatives.append(alt_next)
                
                if alternatives:
                    # Choose the alternative closest to destination
                    next_point = min(alternatives, 
                                    key=lambda p: ((p[0]-dest_x)**2 + (p[1]-dest_y)**2))
                    path.append(next_point)
                    current = next_point
                else:
                    break  # No valid moves
        
        return path
    
    def move_vehicles(self):
        """Move all vehicles according to their routes and traffic rules"""
        vehicles_to_remove = []
        
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle.completed:
                vehicles_to_remove.append(vehicle_id)
                continue
            
            # Calculate route if not yet calculated
            if not vehicle.route:
                vehicle.route = self.calculate_shortest_path(vehicle.origin, vehicle.destination)
            
            # Find current position in route
            if vehicle.current_position in vehicle.route:
                current_idx = vehicle.route.index(vehicle.current_position)
            else:
                # Vehicle is not on its route, recalculate
                vehicle.route = self.calculate_shortest_path(vehicle.current_position, vehicle.destination)
                current_idx = 0
            
            # Check if reached destination
            if current_idx == len(vehicle.route) - 1:
                vehicle.mark_completed()
                vehicles_to_remove.append(vehicle_id)
                continue
            
            # Determine next position
            next_position = vehicle.route[current_idx + 1]
            
            # Check for traffic light at current position
            can_move = True
            for light_id, light in self.traffic_lights.items():
                if light.position == vehicle.current_position:
                    # Determine direction of movement
                    x1, y1 = vehicle.current_position
                    x2, y2 = next_position
                    
                    if x1 == x2:  # Moving vertically
                        direction = 'north' if y2 > y1 else 'south'
                    else:  # Moving horizontally
                        direction = 'east' if x2 > x1 else 'west'
                    
                    # Check if light is green for this direction
                    # Emergency vehicles can bypass red lights
                    if not light.is_green(direction) and not vehicle.is_emergency:
                        can_move = False
                        vehicle.increment_waiting()
                        break
            
            # Move the vehicle if possible
            if can_move:
                vehicle.update_position(next_position)
            else:
                vehicle.increment_waiting()
        
        # Remove completed vehicles
        for vehicle_id in vehicles_to_remove:
            if vehicle_id in self.vehicles:
                del self.vehicles[vehicle_id]
    
    def update_traffic_lights(self):
        """Update all traffic lights"""
        for light in self.traffic_lights.values():
            light.step()
    
    def calculate_metrics(self):
        """Calculate traffic metrics for the current step"""
        # Average waiting time
        waiting_times = [v.waiting_time for v in self.vehicles.values()]
        avg_wait_time = np.mean(waiting_times) if waiting_times else 0
        
        # Throughput (completed vehicles in the last 10 steps)
        completed_last_period = sum(1 for v in self.vehicles.values() if v.completed)
        
        # Emergency vehicle response time
        emergency_vehicles = [v for v in self.vehicles.values() if v.is_emergency]
        response_times = []
        for ev in emergency_vehicles:
            if ev.completed:
                response_times.append(self.current_step - ev.start_time)
        avg_emergency_time = np.mean(response_times) if response_times else 0
        
        return {
            'avg_wait_time': avg_wait_time,
            'throughput': completed_last_period,
            'emergency_response_time': avg_emergency_time
        }
    
    def step(self):
        """Advance the simulation by one step"""
        self.current_step += 1
        
        # Add new vehicles
        self.add_vehicles()
        
        # Update traffic lights
        self.update_traffic_lights()
        
        # Move vehicles
        self.move_vehicles()
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Store metrics
        for key, value in metrics.items():
            if key in self.stats:
                self.stats[key].append(value)
        
        return metrics
    
    def get_current_state(self):
        """Get the current state of the simulation for visualization and RL"""
        state = {
            'vehicles': [
                {
                    'id': v.id,
                    'position': v.current_position,
                    'is_emergency': v.is_emergency,
                    'waiting_time': v.waiting_time
                }
                for v in self.vehicles.values()
            ],
            'traffic_lights': [
                {
                    'id': t.id,
                    'position': t.position,
                    'state': t.state
                }
                for t in self.traffic_lights.values()
            ],
            'traffic_density': self.traffic_density,
            'current_step': self.current_step,
            'grid_size': self.grid_size,
            
            # Additional state information for RL agent
            'queue_lengths': self._calculate_queue_lengths(),
            'waiting_times': [v.waiting_time for v in self.vehicles.values()],
            'current_phase': [t.state for t in self.traffic_lights.values()]
        }
        return state
    
    def _calculate_queue_lengths(self):
        """Calculate queue lengths at each intersection"""
        queues = defaultdict(int)
        
        for vehicle in self.vehicles.values():
            if vehicle.waiting_time > 0:
                # Find nearest intersection
                min_dist = float('inf')
                nearest_intersection = None
                
                for intersection in self.intersections:
                    dist = ((vehicle.current_position[0] - intersection[0])**2 + 
                            (vehicle.current_position[1] - intersection[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_intersection = intersection
                
                if nearest_intersection and min_dist < 2:  # Only count if close to intersection
                    queues[nearest_intersection] += 1
        
        return list(queues.values())
    
    def get_state_size(self):
        """Return the size of the state vector for RL"""
        # This is a simplified representation; in a real system,
        # you'd need a more sophisticated state representation
        return 30  # Example size
    
    def get_action_size(self):
        """Return the number of possible actions for RL"""
        return len(self.traffic_lights) * 2  # Each light has 2 states
    
    def apply_action(self, action):
        """Apply an RL agent's action to the traffic lights"""
        # Convert action index to traffic light and state
        num_lights = len(self.traffic_lights)
        
        if action < num_lights * 2:
            light_idx = action // 2
            new_state = action % 2
            
            # Get the traffic light
            light_id = list(self.traffic_lights.keys())[light_idx]
            light = self.traffic_lights[light_id]
            
            # Apply the new state
            prev_state = light.state
            light.force_state(new_state)
            
            # Calculate immediate reward based on queue reduction
            prev_queues = sum(self._calculate_queue_lengths())
            self.step()  # Advance simulation
            new_queues = sum(self._calculate_queue_lengths())
            
            # Reward is reduction in queue lengths
            reward = prev_queues - new_queues
            
            # Penalize unnecessary changes
            if prev_state == new_state:
                reward += 0.5  # Small bonus for maintaining state if beneficial
            
            return reward
        else:
            # Invalid action
            return -10  # Large penalty
    
    def visualize_traffic(self, state):
        """Visualize the current traffic state using Plotly"""
        vehicles = state['vehicles']
        traffic_lights = state['traffic_lights']
        grid_size = state['grid_size']
        
        # Create figure
        fig = go.Figure()
        
        # Add roads
        road_x = []
        road_y = []
        
        for (start, end) in self.roads:
            road_x.extend([start[0], end[0], None])
            road_y.extend([start[1], end[1], None])
        
        fig.add_trace(go.Scatter(
            x=road_x, y=road_y,
            mode='lines',
            line=dict(color='gray', width=2),
            name='Roads'
        ))
        
        # Add traffic lights
        light_x = []
        light_y = []
        light_colors = []
        
        for light in traffic_lights:
            light_x.append(light['position'][0])
            light_y.append(light['position'][1])
            light_colors.append('green' if light['state'] == 1 else 'red')
        
        fig.add_trace(go.Scatter(
            x=light_x, y=light_y,
            mode='markers',
            marker=dict(
                size=15,
                color=light_colors,
                symbol='square'
            ),
            name='Traffic Lights'
        ))
        
        # Add vehicles
        vehicle_x = []
        vehicle_y = []
        vehicle_colors = []
        
        for vehicle in vehicles:
            vehicle_x.append(vehicle['position'][0])
            vehicle_y.append(vehicle['position'][1])
            vehicle_colors.append('red' if vehicle['is_emergency'] else 'blue')
        
        fig.add_trace(go.Scatter(
            x=vehicle_x, y=vehicle_y,
            mode='markers',
            marker=dict(
                size=10,
                color=vehicle_colors
            ),
            name='Vehicles'
        ))
        
        # Configure layout
        fig.update_layout(
            title='Traffic Simulation',
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
