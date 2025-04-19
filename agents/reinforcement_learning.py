import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQNModel(nn.Module):
    """
    Deep Q-Network model for traffic light control
    """
    def __init__(self, input_size, output_size):
        super(DQNModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class TrafficRLAgent:
    """
    Reinforcement Learning agent for traffic light optimization
    using Deep Q-Learning
    """
    def __init__(self, traffic_simulator, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.traffic_simulator = traffic_simulator
        self.state_size = traffic_simulator.get_state_size()
        self.action_size = traffic_simulator.get_action_size()
        
        # RL parameters
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        
        # Initialize DQN model
        self.model = DQNModel(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay memory
        self.memory = deque(maxlen=2000)
        
        # Training flag (initially in inference mode)
        self.training = False
    
    def get_state_tensor(self, state):
        """Convert state dict to tensor for the model"""
        # Extract state features (this should be adapted to your specific state representation)
        features = np.array([
            state.get('traffic_density', []),
            state.get('queue_lengths', []),
            state.get('waiting_times', []),
            state.get('current_phase', [])
        ], dtype=np.float32).flatten()
        
        # Ensure we have the right state size
        if len(features) < self.state_size:
            features = np.pad(features, (0, self.state_size - len(features)))
        elif len(features) > self.state_size:
            features = features[:self.state_size]
            
        return torch.FloatTensor(features).unsqueeze(0)
    
    def get_action(self, state):
        """
        Get action using epsilon-greedy policy
        In production, we might just use the best action without exploration
        """
        if not self.training or np.random.rand() > self.epsilon:
            # Exploitation: use the model to predict best action
            state_tensor = self.get_state_tensor(state)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
        else:
            # Exploration: random action
            return random.randrange(self.action_size)
    
    def memorize(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Train the model using experience replay"""
        if len(self.memory) < batch_size:
            return
            
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = self.get_state_tensor(state)
            next_state_tensor = self.get_state_tensor(next_state)
            
            # Current Q-values
            current_q = self.model(state_tensor)
            
            # Target Q-values
            target_q = current_q.clone()
            
            if done:
                target_q[0][action] = reward
            else:
                with torch.no_grad():
                    next_q_values = self.model(next_state_tensor)
                target_q[0][action] = reward + self.gamma * torch.max(next_q_values).item()
            
            # Update model
            self.optimizer.zero_grad()
            loss = self.criterion(current_q, target_q)
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, episodes=100, max_steps=1000, batch_size=32):
        """Train the agent over multiple episodes"""
        self.training = True
        
        for episode in range(episodes):
            self.traffic_simulator.reset()
            state = self.traffic_simulator.get_current_state()
            total_reward = 0
            
            for step in range(max_steps):
                # Get action, execute it, and get new state and reward
                action = self.get_action(state)
                reward = self.traffic_simulator.apply_action(action)
                next_state = self.traffic_simulator.get_current_state()
                done = step == max_steps - 1  # In a city simulation, episodes don't really "end"
                
                # Store experience and learn
                self.memorize(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                # Train model through experience replay
                self.replay(batch_size)
                
                if done:
                    break
            
            print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")
        
        self.training = False
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model.load_state_dict(torch.load(filepath))
        self.epsilon = self.epsilon_min  # Set epsilon to minimum for minimal exploration
