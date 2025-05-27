"""
Multi-Agent Reinforcement Learning environment for cybersecurity responses
"""
import json
import random
import csv
from pathlib import Path

class CyberSecurityEnvironment:
    """MARL environment for cybersecurity response training"""
    
    def __init__(self):
        self.state_size = 100
        self.action_size = 6
        self.num_agents = 3
        
        # Define actions
        self.actions = [
            "block_ip",
            "isolate_endpoint", 
            "update_firewall",
            "alert_admin",
            "quarantine_file",
            "reset_connection"
        ]
        
        # Define agent roles
        self.agents = {
            0: "DetectionAgent",
            1: "ResponseAgent", 
            2: "AnalysisAgent"
        }
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.state = [random.random() for _ in range(self.state_size)]
        self.threat_level = random.uniform(0, 1)
        self.time_step = 0
        self.max_steps = 100
        
        return self.get_state()
    
    def get_state(self):
        """Get current environment state"""
        return {
            "global_state": self.state,
            "threat_level": self.threat_level,
            "time_step": self.time_step,
            "agents_active": self.num_agents
        }
    
    def step(self, actions):
        """Execute actions and return new state, rewards, done"""
        # Validate actions
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")
        
        # Calculate rewards based on action coordination
        rewards = self.calculate_rewards(actions)
        
        # Update state
        self.update_state(actions)
        
        # Check if episode is done
        self.time_step += 1
        done = (self.time_step >= self.max_steps) or (self.threat_level < 0.1)
        
        return self.get_state(), rewards, done
    
    def calculate_rewards(self, actions):
        """Calculate rewards for each agent based on actions"""
        rewards = []
        
        for i, action in enumerate(actions):
            if action < 0 or action >= self.action_size:
                # Invalid action penalty
                reward = -1.0
            else:
                # Base reward for valid action
                reward = 0.1
                
                # Bonus for threat reduction
                if self.actions[action] in ["block_ip", "isolate_endpoint", "quarantine_file"]:
                    reward += self.threat_level * 0.5
                
                # Coordination bonus (if multiple agents take complementary actions)
                if i == 0 and action in [0, 1]:  # Detection agent blocking/isolating
                    reward += 0.2
                elif i == 1 and action in [2, 5]:  # Response agent updating/resetting
                    reward += 0.2
                elif i == 2 and action == 3:  # Analysis agent alerting
                    reward += 0.1
            
            rewards.append(reward)
        
        return rewards
    
    def update_state(self, actions):
        """Update environment state based on actions"""
        # Reduce threat level based on effective actions
        threat_reduction = 0
        
        for action in actions:
            if action in [0, 1, 4]:  # Blocking, isolation, quarantine
                threat_reduction += 0.1
            elif action in [2, 5]:  # Firewall update, connection reset
                threat_reduction += 0.05
        
        self.threat_level = max(0, self.threat_level - threat_reduction)
        
        # Update state vector (simple random walk with action influence)
        for i in range(len(self.state)):
            self.state[i] += random.uniform(-0.1, 0.1)
            self.state[i] = max(0, min(1, self.state[i]))  # Clamp to [0,1]

def create_marl_training_data():
    """Generate training scenarios for MARL"""
    print("Creating MARL training scenarios...")
    
    env = CyberSecurityEnvironment()
    scenarios = []
    
    # Generate diverse training scenarios
    for episode in range(100):
        env.reset()
        scenario = {
            "episode": episode,
            "initial_state": env.get_state(),
            "steps": []
        }
        
        done = False
        while not done:
            # Generate random actions for demonstration
            actions = [random.randint(0, env.action_size - 1) for _ in range(env.num_agents)]
            
            state_before = env.get_state()
            new_state, rewards, done = env.step(actions)
            
            step_data = {
                "state": state_before,
                "actions": actions,
                "rewards": rewards,
                "next_state": new_state,
                "done": done
            }
            
            scenario["steps"].append(step_data)
        
        scenarios.append(scenario)
    
    # Save scenarios
    marl_dir = Path("data/marl_scenarios")
    marl_dir.mkdir(exist_ok=True)
    
    with open(marl_dir / "training_scenarios.json", 'w') as f:
        json.dump(scenarios, f, indent=2)
    
    print(f"Generated {len(scenarios)} MARL training scenarios")
    print(f"   Average episode length: {sum(len(s['steps']) for s in scenarios) / len(scenarios):.1f}")

def create_marl_config():
    """Create MARL-specific configuration"""
    config = {
        "environment": {
            "name": "CyberSecurityMARL",
            "state_size": 100,
            "action_size": 6,
            "num_agents": 3,
            "max_episode_steps": 100,
            "reward_scale": 1.0
        },
        "agents": {
            "DetectionAgent": {
                "id": 0,
                "preferred_actions": ["block_ip", "isolate_endpoint"],
                "learning_rate": 0.001,
                "epsilon": 0.1,
                "network": {
                    "hidden_layers": [256, 128],
                    "activation": "relu"
                }
            },
            "ResponseAgent": {
                "id": 1,
                "preferred_actions": ["update_firewall", "reset_connection"],
                "learning_rate": 0.001,
                "epsilon": 0.1,
                "network": {
                    "hidden_layers": [256, 128],
                    "activation": "relu"
                }
            },
            "AnalysisAgent": {
                "id": 2,
                "preferred_actions": ["alert_admin", "quarantine_file"],
                "learning_rate": 0.001,
                "epsilon": 0.1,
                "network": {
                    "hidden_layers": [256, 128],
                    "activation": "relu"
                }
            }
        },
        "training": {
            "episodes": 1000,
            "batch_size": 32,
            "memory_size": 10000,
            "target_update": 100,
            "save_frequency": 100
        }
    }
    
    with open("configs/marl_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Created MARL configuration")

if __name__ == "__main__":
    create_marl_training_data()
    create_marl_config()
