"""
AutoSentinel Phase 4: Multi-Agent Reinforcement Learning
Cybersecurity simulation environment and agent training
"""

import json
import os
import random
import math
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CybersecurityEnvironment:
    """Cybersecurity simulation environment for MARL training"""
    
    def __init__(self, network_size=10, max_steps=50):
        self.network_size = network_size
        self.max_steps = max_steps
        self.reset()
        
        # Action space: [monitor, block, isolate, alert, patch, backup]
        self.action_space = ["monitor", "block", "isolate", "alert", "patch", "backup"]
        self.num_actions = len(self.action_space)
        
        # Attack types
        self.attack_types = ["ddos", "malware", "phishing", "ransomware", "insider_threat"]
        
    def reset(self):
        """Reset environment to initial state"""
        self.step_count = 0
        self.network_health = 1.0
        self.security_score = 1.0
        self.active_threats = []
        self.detected_threats = []
        self.blocked_ips = set()
        self.isolated_nodes = set()
        
        # Initialize network nodes
        self.nodes = {}
        for i in range(self.network_size):
            self.nodes[i] = {
                "health": 1.0,
                "compromised": False,
                "patched": True,
                "monitoring": False,
                "traffic": random.uniform(0.1, 1.0)
            }
        
        return self.get_state()
    
    def get_state(self):
        """Get current environment state"""
        state = {
            "network_health": self.network_health,
            "security_score": self.security_score,
            "step_count": self.step_count,
            "num_active_threats": len(self.active_threats),
            "num_detected_threats": len(self.detected_threats),
            "num_blocked_ips": len(self.blocked_ips),
            "num_isolated_nodes": len(self.isolated_nodes),
            "avg_node_health": sum(node["health"] for node in self.nodes.values()) / len(self.nodes),
            "compromised_nodes": sum(1 for node in self.nodes.values() if node["compromised"]),
            "monitoring_coverage": sum(1 for node in self.nodes.values() if node["monitoring"]) / len(self.nodes)
        }
        return state
    
    def generate_threat(self):
        """Generate a new cybersecurity threat"""
        if random.random() < 0.3:  # 30% chance of new threat each step
            threat = {
                "id": len(self.active_threats),
                "type": random.choice(self.attack_types),
                "severity": random.uniform(0.1, 1.0),
                "target_node": random.randint(0, self.network_size - 1),
                "detected": False,
                "blocked": False,
                "steps_active": 0
            }
            self.active_threats.append(threat)
            logger.debug(f"New threat generated: {threat['type']} targeting node {threat['target_node']}")
    
    def update_threats(self):
        """Update existing threats"""
        for threat in self.active_threats[:]:
            threat["steps_active"] += 1
            
            # Threat detection based on monitoring
            target_node = self.nodes[threat["target_node"]]
            if target_node["monitoring"] and not threat["detected"]:
                if random.random() < 0.7:  # 70% detection chance with monitoring
                    threat["detected"] = True
                    self.detected_threats.append(threat)
                    logger.debug(f"Threat {threat['id']} detected!")
            
            # Threat impact
            if not threat["blocked"] and not threat["detected"]:
                # Undetected threats cause damage
                target_node["health"] -= threat["severity"] * 0.1
                if target_node["health"] < 0.5:
                    target_node["compromised"] = True
                
                # Spread to connected nodes
                if random.random() < 0.2:  # 20% chance to spread
                    connected_node = random.randint(0, self.network_size - 1)
                    if connected_node != threat["target_node"]:
                        self.nodes[connected_node]["health"] -= threat["severity"] * 0.05
            
            # Remove old threats
            if threat["steps_active"] > 10:
                self.active_threats.remove(threat)
    
    def step(self, actions):
        """Execute actions and advance environment one step"""
        self.step_count += 1
        
        # Generate new threats
        self.generate_threat()
        
        # Process agent actions
        rewards = {}
        for agent_id, action in actions.items():
            reward = self.process_action(agent_id, action)
            rewards[agent_id] = reward
        
        # Update existing threats
        self.update_threats()
        
        # Update network metrics
        self.update_network_metrics()
        
        # Check if episode is done
        done = (self.step_count >= self.max_steps or 
                self.network_health < 0.1 or 
                self.security_score < 0.1)
        
        return self.get_state(), rewards, done
    
    def process_action(self, agent_id, action_idx):
        """Process a single agent action"""
        if action_idx >= len(self.action_space):
            return -0.1  # Invalid action penalty
        
        action = self.action_space[action_idx]
        reward = 0.0
        
        # Choose random target (in real implementation, agents would specify targets)
        target_node = random.randint(0, self.network_size - 1)
        
        if action == "monitor":
            self.nodes[target_node]["monitoring"] = True
            reward = 0.1  # Small reward for monitoring
            
        elif action == "block":
            # Block threats targeting this node
            blocked_threats = 0
            for threat in self.active_threats:
                if threat["target_node"] == target_node and not threat["blocked"]:
                    threat["blocked"] = True
                    blocked_threats += 1
            reward = blocked_threats * 0.5
            
        elif action == "isolate":
            self.isolated_nodes.add(target_node)
            # Prevent threat spread from isolated node
            reward = 0.2
            
        elif action == "alert":
            # Increase detection probability
            reward = 0.1
            
        elif action == "patch":
            self.nodes[target_node]["patched"] = True
            self.nodes[target_node]["health"] = min(1.0, self.nodes[target_node]["health"] + 0.2)
            reward = 0.3
            
        elif action == "backup":
            # Improve resilience
            reward = 0.1
        
        return reward
    
    def update_network_metrics(self):
        """Update overall network health and security metrics"""
        # Calculate network health
        total_health = sum(node["health"] for node in self.nodes.values())
        self.network_health = total_health / len(self.nodes)
        
        # Calculate security score
        detected_ratio = len(self.detected_threats) / max(1, len(self.active_threats) + len(self.detected_threats))
        compromised_ratio = sum(1 for node in self.nodes.values() if node["compromised"]) / len(self.nodes)
        monitoring_ratio = sum(1 for node in self.nodes.values() if node["monitoring"]) / len(self.nodes)
        
        self.security_score = (detected_ratio * 0.4 + 
                              (1 - compromised_ratio) * 0.4 + 
                              monitoring_ratio * 0.2)

class SimpleQLearningAgent:
    """Simple Q-Learning agent for cybersecurity actions"""
    
    def __init__(self, agent_id, num_actions=6, learning_rate=0.1, epsilon=0.1, discount=0.95):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount = discount
        
        # Q-table (simplified state representation)
        self.q_table = {}
        
        # Experience tracking
        self.total_reward = 0.0
        self.episode_rewards = []
        
    def get_state_key(self, state):
        """Convert state to string key for Q-table"""
        # Simplified state representation
        return (
            int(state["network_health"] * 10),
            int(state["security_score"] * 10),
            min(state["num_active_threats"], 5),
            min(state["compromised_nodes"], 5)
        )
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)
        
        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.num_actions
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return self.q_table[state_key].index(max(self.q_table[state_key]))
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning update rule"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-values if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.num_actions
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * self.num_actions
        
        # Q-learning update
        old_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key])
        new_q = old_q + self.learning_rate * (reward + self.discount * max_next_q - old_q)
        self.q_table[state_key][action] = new_q
        
        self.total_reward += reward

class MARLTrainer:
    """Multi-Agent Reinforcement Learning trainer"""
    
    def __init__(self, num_agents=3, model_dir="models/marl"):
        self.num_agents = num_agents
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize environment and agents
        self.env = CybersecurityEnvironment()
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = SimpleQLearningAgent(agent_id=i)
    
    def train_episode(self):
        """Train agents for one episode"""
        state = self.env.reset()
        done = False
        episode_rewards = {i: 0.0 for i in range(self.num_agents)}
        
        while not done:
            # Get actions from all agents
            actions = {}
            agent_states = {}
            for agent_id, agent in self.agents.items():
                agent_states[agent_id] = state
                actions[agent_id] = agent.choose_action(state)
            
            # Environment step
            next_state, rewards, done = self.env.step(actions)
            
            # Update agents
            for agent_id, agent in self.agents.items():
                agent.update_q_value(
                    agent_states[agent_id], 
                    actions[agent_id], 
                    rewards[agent_id], 
                    next_state
                )
                episode_rewards[agent_id] += rewards[agent_id]
            
            state = next_state
        
        return episode_rewards, self.env.security_score, self.env.network_health
    
    def train(self, episodes=100):
        """Train the multi-agent system"""
        logger.info(f"Starting MARL training with {self.num_agents} agents...")
        
        episode_scores = []
        
        for episode in range(episodes):
            episode_rewards, security_score, network_health = self.train_episode()
            
            avg_reward = sum(episode_rewards.values()) / len(episode_rewards)
            episode_scores.append({
                "episode": episode,
                "avg_reward": avg_reward,
                "security_score": security_score,
                "network_health": network_health
            })
            
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Avg Reward: {avg_reward:.3f}, "
                           f"Security: {security_score:.3f}, Health: {network_health:.3f}")
        
        # Save training results
        self.save_results(episode_scores)
        self.save_agents()
        
        return episode_scores
    
    def save_results(self, episode_scores):
        """Save training results"""
        results_path = self.model_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(episode_scores, f, indent=2)
        logger.info(f"Training results saved to {results_path}")
    
    def save_agents(self):
        """Save trained agents"""
        for agent_id, agent in self.agents.items():
            agent_state = {
                "agent_id": agent_id,
                "total_reward": agent.total_reward,
                "q_table_size": len(agent.q_table),
                "learning_rate": agent.learning_rate,
                "epsilon": agent.epsilon,
                "discount": agent.discount
            }
            
            agent_path = self.model_dir / f"agent_{agent_id}.json"
            with open(agent_path, "w") as f:
                json.dump(agent_state, f, indent=2)
        
        logger.info(f"Agents saved to {self.model_dir}")
    
    def evaluate(self, episodes=10):
        """Evaluate trained agents"""
        logger.info("Evaluating trained agents...")
        
        # Set agents to evaluation mode (no exploration)
        original_epsilons = {}
        for agent_id, agent in self.agents.items():
            original_epsilons[agent_id] = agent.epsilon
            agent.epsilon = 0.0  # No exploration during evaluation
        
        eval_scores = []
        for episode in range(episodes):
            _, security_score, network_health = self.train_episode()
            eval_scores.append({
                "security_score": security_score,
                "network_health": network_health
            })
        
        # Restore original exploration rates
        for agent_id, agent in self.agents.items():
            agent.epsilon = original_epsilons[agent_id]
        
        # Calculate averages
        avg_security = sum(score["security_score"] for score in eval_scores) / len(eval_scores)
        avg_health = sum(score["network_health"] for score in eval_scores) / len(eval_scores)
        
        logger.info(f"Evaluation complete: Avg Security: {avg_security:.3f}, "
                   f"Avg Health: {avg_health:.3f}")
        
        return avg_security, avg_health

def main():
    """Main MARL training function"""
    print("AutoSentinel MARL Training - Phase 4")
    print("=" * 50)
    
    # Initialize trainer
    trainer = MARLTrainer(num_agents=3)
    
    # Train agents
    episode_scores = trainer.train(episodes=50)
    
    # Evaluate performance
    avg_security, avg_health = trainer.evaluate(episodes=10)
    
    # Print final results
    print(f"\nTraining Results:")
    print(f"Episodes completed: {len(episode_scores)}")
    print(f"Final average security score: {avg_security:.3f}")
    print(f"Final average network health: {avg_health:.3f}")
    
    # Print agent statistics
    print(f"\nAgent Statistics:")
    for agent_id, agent in trainer.agents.items():
        q_table_size = len(agent.q_table)
        total_reward = agent.total_reward
        print(f"Agent {agent_id}: Q-table size: {q_table_size}, Total reward: {total_reward:.2f}")
    
    print(f"\nModels saved to: {trainer.model_dir}")
    print("MARL training phase complete!")
    print("\nNext: Integrate with AutoSentinel main system")

if __name__ == "__main__":
    main()
