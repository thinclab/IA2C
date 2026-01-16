import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from mappo_ac_net import ActorNetwork, CriticNetwork
# Proximal Policy Optimization algorithm implementation
class PPO:
    def __init__(self, local_obs_dim, global_obs_dim, action_dim, num_agents):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize policy network and optimizer
        self.actors = []
        self.critics = []
        self.optimizers = [[] for i in range(num_agents)]
        for i in range(num_agents):
            self.actors.append(ActorNetwork(local_obs_dim, action_dim))
            self.critics.append(CriticNetwork(local_obs_dim, global_obs_dim))
        #self.policy = ActorCritic(state_dim, action_dim).to(self.device)
            self.optimizers[i].append(optim.Adam(self.actors[-1].parameters(), lr=3e-4))
            self.optimizers[i].append(optim.Adam(self.critics[-1].parameters(), lr=3e-4))
        
        # Hyperparameters
        self.gamma = 0.99        # Discount factor for future rewards
        self.eps_clip = 0.2      # Clip range for policy ratio
        self.K_epochs = 10       # Number of optimization epochs per update
        self.batch_size = 64     # Mini-batch size for optimization
        self.entropy_coef = 0.01  # Coefficient for entropy regularization

    def update(self, states, actions, old_log_probs, returns, advantages):
        """Update policy network using PPO-clip objective"""
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device).detach()
        returns = torch.FloatTensor(returns).to(self.device).detach()
        advantages = torch.FloatTensor(advantages).to(self.device).detach()

        # Multiple epochs of optimization
        for _ in range(self.K_epochs):
            # Create random indices for mini-batch sampling
            indices = torch.randperm(len(states))
            
            # Mini-batch updates
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                # Get new policy distribution
                mean, std = self.policy.get_action(states[idx])
                dist = Normal(mean, std)
                
                # Calculate new log probabilities and entropy
                new_log_probs = dist.log_prob(actions[idx]).sum(-1)
                entropy = dist.entropy().mean()
                
                # Compute probability ratio
                ratios = torch.exp(new_log_probs - old_log_probs[idx])
                
                # PPO-clip objective
                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # Value loss (MSE)
                values = self.policy.get_value(states[idx]).squeeze()
                critic_loss = (returns[idx] - values).pow(2).mean()
                
                # Total loss
                loss = actor_loss + 0.5 * critic_loss
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # Gradient clipping
                self.optimizer.step()

    def train(self, env_name, max_episodes=1000, max_steps=200):
        """Main training loop"""
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0
            
            # Buffer for trajectory data
            states = []
            actions = []
            rewards = []
            old_log_probs = []
            
            # Collect trajectory data
            for _ in range(max_steps):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                    mean, std = self.policy.get_action(state_tensor)
                    dist = Normal(mean, std)
                    action = dist.sample().squeeze(0).cpu().numpy()
                    log_prob = dist.log_prob(torch.FloatTensor(action).to(self.device)).sum()
                
                # Execute action and store transition
                next_state, reward, done, _ = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                old_log_probs.append(log_prob.item())
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Calculate discounted returns
            returns = []
            discounted_reward = 0
            for r in reversed(rewards):
                discounted_reward = r + self.gamma * discounted_reward
                returns.insert(0, discounted_reward)
            
            # Normalize advantages
            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            advantages = returns - returns.mean()
            
            # Update policy with collected data
            self.update(states, actions, old_log_probs, returns.numpy(), advantages.numpy())
            
            # Print training progress
            print(f"Episode: {episode+1}, Reward: {episode_reward}")