import torch
import gymnasium as gym
import numpy as np
from a2c_nets import ActorNetwork, CriticNetwork
from distributions import CategoricalPdType
from Org import Org

# Initialize environment and networks
env = Org()
obs_dim = 6
act_dim = 3
latent_dim = 64  # Example latent dimension, adjust as necessary
public_obs_dim = obs_dim
private_obs_dim = obs_dim
action_dist_dim = act_dim
action_space_dim = act_dim

actor = ActorNetwork(public_obs_dim, private_obs_dim, act_dim, action_dist_dim, latent_dim, action_space_dim)
critic = CriticNetwork(public_obs_dim, private_obs_dim, act_dim, action_dist_dim, latent_dim)

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

# Hyperparameters
num_episodes = 1000
gamma = 0.99

for episode in range(num_episodes):
    obs = env.reset()
    public_obs = torch.tensor([obs[0]], dtype=torch.float32).unsqueeze(0)
    private_obs = torch.tensor([obs[0]], dtype=torch.float32).unsqueeze(0)
    action = torch.zeros(1, 3, dtype=torch.float32)   # Placeholder for the action
    action_dist = torch.zeros(1, 3, dtype=torch.float32) # Placeholder for the action distribution

    done = False
    episode_reward = 0
    action_tensor = torch.zeros(1, 3, dtype=torch.float32)
    while not done:
        # Get action probabilities from actor
        print(action)
        action_probs = actor(public_obs, private_obs, action_tensor, action_dist)
        
        # Sample action from the distribution
        action_dist = CategoricalPdType(action_space_dim).pdfromflat(action_probs).sample().unsqueeze(0)
        action = action_dist.item()
        action_tensor = torch.zeros(1, 3, dtype=torch.float32)
        action_tensor[0, action] = 1.0
        # Step in the environment
        next_obs, reward, done, _ = env.step([action])
        next_public_obs = torch.tensor([next_obs[0]], dtype=torch.float32).unsqueeze(0)
        next_private_obs = torch.tensor([next_obs[0]], dtype=torch.float32).unsqueeze(0)

        # Calculate critic loss and update
        critic_loss = critic.compute_loss(public_obs, private_obs, action, action_dist, next_public_obs, action_dist, alpha=torch.tensor([1.0]), C_prime=torch.tensor([1.0]), C=torch.tensor([1.0]))
        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

        # Calculate advantage and actor loss
        Q_val = critic(next_public_obs, next_private_obs, action, action_dist)[0]
        advantage = reward + (1 - done) * gamma * Q_val - critic(public_obs, private_obs, action, action_dist)[0]
        actor_loss = -torch.log(action_probs[0, action]) * advantage

        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        # Update observation
        obs = next_obs
        public_obs = next_public_obs
        private_obs = next_private_obs

        episode_reward += reward

    print(f"Episode {episode + 1}, Reward: {episode_reward}")

env.close()
