import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
import torch.distributions as dist

class Encoder(nn.Module):
    def __init__(self, public_obs_dim, private_obs_dim, action_dim, action_dist_dim, latent_dim):
        super(Encoder, self).__init__()
        # Define individual input layers
        self.public_obs_fc = nn.Linear(public_obs_dim, 128)
        self.private_obs_fc = nn.Linear(private_obs_dim, 128)
        self.action_fc = nn.Linear(action_dim, 128)
        self.action_dist_fc = nn.Linear(action_dist_dim, 128)
        
        # Define a layer to combine all inputs
        self.combined_fc = nn.Linear(128 * 4, 256)
        
        # Define the final layer to produce the latent vector
        self.latent_fc = nn.Linear(256, latent_dim)
        
    def forward(self, public_obs, private_obs, action, action_dist):
        # Process each input separately
        public_obs_out = torch.relu(self.public_obs_fc(public_obs))
        private_obs_out = torch.relu(self.private_obs_fc(private_obs))
        action_out = torch.relu(self.action_fc(action))
        action_dist_out = torch.relu(self.action_dist_fc(action_dist))
        
        # Concatenate the processed inputs
        print("pubdim= ", action_out.size())
        combined = torch.cat((public_obs_out, private_obs_out, action_out, action_dist_out), dim=1)
        
        # Process the combined inputs
        combined_out = torch.relu(self.combined_fc(combined))
        
        # Produce the latent vector
        latent = self.latent_fc(combined_out)
        
        return latent

class Decoder(nn.Module):
    def __init__(self, latent_dim, public_obs_dim, action_dist_dim):
        super(Decoder, self).__init__()
        # Define the layers to reconstruct the inputs
        self.fc1 = nn.Linear(latent_dim, 256)
        self.public_obs_fc = nn.Linear(256, public_obs_dim)
        self.action_dist_fc = nn.Linear(256, action_dist_dim)
        
    def forward(self, latent):
        x = torch.relu(self.fc1(latent))
        public_obs_recon = self.public_obs_fc(x)
        action_dist_recon = self.action_dist_fc(x)
        
        return public_obs_recon, action_dist_recon

class CriticNetwork(nn.Module):
    def __init__(self, public_obs_dim, private_obs_dim, action_dim, action_dist_dim, latent_dim):
        super(CriticNetwork, self).__init__()

        self.encoder = Encoder(public_obs_dim, private_obs_dim, action_dim, action_dist_dim, latent_dim)
        self.decoder = Decoder(latent_dim, public_obs_dim, action_dist_dim)
    
    def forward(self, public_obs, private_obs, action, action_dist):
        latent = self.encoder(public_obs, private_obs, action, action_dist)
        return self.decoder(latent)
    
    def compute_loss(self, public_obs, private_obs, action, action_dist, true_public_obs, true_action_dist, alpha, C_prime, C):
        latent, (public_obs_recon, action_dist_recon) = self.forward(public_obs, private_obs, action, action_dist)

        # Mean squared error for the reconstruction of the public observation
        recon_loss = nn.MSELoss()(public_obs_recon, true_public_obs)
        
        # Log posterior of the Dirichlet-multinomial model
        log_posterior = -torch.sum(dist.Dirichlet(alpha + C_prime).log_prob(action_dist_recon))
        
        # KL divergence
        kl_div = dist.kl_divergence(dist.Dirichlet(C), dist.Dirichlet(alpha + C_prime)).sum()
        
        # Total loss
        loss = recon_loss + log_posterior + kl_div
        return loss

class ActorNetwork(nn.Module):
    def __init__(self, public_obs_dim, private_obs_dim, action_dim, action_dist_dim, latent_dim, action_space_dim):
        super(ActorNetwork, self).__init__()
        print(public_obs_dim)
        self.encoder = Encoder(public_obs_dim, private_obs_dim, action_dim, action_dist_dim, latent_dim)
        
        # Define the actor network layers
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, action_space_dim)
        
        # Softmax layer to get action probabilities
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, public_obs, private_obs, action, action_dist):
        # Get the latent representation from the encoder
        print(public_obs)
        latent = self.encoder(public_obs, private_obs, action, action_dist)
        
        # Pass through the actor network layers
        x = torch.relu(self.fc1(latent))
        x = torch.relu(self.fc2(x))
        action_probs = self.softmax(self.action_head(x))
        
        return action_probs
