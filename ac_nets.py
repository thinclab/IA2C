import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

# Encoder with LSTM
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(Encoder, self).__init__()
        self.public_fc = nn.Linear(input_dim, hidden_dim)
        self.private_fc = nn.Linear(input_dim, hidden_dim)
        self.action_fc = nn.Linear(action_dim, hidden_dim)
        self.action_dist_fc = nn.Linear(action_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, 32, batch_first=True)  # Latent layer size is 32

    def forward(self, public_obs, private_obs, action, action_dist, hidden_state):
        public_out = torch.relu(self.public_fc(public_obs))
        private_out = torch.relu(self.private_fc(private_obs))
        action_out = torch.relu(self.action_fc(action))
        action_dist_out = torch.relu(self.action_dist_fc(action_dist))
        combined = torch.cat((public_out, private_out, action_out, action_dist_out), dim=-1)

        combined, hidden_state = self.lstm(combined.unsqueeze(0), hidden_state)
        latent = combined.squeeze(0)  # Latent size is 32
        return latent, hidden_state

# Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, action_dim):
        super(Decoder, self).__init__()
        self.obs_reconstruct = nn.Linear(32, output_dim)  # Latent size is 32
        self.action_dist_reconstruct = nn.Linear(32, action_dim)

    def forward(self, latent):
        next_obs = torch.relu(self.obs_reconstruct(latent))
        next_action_dist = F.softmax(self.action_dist_reconstruct(latent), dim=-1)
        return next_obs, next_action_dist

# Actor Network
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, action_dim)
        self.fc1 = nn.Linear(32, hidden_dim)  # Input from latent space (32)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, public_obs, private_obs, action, action_dist, hidden_state):
        latent, hidden_state = self.encoder(public_obs, private_obs, action, action_dist, hidden_state)
        x = torch.tanh(self.fc1(latent))  # First hidden layer with Tanh
        action_probs = F.softmax(self.fc2(torch.relu(x)), dim=-1)  # Second hidden layer with ReLU
        return action_probs, latent, hidden_state

# Critic Network
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, action_dim)
        self.decoder = Decoder(hidden_dim, input_dim, action_dim)
        self.fc1 = nn.Linear(32 + input_dim, hidden_dim)  # Input from latent space (32) + public_obs
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, public_obs, private_obs, action, action_dist, hidden_state):
        latent, hidden_state = self.encoder(public_obs, private_obs, action, action_dist, hidden_state)
        combined_input = torch.cat((public_obs, latent), dim=-1)
        x = torch.tanh(self.fc1(combined_input))  # First hidden layer with Tanh
        value = self.fc2(torch.relu(x))  # Second hidden layer with ReLU
        next_obs, next_action_dist = self.decoder(latent)
        return value, next_obs, next_action_dist, hidden_state

# Loss Function
def compute_loss(observed_obs, predicted_obs, observed_action_dist, predicted_action_dist, alpha, C):
=
    # 1. Reconstruction Loss: MSE between predicted and observed next observations
    reconstruction_loss = F.mse_loss(predicted_obs, observed_obs)

    # 2. Log Posterior of the Dirichlet-Multinomial Model
    log_posterior_loss = -torch.sum(observed_action_dist * torch.log(predicted_action_dist + 1e-10))

    # 3. KL-Divergence between reconstructed and true Dirichlet distributions
    C_prime = alpha + C
    reconstructed_dirichlet = Dirichlet(predicted_action_dist)
    true_dirichlet = Dirichlet(C_prime)
    kl_divergence = torch.distributions.kl_divergence(reconstructed_dirichlet, true_dirichlet)

    # Total loss
    total_loss = reconstruction_loss + log_posterior_loss + kl_divergence

    return total_loss
