import numpy as np
import torch
import torch.nn as nn
from numpy import dtype
import torch.optim as optim
LR = 0.000001

class EncoderDecoderNetwork(nn.Module):
    def __init__(self, input_dim, latent_size, output_public_obs_dim, out_theta):
        super(EncoderDecoderNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size),
        )

        self.decoder_observation = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_public_obs_dim)
        )

        self.decoder_action_dist = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, out_theta),
            nn.Softmax(dim=-1)
        )
        self.mse_lose = torch.nn.MSELoss()
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.optimizer_encdec = optim.Adam(list(self.encoder.parameters()) + 
                                           list(self.decoder_observation.parameters())+ 
                                            list(self.decoder_action_dist.parameters()), lr=0.001)



    def forward(self, public_obs, private_obs, action, theta):
        # Concatenate observation and action
        public_obs = torch.tensor(public_obs, dtype=torch.float32)  # Ensure proper dtype
        private_obs = torch.tensor(private_obs, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.float32)
        last_step_theta = torch.tensor(theta, dtype=torch.float32)
        x = torch.cat([public_obs,private_obs, action, last_step_theta], dim=-1)
        z = self.encoder(x)

        # Decode next observation and updated action distribution
        next_pub_obs = self.decoder_observation(z)
        next_private_obs = self.decoder_action_dist(z)

        return z, next_pub_obs, next_private_obs

    def loss_calculator(self, pred_next_pub_obs, next_state, pred_act_config, true_act_config, alpha):
        pred_next_pub_obs = torch.tensor(pred_next_pub_obs, dtype=torch.float32, requires_grad=True)
        next_state = torch.tensor(next_state, dtype=torch.float32, requires_grad=True)
        pred_act_config = torch.tensor(pred_act_config, dtype=torch.float32, requires_grad=True)
        true_act_config = torch.tensor(true_act_config, dtype=torch.float32, requires_grad=True)
        alpha = torch.tensor(alpha, dtype=torch.float32)
        #first component
        loss_obs = self.mse_lose(pred_next_pub_obs, next_state)
        #second component
        norm_alpha_true_ac = torch.softmax(alpha + true_act_config, dim=-1)
        dirichlet_likelihood = torch.distributions.Dirichlet(norm_alpha_true_ac)

        norm_pred_act_onfig = torch.softmax(pred_act_config, dim=-1)
        loss_theta = -dirichlet_likelihood.log_prob(norm_pred_act_onfig)
        #third component
        dir_pred = torch.distributions.Dirichlet(norm_pred_act_onfig)
        dir_true = torch.distributions.Dirichlet(norm_alpha_true_ac)
        loss_kl = torch.distributions.kl.kl_divergence(dir_pred, dir_true)

        total_loss = (loss_obs + loss_theta + loss_kl).mean()

        return total_loss

    def batch_update(self, pred_next_pub_obs, next_state, pred_act_config, true_act_config, alpha):
        self.optimizer_encdec.zero_grad()
        loss_endec = self.loss_calculator(pred_next_pub_obs, next_state, pred_act_config, true_act_config, alpha)
        loss_endec.backward()
        self.optimizer_encdec.step()
