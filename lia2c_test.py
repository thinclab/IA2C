import torch
import torch.optim as optim
from a2c_nets import Actor, Critic, compute_loss
from Org import Org  # Import your environment class from Org.py

# Hyperparameters and configurations
input_dim = 6 
hidden_dim = 128
action_dim = 5  
num_agents = 3 

# Initialize the networks
actors = [Actor(input_dim, hidden_dim, action_dim) for _ in range(num_agents)]
critics = [Critic(input_dim, hidden_dim, action_dim) for _ in range(num_agents)]

actor_optimizers = [optim.Adam(actors[i].parameters(), lr=1e-3) for i in range(num_agents)]
critic_optimizers = [optim.Adam(critics[i].parameters(), lr=1e-4) for i in range(num_agents)]

alpha = 1.0  
C = 10.0  

# Initialize environment
env = Org()  # Create an instance of your environment

# Training loop
num_steps = 10000  

def init_hidden_state(hidden_dim, num_agents):
    h_0 = torch.zeros(1, num_agents, hidden_dim)
    c_0 = torch.zeros(1, num_agents, hidden_dim)
    return (h_0, c_0)

for step in range(num_steps):
    # Adjusted to unpack what reset() returns
    public_obs, _ = env.reset()
    private_obs = public_obs  # Assuming private_obs is the same as public_obs initially
    action = [torch.zeros(action_dim) for _ in range(num_agents)]
    action_dist = [torch.ones(action_dim) / action_dim for _ in range(num_agents)]
    
    # Convert observations to PyTorch tensors
    public_obs = torch.tensor(public_obs, dtype=torch.float32)
    private_obs = torch.tensor(private_obs, dtype=torch.float32)
    
    done = [False] * num_agents
    hidden_states = [init_hidden_state(hidden_dim, num_agents) for _ in range(num_agents)]

    while not all(done):
        action_probs = []
        new_hidden_states = []
        latents = []

        for i in range(num_agents):
            action_prob, latent, new_hidden_state = actors[i](public_obs, private_obs, action[i], action_dist[i], hidden_states[i])
            action_probs.append(action_prob)
            latents.append(latent)
            new_hidden_states.append(new_hidden_state)

        actions = [torch.multinomial(action_prob, 1) for action_prob in action_probs]
        
        actions = [action.item() for action in actions]
        next_public_obs, rewards, done, _ = env.step(actions)
        next_private_obs = next_public_obs  # Assuming private_obs updates similarly

        # Convert next observations to PyTorch tensors
        next_public_obs = torch.tensor(next_public_obs, dtype=torch.float32)
        next_private_obs = torch.tensor(next_private_obs, dtype=torch.float32)

        next_action_dist = [None] * num_agents
        for i in range(num_agents):
            value, next_obs_pred, next_action_dist_pred, new_hidden_state = critics[i](public_obs, private_obs, actions[i], action_dist[i], new_hidden_states[i])
            hidden_states[i] = new_hidden_state

            # Compute the LIA2C loss
            loss = compute_loss(
                observed_obs=next_public_obs,
                predicted_obs=next_obs_pred,
                observed_action_dist=action_dist[i],
                predicted_action_dist=next_action_dist_pred,
                alpha=alpha,
                C=C
            )

            # Optimize the networks
            actor_optimizers[i].zero_grad()
            critic_optimizers[i].zero_grad()
            loss.backward()
            actor_optimizers[i].step()
            critic_optimizers[i].step()

        # Update observations and action distributions
        public_obs, private_obs, action_dist = next_public_obs, next_private_obs, next_action_dist
