import OpenOrg
from OpenOrg import OrgEnvironment
import LIA2CAgent
from LIA2CAgent import LIA2CAgent
import random
import numpy as np
def train_agent(num_episodes=2000, max_steps=100):
    
    env = OrgEnvironment(num_employees=10, noise_level=0.1)
    observation_size = len(OpenOrg.OBSERVATIONS)  
    action_size = len(OpenOrg.EMPLOYEE_ACTIONS)
    latent_size = 16

    agent = LIA2CAgent(observation_size, action_size, latent_size)

    for episode in range(num_episodes):
        observations = {}
        rewards = {}
        dones = {}
        log_probs = {}
        trajectory = []

        # Initialize observations for all agents
        public_obs = env.get_public_observation()
        for agent_name in env.agents:
            observations[agent_name] = {'public': public_obs, 'private': None}
            rewards[agent_name] = 0
            dones[agent_name] = False
            log_probs[agent_name] = []

        for step in range(max_steps):
            actions = {}

            # Employees select actions
            for emp in env.employees:
                if not dones[emp]:
                    obs_vector = observation_to_vector(observations[emp]['public'])
                    action_index, log_prob = agent.select_action(obs_vector)
                    actions[emp] = OpenOrg.EMPLOYEE_ACTIONS[action_index]
                    log_probs[emp].append(log_prob)
                else:
                    actions[emp] = None

            # Moustapha Implement Manager Actions
            actions[env.manager] = random.choice(OpenOrg.MANAGER_ACTIONS)
            actions[env.manager] = 'balance'
            #print(f"managerACtion={actions[env.manager]}")
            # Environment step
            next_observations, step_rewards, done = env.step(actions)

            # Store trajectory for agent update
            for emp in env.employees:
                if not dones[emp]:
                    next_obs_vector = observation_to_vector(next_observations[emp]['public'])
                    reward = step_rewards[emp]
                    trajectory.append((obs_vector, action_index, reward, next_obs_vector, done))

            # Update observations and rewards
            observations = next_observations
            for agent_name in env.agents:
                rewards[agent_name] += step_rewards[agent_name]
                dones[agent_name] = done

            if all(dones.values()):
                break

        # Update agent after each episode
        agent.update(trajectory)

        
        total_reward = sum(rewards.values())
        print(f'Episode {episode}, Total Reward: {total_reward}')
        print(f"actions={actions}")
        ##Look at the losses of the encoder-decoder
def observation_to_vector(observation):
    """Converts observation to a one-hot encoded vector."""
    vector = np.zeros(len(OpenOrg.OBSERVATIONS))
    index = OpenOrg.OBSERVATIONS.index(observation)
    vector[index] = 1
    return vector

def action_to_one_hot(action_index):
    """Converts action index to one-hot encoded vector."""
    vector = np.zeros(len(OpenOrg.EMPLOYEE_ACTIONS))
    vector[action_index] = 1
    return vector

if __name__ == '__main__':
    train_agent()
