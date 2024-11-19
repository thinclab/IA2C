import OpenOrg
from OpenOrg import OrgEnvironment
import LIA2CAgent
from LIA2CAgent import LIA2CAgent
import random
import numpy as np
from collections import defaultdict
import torch


def train_agent(num_episodes=1000, max_steps=100):
    env = OrgEnvironment(num_employees=10, noise_level=0.1)
    observation_size = len(OpenOrg.OBSERVATIONS)  # One-hot encoding of observations
    employee_action_size = len(OpenOrg.EMPLOYEE_ACTIONS)
    manager_action_size = len(OpenOrg.MANAGER_ACTIONS)
    private_obs_size = employee_action_size * (env.num_employees - 1)  # For employees
    manager_private_obs_size = employee_action_size * env.num_employees  # For manager observing employees
    latent_size = 16


    # Create a LIA2CAgent instance for each employee
    agents = {emp: LIA2CAgent(observation_size, employee_action_size, latent_size, private_obs_size) for emp in env.employees}
    # Create a LIA2CAgent instance for the manager
    manager_agent = LIA2CAgent(observation_size, manager_action_size, latent_size, manager_private_obs_size)


    for episode in range(num_episodes):
        total_rewards = defaultdict(float)
        trajectories = {emp: [] for emp in env.employees}
        manager_trajectory = []
        dones = {emp: False for emp in env.employees}
        manager_done = False


        # Initialize the environment
        env.state = 'medium'  # Reset environment state


        # Initialize observations
        observations = {}
        obs_vectors = {}


        for step in range(max_steps):
            actions = {}


            # Employee agents select actions
            for emp in env.employees:
                if not dones[emp]:
                    agent = agents[emp]
                    # Get current observation
                    if step == 0:
                        public_obs = env.get_public_observation()
                        obs_vector = observation_to_vector(public_obs)
                        private_obs_vector = np.zeros(private_obs_size)
                        combined_obs_vector = np.concatenate([obs_vector, private_obs_vector])
                        obs_vectors[emp] = combined_obs_vector
                    else:
                        combined_obs_vector = obs_vectors[emp]


                    # Select action
                    action_index, _ = agent.select_action(combined_obs_vector[:observation_size])
                    action = OpenOrg.EMPLOYEE_ACTIONS[action_index]
                    actions[emp] = action
                else:
                    actions[emp] = 'resign'  # Or any default action


            # Manager agent selects action
            if not manager_done:
                if step == 0:
                    public_obs = env.get_public_observation()
                    manager_obs_vector = observation_to_vector(public_obs)
                    manager_private_obs_vector = np.zeros(manager_private_obs_size)
                    manager_combined_obs_vector = np.concatenate([manager_obs_vector, manager_private_obs_vector])
                else:
                    manager_combined_obs_vector = manager_obs_vector


                # Select action
                manager_action_index, _ = manager_agent.select_action(manager_combined_obs_vector[:observation_size])
                manager_action = OpenOrg.MANAGER_ACTIONS[manager_action_index]
                actions[env.manager] = manager_action
            else:
                actions[env.manager] = 'self'  # Default action


            # Environment step
            next_observations, step_rewards, done = env.step(actions)


            # Update total rewards and dones
            for emp in env.employees:
                agent = agents[emp]
                if not dones[emp]:
                    reward = step_rewards[emp]
                    total_rewards[emp] += reward


                    # Get next observations
                    next_public_obs = next_observations[emp]['public']
                    next_private_observations = next_observations[emp]['private']


                    # Convert observations to vectors
                    next_obs_vector = observation_to_vector(next_public_obs)
                    next_private_obs_vector = private_observations_to_vector(next_private_observations, OpenOrg.EMPLOYEE_ACTIONS)
                    next_combined_obs_vector = np.concatenate([next_obs_vector, next_private_obs_vector])


                    # Store transition in the agent's trajectory
                    obs_vector = obs_vectors[emp][:observation_size]
                    private_obs_vector = obs_vectors[emp][observation_size:]
                    action_index = OpenOrg.EMPLOYEE_ACTIONS.index(actions[emp])
                    done_flag = (actions[emp] == 'resign') or done
                    trajectories[emp].append((obs_vector, action_index, private_obs_vector, reward, next_obs_vector, done_flag))


                    # Update observations for the agent
                    obs_vectors[emp] = next_combined_obs_vector


                    # Check if the agent is done
                    if actions[emp] == 'resign' or done:
                        dones[emp] = True


            # Manager updates
            if not manager_done:
                manager_reward = step_rewards[env.manager]
                # Get next observations
                next_public_obs = next_observations[env.manager]['public']
                manager_next_private_observations = {}
                for emp in env.employees:
                    if actions[emp] != 'resign':
                        manager_next_private_observations[emp] = actions[emp]


                # Convert observations to vectors
                manager_next_obs_vector = observation_to_vector(next_public_obs)
                manager_next_private_obs_vector = private_observations_to_vector(manager_next_private_observations, OpenOrg.EMPLOYEE_ACTIONS)
                manager_next_combined_obs_vector = np.concatenate([manager_next_obs_vector, manager_next_private_obs_vector])


                # Store transition in manager's trajectory
                manager_obs_vector = manager_combined_obs_vector[:observation_size]
                manager_private_obs_vector = manager_combined_obs_vector[observation_size:]
                manager_action_index = OpenOrg.MANAGER_ACTIONS.index(actions[env.manager])
                manager_done_flag = done
                manager_trajectory.append((manager_obs_vector, manager_action_index, manager_private_obs_vector, manager_reward, manager_next_obs_vector, manager_done_flag))


                # Update observations for the manager
                manager_combined_obs_vector = manager_next_combined_obs_vector


                if done:
                    manager_done = True


            if all(dones.values()) and manager_done:
                break


        # Update each agent after the episode
        for emp in env.employees:
            agent = agents[emp]
            trajectory = trajectories[emp]
            if trajectory:
                #print(f"agent Trajecetory = {trajectory[0][2]}")
                agent.update(trajectory)


        # Update manager agent
        if manager_trajectory:
           
            #print(f"manager Trajecetory = {manager_trajectory[0][2]}")
            if not all(len(step) == 6 and len(step[2]) > 0 for step in manager_trajectory):
                print(1)
                #print(f"Invalid manager trajectory: {manager_trajectory}")
            else:  
                manager_agent.update(manager_trajectory)


        # Logging
        avg_total_reward = sum(total_rewards.values()) / len(env.employees)
        manager_total_reward = sum([step[3] for step in manager_trajectory])
        print(f'Episode {episode + 1}, Average Employee Reward: {avg_total_reward}, Total Reward: {sum(total_rewards.values())} Manager Reward: {manager_total_reward}, #Employees: {len(env.employees)}')


    print("\nFinal Policies:")
    for emp in env.employees:
        if not dones[emp]:
            agent = agents[emp]
            observation = obs_vectors[emp][:observation_size]
            policy = agent.get_policy(observation)
            print(f"Policy for {emp}: {policy}")


    manager_observation = manager_combined_obs_vector[:observation_size]
    manager_policy = manager_agent.get_policy(manager_observation)
    print(f"Policy for Manager: {manager_policy}")


    # Count and show true employees
    true_employees = [emp for emp, done in dones.items() if not done]
    print(f"\nNumber of true employees: {len(true_employees)}")
    print(f"Active employees: {true_employees}")
   
def observation_to_vector(observation):
    """Converts public observation to a one-hot encoded vector."""
    vector = np.zeros(len(OpenOrg.OBSERVATIONS))
    index = OpenOrg.OBSERVATIONS.index(observation)
    vector[index] = 1
    return vector


def action_to_vector(action, action_space):
    """Converts action to a one-hot encoded vector."""
    vector = np.zeros(len(action_space))
    index = action_space.index(action)
    vector[index] = 1
    return vector


def private_observations_to_vector(private_observations, action_space):
    vectors = []
    for agent_name in sorted(private_observations.keys()):
        if agent_name not in private_observations:
            #print(f"Warning: Missing private observation for {agent_name}")
            continue
        action = private_observations[agent_name]
        action_vector = action_to_vector(action, action_space)
        vectors.append(action_vector)
    if not vectors:  # Handle case where no valid vectors are found
        #rint("Warning: No valid private observations found, returning zero vector.")
        return np.zeros(len(action_space))  # Default to zero vector actually replace
    combined_vector = np.concatenate(vectors)
    return combined_vector


def get_policy(self, observation):
    """
    Returns the action probabilities for a given observation.
    """
    observation_tensor = torch.FloatTensor(observation).unsqueeze(0)
    action_probs = self.actor_critic.actor(observation_tensor)
    return action_probs.squeeze().detach().numpy()


if __name__ == '__main__':
    train_agent()



