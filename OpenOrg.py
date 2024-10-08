import numpy as np
import random



# possible states and observations
STATES = ['very_low', 'low', 'medium', 'high', 'very_high']
OBSERVATIONS = ['meager', 'several', 'many']

# actions for employees and managers
EMPLOYEE_ACTIONS = ['self', 'balance', 'group', 'resign']
MANAGER_ACTIONS = ['self', 'balance', 'group', 'fire', 'hire']

# observations to states
STATE_OBS_MAP = {
    'very_low': 'meager',
    'low': 'meager',
    'medium': 'several',
    'high': 'several',
    'very_high': 'many'
}

# rewards
INDIVIDUAL_REWARDS = {
    'self': 3,
    'balance': 1,
    'group': 0,
    'resign': 0,
    'fire': 0,
    'hire': 0
}

GROUP_REWARDS = {
    'self': 0,
    'balance': 1,
    'group': 2,
    'resign': 0,
    'fire': 0,
    'hire': 0
}

# State transition 
STATE_TRANSITIONS = {
    # (current_state, net_action_effect): next_state
    ('very_low', 'decrease'): 'very_low',
    ('very_low', 'stable'): 'low',
    ('very_low', 'increase'): 'medium',
    
    ('low', 'decrease'): 'very_low',
    ('low', 'stable'): 'medium',
    ('low', 'increase'): 'high',

    ('medium', 'decrease'): 'low',
    ('medium', 'stable'): 'medium',
    ('medium', 'increase'): 'high',

    ('high', 'decrease'): 'medium',
    ('high', 'stable'): 'high',
    ('high', 'increase'): 'very_high',

    ('very_high', 'decrease'): 'high',
    ('very_high', 'stable'): 'very_high',
    ('very_high', 'increase'): 'very_high',
}

class OrgEnvironment:
    def __init__(self, num_employees=10, noise_level=0.1):
        self.num_employees = num_employees
        self.employees = ['employee_{}'.format(i) for i in range(num_employees)]
        self.manager = 'manager'
        self.agents = self.employees + [self.manager]
        self.state = 'medium'  
        self.noise_level = noise_level  

    def get_public_observation(self):
        """Returns the public observation based on the current state."""
        return STATE_OBS_MAP[self.state]

    def get_private_observation(self, true_action):
        """Returns a noisy private observation of an action."""
        if random.random() < (1 - self.noise_level):
            return true_action  
        else:
            return random.choice(EMPLOYEE_ACTIONS)  

    def step(self, actions):
        """
        Executes a step in the environment.
        actions: dict mapping agent names to their actions
        Returns:
            observations: dict mapping agent names to their observations
            rewards: dict mapping agent names to their rewards
            done: whether the episode is over
        """
        observations = {}
        rewards = {}

        # Calculate net effect of actions on state
        net_action_effect = self.calculate_net_action_effect(actions)

        # State transition
        self.state = self.get_next_state(net_action_effect)

        # Public observation
        public_obs = self.get_public_observation()

        # Generate observations and rewards for each agent
        for agent in self.agents:
            if agent.startswith('employee'):
                true_action = actions[agent]
                private_obs = self.get_private_observation(true_action)
                observations[agent] = {'public': public_obs, 'private': private_obs}
                rewards[agent] = INDIVIDUAL_REWARDS[true_action] + self.get_group_reward(actions)
            elif agent == self.manager:
                # Manager's observations and rewards
                observations[agent] = {'public': public_obs, 'private': None}
                rewards[agent] = self.calculate_manager_reward(actions)

        # Check if episode is done (you can define your own termination condition)
        done = False

        return observations, rewards, done

    def calculate_net_action_effect(self, actions):
        """
        Calculates the net effect of all actions on the organization's state.
        Returns 'increase', 'decrease', or 'stable'.
        """
        group_actions = sum(1 for action in actions.values() if action == 'group')
        self_actions = sum(1 for action in actions.values() if action == 'self')

        if group_actions > self_actions:
            return 'increase'
        elif group_actions < self_actions:
            return 'decrease'
        else:
            return 'stable'

    def get_next_state(self, net_action_effect):
       
        key = (self.state, net_action_effect)
        return STATE_TRANSITIONS.get(key, self.state)

    def get_group_reward(self, actions):
       
        group_reward = 0
        for action in actions.values():
            group_reward += GROUP_REWARDS[action]
        return group_reward / len(self.agents)  

    def calculate_manager_reward(self, actions):
        
        individual_rewards = sum(INDIVIDUAL_REWARDS[actions[agent]] for agent in self.employees)
        group_reward = self.get_group_reward(actions)
        hiring_cost = 1 * sum(1 for action in actions.values() if action == 'hire')
        # Apply sigmoid to individual rewards
        manager_reward = self.sigmoid(individual_rewards) + group_reward - hiring_cost
        return manager_reward

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
