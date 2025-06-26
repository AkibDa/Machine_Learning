import numpy as np

# Define the grid size and number of states
grid_size = 5
n_states = grid_size * grid_size

# Define the reward structure: -1 for all states, +10 for the goal, -10 for the pitfall
rewards = np.full((n_states,), -1)  # Default reward of -1
rewards[24] = 10  # Goal state at position 24 (bottom-right)
rewards[12] = -10  # Pitfall at position 12 (center)

# Define the number of actions (up, down, left, right)
n_actions = 4

def epsilon_greedy_action(Q_table, state, epsilon):
    # Epsilon-greedy strategy: with probability epsilon, take a random action (exploration)
    # otherwise take the action with the highest Q-value for the given state (exploitation)
    if np.random.rand() < epsilon:  # Exploration
        return np.random.randint(0, Q_table.shape[1])  # Random action
    else:  # Exploitation
        return np.argmax(Q_table[state])  # Action with the highest Q-value


alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate for epsilon-greedy policy

# Initialize the Q-table
Q_table = np.zeros((n_states, n_actions))

# Training loop
for episode in range(1000):
    state = np.random.randint(0, n_states)  # Start at random state
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)  # Random next state
        reward = rewards[next_state]

        # Q-learning update rule
        Q_table[state, action] = Q_table[state, action] + alpha * (reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action])

        state = next_state
        if next_state == 24 or next_state == 12:  # End episode if goal or pitfall is reached
            done = True