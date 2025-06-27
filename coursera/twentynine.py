# Supervised learning: Predicting house prices

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Sample data (square footage, bedrooms, neighborhood as encoded values)
X = [[2000, 3, 1], [1500, 2, 2], [1800, 3, 3], [1200, 2, 1]]
y = [500000, 350000, 450000, 300000]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Unsupervised learning: Customer segmentation

from sklearn.cluster import KMeans
import numpy as np

# Sample customer data (number of purchases, total spending, product categories)
X = np.array([[5, 1000, 2], [10, 5000, 5], [2, 500, 1], [8, 3000, 3]])

# Create and fit the KMeans model
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Print the cluster centers and labels
print(f"Cluster Centers: {kmeans.cluster_centers_}")
print(f"Labels: {kmeans.labels_}")

# RL: Training an agent to play tic-tac-toe

import numpy as np

# Initialize Q-table with zeros for all state-action pairs
Q_table = np.zeros((9, 9))  # 9 possible states (board positions) and 9 possible actions

# Learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Sample function to select action using epsilon-greedy policy
def epsilon_greedy_action(state, Q_table, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, 9)  # Random action (explore)
    else:
        return np.argmax(Q_table[state])  # Best action (exploit)

# Update Q-values after each game (simplified example)
def update_q_table(state, action, reward, next_state, Q_table):
    Q_table[state, action] = Q_table[state, action] + alpha * (
        reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action]
    )

# Example simulation of a game where the agent learns
for episode in range(1000):
    state = np.random.randint(0, 9)  # Random initial state
    done = False
    while not done:
        action = epsilon_greedy_action(state, Q_table, epsilon)
        next_state = np.random.randint(0, 9)  # Simulate next state
        reward = 1 if next_state == 'win' else -1 if next_state == 'loss' else 0  # Simulate rewards
        update_q_table(state, action, reward, next_state, Q_table)
        state = next_state
        if reward != 0:
            done = True  # End the game if win/loss