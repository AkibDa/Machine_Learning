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