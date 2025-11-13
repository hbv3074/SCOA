import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Step 1: Load Dataset
# ----------------------------
df = pd.read_csv("SCOA_A7.csv")

# Select numerical features for clustering
data = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values.astype(float)

# ----------------------------
# Step 2: PSO Parameters
# ----------------------------
num_particles = 30
num_iterations = 100
num_clusters = 4  # You can adjust this
w = 0.5       # inertia weight
c1 = 1.5      # cognitive coefficient
c2 = 1.5      # social coefficient

# ----------------------------
# Step 3: Particle Class
# ----------------------------
class Particle:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters
        self.position = data[np.random.choice(range(len(data)), num_clusters)].astype(float)
        self.velocity = np.zeros_like(self.position, dtype=float)
        self.best_position = np.copy(self.position)
        self.best_score = self.evaluate()

    def evaluate(self):
        distances = np.linalg.norm(self.data[:, None] - self.position[None, :], axis=2)
        closest = np.argmin(distances, axis=1)
        score = sum(np.linalg.norm(self.data[i] - self.position[closest[i]])**2 for i in range(len(self.data)))
        return score

    def update(self, global_best):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social
        self.position = self.position + self.velocity  # Ensure float operation
        score = self.evaluate()
        if score < self.best_score:
            self.best_score = score
            self.best_position = np.copy(self.position)

# ----------------------------
# Step 4: Initialize Swarm
# ----------------------------
swarm = [Particle(data, num_clusters) for _ in range(num_particles)]
global_best = min(swarm, key=lambda p: p.best_score).best_position

# ----------------------------
# Step 5: PSO Loop
# ----------------------------
for _ in range(num_iterations):
    for particle in swarm:
        particle.update(global_best)
    global_best = min(swarm, key=lambda p: p.best_score).best_position

# ----------------------------
# Step 6: Final Clustering
# ----------------------------
distances = np.linalg.norm(data[:, None] - global_best[None, :], axis=2)
labels = np.argmin(distances, axis=1)

# ----------------------------
# Step 7: Visualization
# ----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 1], data[:, 2], c=labels, cmap='viridis')
plt.scatter(global_best[:, 1], global_best[:, 2], c='red', marker='x', s=200)
plt.title("PSO-based Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()
