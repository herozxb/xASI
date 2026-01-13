import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Define a simple MLP (2D input -> 1D output)
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Input 2D -> 64 units
        self.fc2 = nn.Linear(64, 1)  # 64 units -> 1D output
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU after first layer
        x = self.fc2(x)  # Output layer
        return x


def generate_circle_data(num_samples=1000):
    # Random points in the range [-1, 1] for both x and y
    x = np.random.uniform(-1, 1, (num_samples, 2))
    # Calculate the Euclidean distance from (0, 0) for each point
    distance = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    # Label points inside the unit circle as 1, outside as 0
    y = (distance <= 1).astype(np.float32)  # 1 inside the circle, 0 outside
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Generate circle data
X_train, y_train = generate_circle_data(1000)

def generate_rectangle_data(num_samples=1000):
    # Random points in the range [-0.25, 0.25] for both x and y (to form a rectangle of 0.5x0.5)
    x = np.random.uniform(-1, 1, (num_samples, 2))
    
    # Check if points are inside the rectangle (bounded by x: [-0.25, 0.25] and y: [-0.25, 0.25])
    inside_rectangle = (np.abs(x[:, 0]) <= 0.25) & (np.abs(x[:, 1]) <= 0.25)
    
    # Label points inside the rectangle as 1, outside as 0
    y = inside_rectangle.astype(np.float32)  # 1 inside, 0 outside
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Generate rectangle data
# X_train, y_train = generate_rectangle_data(1000)

# Convert the points and labels to numpy arrays for easier plotting
X_train_np = X_train.numpy()
y_train_np = y_train.numpy()

# Plot the points
plt.figure(figsize=(6, 6))

# Points inside the rectangle (y_train == 1) - plot in blue
plt.scatter(X_train_np[y_train_np[:, 0] == 1, 0], X_train_np[y_train_np[:, 0] == 1, 1], c='blue', label='Inside Rectangle', alpha=0.5)

# Points outside the rectangle (y_train == 0) - plot in red
plt.scatter(X_train_np[y_train_np[:, 0] == 0, 0], X_train_np[y_train_np[:, 0] == 0, 1], c='red', label='Outside Rectangle', alpha=0.5)

# Set axis limits
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Rectangle Data: Inside and Outside Points')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


# Instantiate the model, loss function, and optimizer
model = SimpleMLP()
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X_train)  # Forward pass
    loss = criterion(output, y_train)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update parameters
    
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")




# Create a grid of points for the heatmap (e.g., -1 to 1 in both x and y)
x_range = np.linspace(-2.5, 2.5, 1000)
y_range = np.linspace(-2.5, 2.5, 1000)
xx, yy = np.meshgrid(x_range, y_range)
grid_points = np.vstack([xx.ravel(), yy.ravel()]).T  # 2D points for the grid

# Convert grid points to tensor and get MLP predictions
grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
with torch.no_grad():
    predictions = model(grid_points_tensor).numpy()

# Reshape predictions to match the grid shape for heatmap
predictions = predictions.reshape(xx.shape)


# Plot the heatmap
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, predictions, levels=np.linspace(0, 2000, 100), cmap='coolwarm')
plt.colorbar()
plt.title("MLP Output Heatmap")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# Plot the 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(xx, yy, predictions, cmap='coolwarm', edgecolor='none')

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Prediction')
ax.set_title("MLP Output Heatmap (3D)")

# Show the plot
plt.show()

