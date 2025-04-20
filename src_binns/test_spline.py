import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
MODEL_TYPE = "kan"

def true_fn(x):
    output = np.where(x > 0.5, 4*x**2, 1.0)
    return output

# Generate training data
def generate_data(n_samples, x_range, true_fn):
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    eps = np.random.normal(0, 0.02, n_samples)
    y = true_fn(x) + eps
    return x, y

# Generate training set
n_train_samples = 100
x_train, y_train = generate_data(n_train_samples, [0.2, 0.8], true_fn)
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# Create a DataLoader for mini-batch training
batch_size = 10
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Generate test set with 4,000 evenly spaced points between [-2, 2]
x_test = np.linspace(-2, 4, 1000)
y_test = true_fn(x_test)  # True function without noise
x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def predict_summed(func, input):
    # Returns predicted parameters (post-sigmoid), summed across the batch dim
    return func(input).sum(dim=0)


def get_jacobian(model, input, noise_std=0):
    """
    Returns Jacobian, of shape [batch, n_param, n_input].
    For each batch item, it is dParam/dInput.

    If input is not provided, assume something is cached in self.new_input
    """
    if noise_std > 0:
        input += torch.randn(input.shape) * input.std(dim=0, keepdim=True) * noise_std

    # Using jacrev, summing the outputs across batch as each example's output
    # only depends on that example's input
    batch_jacobian1 = torch.func.jacrev(predict_summed, argnums=1)(model, input)  # [n_outputs, batch, n_inputs]
    batch_jacobian1 = batch_jacobian1.permute((1, 0, 2))  # [batch, n_outputs, n_inputs]
    return batch_jacobian1


# Initialize the model, loss function, and optimizer
if MODEL_TYPE == "nam":
    model = SimpleNN()
elif MODEL_TYPE == "kan":
    import kan
    model = kan.KAN(width=[1, 1], grid=10, k=3, grid_eps=1, grid_margin=1.0, seed=42, base_fun="silu_identity", affine_trainable=True)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.1, weight_decay=1e-4)

# Create a figure and axis for plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-2, 4)
ax.set_ylim(-2, 4)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Model Predictions Over Training Epochs')
ax.grid(True)

# Scatter plot for training data
scatter = ax.scatter(x_train.numpy(), y_train.numpy(), label='Train Data', color='blue', alpha=0.5)

# Line plot for true function
true_line, = ax.plot(x_test.numpy(), y_test.numpy(), label='True Function ($y = x^2$)', color='green', linestyle='--', linewidth=2)

# Line plot for model predictions
pred_line, = ax.plot([], [], label='Model Predictions', color='red', linewidth=2)
if MODEL_TYPE == "kan":
    base_function, = ax.plot([], [], label='Base function', color='brown', linewidth=1)
    basis_functions = [ax.plot([], [], label=f'Basis {i}', color='gray', linewidth=1)[0] for i in range(model.grid + model.k)]

# Annotation for epoch number
epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

# Initialize the line data
def init():
    pred_line.set_data([], [])
    epoch_text.set_text('')
    return pred_line, epoch_text

# Update function for animation
def update(epoch):
    model.train()
    ibatch = 0
    for batch_x, batch_y in train_loader:
        if MODEL_TYPE == "kan" and ibatch == 0 and epoch < 10:
            # KAN-specific: update grid
            model.update_grid(batch_x)
            print("New grid", model.act_fun[0].grid)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        if MODEL_TYPE == "kan":
            # KAN-specific: coefficient smoothness penalty
            diff_penalty = model.reg(reg_metric="edge_backward", lamb_l1=0, lamb_entropy=0, lamb_coef=0, lamb_coefdiff=1)
            if epoch % 5 == 0 and ibatch == 0:
                print(f"Epoch {epoch}: Sup loss {loss.item():.4f}, Diff penalty {diff_penalty.item():.4f}. Coef {model.act_fun[0].coef}")
                print("Silu input offset", model.act_fun[0].silu_input_offset, "Silu input scale", model.act_fun[0].silu_input_scale, "Silu scale", model.act_fun[0].scale_silu, "Identity scale", model.act_fun[0].scale_base)
            loss += diff_penalty * 0.1 # + 0.001*jacobian

        loss.backward()
        optimizer.step()
        ibatch += 1

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test)

        if MODEL_TYPE == "kan":
            # Get values of the basis functions
            base_fun_output = model.act_fun[0].compute_base_fun(x_test)
            from kan.spline import B_batch
            basis_test = B_batch(x_test, model.act_fun[0].grid, k=3)  # [batch, 1, num_splines] where 1 is the number of inputs
            basis_test = basis_test.squeeze(1) * model.act_fun[0].coef[0, 0, :]

            # Plot the basis function
            base_function.set_data(x_test.numpy(), base_fun_output.numpy())
            for i in range(basis_test.shape[1]):
                basis_functions[i].set_data(x_test.numpy(), basis_test[:, i].numpy())
                # basis_functions[i].set_linewidth(model.act_fun[0].coef[0, 0, i])

    # Update the prediction line
    pred_line.set_data(x_test.numpy(), y_pred_test.numpy())

    # Update the epoch text
    epoch_text.set_text(f'Epoch: {epoch + 1}')
    
    return pred_line, epoch_text

# Create the animation
n_epochs = 200
ani = FuncAnimation(fig, update, frames=n_epochs, init_func=init, blit=True, interval=50, repeat=False)

# Add a legend
ax.legend(loc='upper right')

# Show the animation
plt.show()

# To save the animation as a video file (optional)
ani.save('kan_test_hybridbase.mp4', writer='ffmpeg', fps=5)