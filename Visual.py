from pathlib import Path
import torch
import matplotlib.pyplot as plt
import config

from training.model import PongMLP
from training.rl_model import PongRLMLP


# -----------------------
# LOAD MODEL CORRECTLY
# -----------------------
model_path = config.REINFORCED_LEFT_MODEL_FILE  # or RIGHT_MODEL_FILE

checkpoint = torch.load(model_path, map_location="cpu")

# 🔥 Handle both formats (old + new)
if "hidden_sizes" in checkpoint:
    hidden_sizes = checkpoint["hidden_sizes"]
elif "hidden_size" in checkpoint:
    hidden_sizes = [checkpoint["hidden_size"]]
else:
    raise KeyError("Model missing hidden size info")

model = PongMLP(
    input_size=checkpoint["input_size"],
    hidden_sizes=hidden_sizes
)
# 🔥 FIXED: use hidden_sizes
if(model_path == config.REINFORCED_LEFT_MODEL_FILE): 
    model = PongRLMLP(
        input_size=checkpoint["input_size"]
    )



model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model structure:", model)


# -----------------------
# EXTRACT ALL WEIGHTS
# -----------------------
weights = []
for layer in model.net:
    if isinstance(layer, torch.nn.Linear):
        weights.append(layer.weight.detach().numpy())


# -----------------------
# DRAW NETWORK (GENERIC)
# -----------------------
def draw_network(weights):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # layer sizes
    layer_sizes = [weights[0].shape[1]]
    for w in weights:
        layer_sizes.append(w.shape[0])

    print("Layer sizes:", layer_sizes)

    h_spacing = 2
    v_spacing = 0.8

    neuron_positions = []

    # 🔥 limit visualization size (important)
    MAX_NEURONS = 20

    for i, size in enumerate(layer_sizes):
        size = min(size, MAX_NEURONS)

        layer = []
        y_offset = (size - 1) * v_spacing / 2

        for j in range(size):
            x = i * h_spacing
            y = j * v_spacing - y_offset
            layer.append((x, y))

        neuron_positions.append(layer)

    # -----------------------
    # DRAW CONNECTIONS
    # -----------------------
# -----------------------
# DRAW CONNECTIONS
# -----------------------
    for layer_idx, w in enumerate(weights):
        prev_layer = neuron_positions[layer_idx]
        next_layer = neuron_positions[layer_idx + 1]

        max_w = abs(w).max() + 1e-6

        for i, (x1, y1) in enumerate(prev_layer):
            for j, (x2, y2) in enumerate(next_layer):

                if j >= w.shape[0] or i >= w.shape[1]:
                    continue

                weight = w[j][i]

                # 🔥 normalize for transparency
                norm = abs(weight) / max_w

                # 🔥 smoother visibility curve
                alpha = 0.05 + norm * 0.6

                color = "blue" if weight > 0 else "red"

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=color,
                    alpha=alpha,
                    linewidth=1  # 🔥 constant thickness
                )

    # -----------------------
    # DRAW NEURONS
    # -----------------------
    for layer in neuron_positions:
        for (x, y) in layer:
            circle = plt.Circle((x, y), 0.12, color='black')
            ax.add_patch(circle)

    plt.title("Pong AI Neural Network")
    plt.show()


# -----------------------
# RUN
# -----------------------
draw_network(weights)