import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


def visualize_sample_as_gif(pt_file_path, sample_idx=0, save_path="visualization.gif"):
    # Load the PyTorch tensor
    print(f"Loading data from {pt_file_path}")
    data = torch.load(pt_file_path, map_location=torch.device('cpu'))
    # Check if it's the expected format
    if not isinstance(data, torch.Tensor):
        print("Warning: Loaded data is not a tensor. Attempting to extract tensor...")
        # Try to find tensor in dictionary if it's not directly a tensor
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor) and len(value.shape) == 5:
                    data = value
                    print(f"Found tensor with key: {key}")
                    break
    if len(data.shape) != 5:
        raise ValueError(f"Expected tensor with 5 dimensions, got shape {data.shape}")
    samples, time_steps, channels, height, width = data.shape
    print(f"Tensor shape: {data.shape}")
    if channels != 2:
        print(f"Warning: Expected 2 channels, but got {channels}")
    if sample_idx >= samples:
        raise ValueError(f"Sample index {sample_idx} out of range (0-{samples - 1})")
    # Extract the sample
    sample_data = data[sample_idx].cpu().numpy()
    # Determine global min and max for consistent color scaling
    vmin = 0.5
    vmax = 0.8
    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Sample {sample_idx}: Visualization of Both Channels")
    # Initial frame setup
    imgs = []
    for i in range(2):
        imgs.append(axes[i].imshow(sample_data[0, i], cmap='RdYlBu',
                                   vmin=vmin, vmax=vmax))
        axes[i].set_title(f"Channel {i}")
        axes[i].axis('off')
    # Add a colorbar
    cbar = fig.colorbar(imgs[0], ax=axes, orientation='horizontal', fraction=0.05, pad=0.08)
    # Animation update function
    def update(frame):
        for i in range(2):
            imgs[i].set_array(sample_data[frame, i])
        fig.suptitle(f"Sample {sample_idx}: Time Step {frame}/{time_steps - 1}")
        return imgs
    # Create the animation
    print(f"Creating animation with {time_steps} frames...")
    anim = FuncAnimation(fig, update, frames=time_steps, blit=False)
    # Save as GIF
    print(f"Saving animation to {save_path}")
    anim.save(save_path, writer='pillow', fps=10, dpi=100)
    plt.close(fig)
    print(f"Animation saved to {save_path}")
    return True
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize a sample from PyTorch tensor data as GIF")
    parser.add_argument("pt_file", type=str, help="Path to the .pt file")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to visualize (default: 0)")
    parser.add_argument("--output", type=str, default="visualization.gif", help="Output GIF file path")

    args = parser.parse_args()

    try:
        visualize_sample_as_gif(args.pt_file, args.sample, args.output)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0
if __name__ == "__main__":
    exit(main())
