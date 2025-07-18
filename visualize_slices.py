import pyvista as pv
import os
import glob
import argparse
import yaml
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def create_custom_cmap():
    """Create a custom colormap that highlights melting temperatures"""
    colors = [
        (0, 0, 1),    # Blue (cold)
        (0, 1, 1),     # Cyan 
        (0, 1, 0),     # Green
        (1, 1, 0),     # Yellow
        (1, 0.5, 0),   # Orange
        (1, 0, 0),     # Red (hot)
        (1, 0, 1)      # Magenta (very hot)
    ]
    return LinearSegmentedColormap.from_list("custom", colors)

def create_slice_plotter(meshes, config, mode='both'):
    plotter = pv.Plotter()
    cmap = create_custom_cmap()
    melting_temp = config['material']['melting_temperature']
    
    # Get temperature range
    temp_min = min(m['temperature'].min() for m in meshes)
    temp_max = max(m['temperature'].max() for m in meshes)
    
    # Adjust range to highlight melting region
    clim = [temp_min, max(temp_max, melting_temp * 1.2)]
    
    def update_mesh(value):
        idx = int(value)
        mesh = meshes[idx]
        
        plotter.clear()
        
        # Add temperature slice
        if mode in ['x', 'both']:
            plotter.add_mesh_slice(
                mesh,
                normal='x',
                scalars='temperature',
                clim=clim,
                cmap=cmap,
                show_edges=False
            )
        
        if mode in ['y', 'both']:
            plotter.add_mesh_slice(
                mesh,
                normal='y',
                scalars='temperature',
                clim=clim,
                cmap=cmap,
                show_edges=False
            )
        
        # Add melt pool outline if above melting temp
        melted = mesh.threshold(value=melting_temp, scalars='temperature')
        if melted.n_points > 0:
            plotter.add_mesh(
                melted.outline(),
                color='white',
                line_width=3,
                name='melt_pool'
            )
        
        # Add colorbar
        plotter.add_scalar_bar(
            title="Temperature (K)",
            n_labels=5,
            position_x=0.8,
            width=0.1
        )
    
    # Initial setup
    update_mesh(0)
    
    # Add slider widget with lambda to ignore widget parameter
    plotter.add_slider_widget(
        callback=lambda v: update_mesh(v),
        rng=[0, len(meshes)-1],
        title='Time Step',
        value=0
    )
    
    return plotter

def visualize_slices(mode='both'):
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Find all output files
    files = sorted(glob.glob('output/timestep_*.vtr'))
    if not files:
        print("No output files found in output/ directory")
        return
    
    # Read all meshes
    meshes = [pv.read(file) for file in files]
    
    # Create plotter
    plotter = create_slice_plotter(meshes, config, mode)
    plotter.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['x', 'y', 'both'], 
                       default='both', help='Slice mode (x, y, or both)')
    args = parser.parse_args()
    
    visualize_slices(args.mode)
