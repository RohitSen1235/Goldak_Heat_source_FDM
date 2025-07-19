import pyvista as pv
import os
import glob
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

def visualize_results():
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    melting_temp = config['material']['melting_temperature']
    
    # Find all output files
    files = sorted(glob.glob('output/timestep_*.vtr'))
    if not files:
        print("No output files found in output/ directory")
        return
    
    # Read all meshes
    meshes = [pv.read(file) for file in files]
    cmap = create_custom_cmap()
    
    # Get temperature range
    temp_min = min(m['temperature'].min() for m in meshes)
    temp_max = max(m['temperature'].max() for m in meshes)
    clim = [temp_min, max(temp_max, melting_temp * 1.2)]
    
    # Create minimal plotter with most compatible settings
    plotter = pv.Plotter(
        notebook=False,
        window_size=[800, 600],  # Smaller window
        polygon_smoothing=False   # Disable smoothing
    )
    # Configure for stability
    pv.set_plot_theme('document')
    try:
        plotter.enable_anti_aliasing('fxaa')  # Lightweight anti-aliasing
    except:
        pass  # Skip if anti-aliasing fails
    plotter.enable_trackball_style()
    
    # Reduce memory usage
    if hasattr(pv, 'global_theme'):
        pv.global_theme.auto_close = True
    
    # Add axis references with labels
    plotter.add_axes(
        xlabel='X (mm)',
        ylabel='Y (mm)',
        zlabel='Z (mm)',
        line_width=2,
        labels_off=False
    )
    
    # Enable all interaction modes
    plotter.enable_trackball_style()
    plotter.enable_terrain_style(mouse_wheel_zooms=True)
    
    # Add boundary planes
    bounds = meshes[0].bounds
    top_plane = pv.Plane(
        center=[(bounds[1]+bounds[0])/2, (bounds[3]+bounds[2])/2, bounds[5]],
        direction=[0, 0, -1],
        i_size=bounds[1]-bounds[0],
        j_size=bounds[3]-bounds[2]
    )
    bottom_plane = pv.Plane(
        center=[(bounds[1]+bounds[0])/2, (bounds[3]+bounds[2])/2, bounds[4]],
        direction=[0, 0, 1],
        i_size=bounds[1]-bounds[0],
        j_size=bounds[3]-bounds[2]
    )
    plotter.add_mesh(top_plane, color='red', opacity=0.5, name='top_face')
    plotter.add_mesh(bottom_plane, color='blue', opacity=0.5, name='bottom_face')

    # Add all timesteps to the plotter
    for mesh in meshes:
        # Add cold region representation
        cold = mesh.threshold(value=melting_temp, scalars='temperature', invert=True)
        if cold.n_points > 0:
            # Simplify using quadric decimation if available
            try:
                cold_simple = cold.simplify_mesh(target_reduction=0.3)
                plotter.add_mesh(
                    cold_simple,
                    color='gray',
                    opacity=0.3,
                    name='cold_region',
                    smooth_shading=True,
                    show_scalar_bar=False
                )
            except:
                # Fallback to original mesh if simplification fails
                plotter.add_mesh(
                    cold,
                    color='gray',
                    opacity=0.3,
                    name='cold_region',
                    show_scalar_bar=False
                )
        
        # Add ultra-simple melt pool representation
        melted = mesh.threshold(value=melting_temp, scalars='temperature')
        if melted.n_points > 0:
            plotter.add_mesh(
                melted,
                scalars='temperature',
                clim=clim,
                cmap=cmap,
                opacity=0.7,
                name='melt_pool',
                show_edges=True,  # Helps visualization
                lighting=False,  # Disable lighting calculations
                pickable=False  # Reduce interaction overhead
            )
    
    # Add animation controls with proper callback signature
    def update_mesh(value, *args):
        try:
            idx = int(value)
            current_mesh = meshes[idx]
            melted = current_mesh.threshold(value=melting_temp, scalars='temperature')
            
            plotter.clear()
            # Add all cells below melting temp in gray
            cold = current_mesh.threshold(value=melting_temp, scalars='temperature', invert=True)
            if cold.n_points > 0:
                plotter.add_mesh(
                    cold,
                    color='gray',
                    opacity=0.3,
                    name='cold_region'
                )
            
            if melted.n_points > 0:
                plotter.add_mesh(
                    melted,
                    scalars='temperature',
                    clim=clim,
                    cmap=cmap,
                    opacity=0.8,
                    name='melt_pool'
                )
        except Exception as e:
            print(f"Error updating mesh: {str(e)}")
    
    plotter.add_slider_widget(
        callback=lambda v: update_mesh(v),
        rng=[0, len(files)-1],
        title='Time Step'
    )
    
    # Add colorbar
    plotter.add_scalar_bar(
        title="Temperature (K)",
        n_labels=5,
        position_x=0.8,
        width=0.1
    )
    
    # Main visualization with robust error handling
    try:
        # Show interactive visualization
        plotter.show()
    except Exception as e:
        print(f"Interactive visualization failed: {str(e)}")
        print("Attempting static image export...")
        try:
            # Try with just the last timestep
            last_mesh = meshes[-1]
            pv.Plotter(off_screen=True).add_mesh(
                last_mesh,
                scalars='temperature',
                cmap=cmap
            ).screenshot('temperature_plot.png')
            print("Saved static visualization to temperature_plot.png")
        except Exception as e:
            print(f"Static visualization failed: {str(e)}")
            print("Please check:")
            print("1. PyVista installation")
            print("2. Output files exist in output/ directory")
            print("3. Available GPU memory")
    finally:
        if 'plotter' in locals():
            plotter.close()

if __name__ == "__main__":
    visualize_results()
