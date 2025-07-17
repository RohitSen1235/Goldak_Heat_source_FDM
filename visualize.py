import pyvista as pv
import os
import glob

def visualize_results():
    # Find all output files
    files = sorted(glob.glob('output/timestep_*.vtr'))
    
    if not files:
        print("No output files found in output/ directory")
        return
    
    # Create a plotter
    plotter = pv.Plotter()
    
    # Read all meshes
    meshes = [pv.read(file) for file in files]
    
    # Add all timesteps to the plotter
    for mesh in meshes:
        temp_min = mesh['temperature'].min()
        temp_max = mesh['temperature'].max()
        plotter.add_mesh(mesh, scalars='temperature', clim=[temp_min, temp_max])
    
    # Add controls for animation
    def update_mesh(value):
        idx = int(value)
        current_mesh = meshes[idx]
        temp_min = current_mesh['temperature'].min()
        temp_max = current_mesh['temperature'].max()
        plotter.update_scalars(current_mesh['temperature'])
    
    plotter.add_slider_widget(
        callback=update_mesh,
        rng=[0, len(files)-1],
        title='Time Step'
    )
    
    # Show the visualization
    plotter.show()

if __name__ == "__main__":
    visualize_results()
