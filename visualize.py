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
    
    # Add all timesteps to the plotter
    for file in files:
        mesh = pv.read(file)
        plotter.add_mesh(mesh, scalars='temperature', clim=[300, 500])
    
    # Store all meshes for animation
    meshes = [pv.read(file) for file in files]
    
    # Add controls for animation
    def update_mesh(value):
        idx = int(value)
        plotter.update_scalars(meshes[idx]['temperature'])
    
    plotter.add_slider_widget(
        callback=update_mesh,
        rng=[0, len(files)-1],
        title='Time Step'
    )
    
    # Show the visualization
    plotter.show()

if __name__ == "__main__":
    visualize_results()
