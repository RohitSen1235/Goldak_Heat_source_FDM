import pyvista as pv
import os
import glob
import argparse
import yaml

def create_slice_plotter(meshes, mode, melted=False, melting_temp=None):
    plotter = pv.Plotter()
    temp_min = min(m['temperature'].min() for m in meshes)
    temp_max = max(m['temperature'].max() for m in meshes)
    
    def update_mesh(value):
        idx = int(value)
        mesh = meshes[idx]
        if melted:
            mesh = mesh.threshold(value=melting_temp, scalars='temperature')
        
        bounds = mesh.bounds
        center = [
            (bounds[1] + bounds[0])/2,
            (bounds[3] + bounds[2])/2,
            (bounds[5] + bounds[4])/2
        ]
        
        plotter.clear()
        
        if mode in ['x', 'both']:
            plotter.add_mesh_slice(
                mesh,
                normal='x',
                origin=[center[0], center[1], center[2]],
                scalars='temperature',
                clim=[temp_min, temp_max]
            )
        
        if mode in ['y', 'both']:
            plotter.add_mesh_slice(
                mesh,
                normal='y',
                origin=[center[0], center[1], center[2]],
                scalars='temperature',
                clim=[temp_min, temp_max]
            )
        
        plotter.add_slider_widget(
            callback=lambda v: update_mesh(v),
            rng=[0, len(meshes)-1],
            title='Time Step',
            value=idx
        )
    
    update_mesh(0)
    return plotter

def visualize_slices(mode='both'):
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
    
    # Create two separate plotters
    full_plotter = create_slice_plotter(meshes, mode)
    melted_plotter = create_slice_plotter(meshes, mode, melted=True, melting_temp=melting_temp)
    
    # Show initial plotter
    full_plotter.show()
    melted_plotter.close()  # Keep hidden initially
    
    # Simple toggle between views
    while True:
        choice = input("Show melted only? (y/n/q to quit): ").lower()
        if choice == 'y':
            full_plotter.close()
            melted_plotter.show()
        elif choice == 'n':
            melted_plotter.close()
            full_plotter.show()
        elif choice == 'q':
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['x', 'y', 'both'], 
                       default='both', help='Slice mode (x, y, or both)')
    args = parser.parse_args()
    
    visualize_slices(args.mode)
