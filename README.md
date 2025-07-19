# 3D Heat Equation Solver - Finite Difference Method

A CUDA-accelerated 3D heat equation solver using Finite Difference Method (FDM) with configurable boundary conditions and heat sources. Designed for simulating laser-material interactions in additive manufacturing processes.

## Features

- **Numerical Methods**: Finite Difference Method (FDM) with explicit time stepping
- **Acceleration**: CUDA and Numba-optimized CPU implementations
- **Heat Sources**:
  - Gaussian laser beam
  - Double ellipsoid (Goldak) heat source model
- **Boundary Conditions**:
  - Fixed temperature
  - Adiabatic (no heat flux)
  - Convective cooling
- **Visualization**:
  - 3D interactive visualization with PyVista
  - 2D slice visualization
  - Melt pool tracking

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (optional but recommended)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (for CUDA acceleration)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/3D_HeatEqSolver_FDM.git
cd 3D_HeatEqSolver_FDM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The solver is configured via `config.yaml`. Key sections:

### Material Properties
```yaml
material:
  conductivity: 16.2    # Thermal conductivity (W/m-K)
  density: 7800.0       # Density (kg/m³)
  specific_heat: 500.0  # Specific heat capacity (J/kg-K)
  melting_temperature: 1400  # Melting point (K)
```

### Laser Parameters
```yaml
laser:
  power: 250            # Laser power (W)
  spot_size: 1e-5       # Beam diameter (m)
  position: [x, y]      # Beam center position (m)
  reflectivity: 0.8     # Surface reflectivity (0-1)
```

### Double Ellipsoid Model (Goldak)
```yaml
double_ellipsoid:
  enabled: true         # Toggle between heat source models
  Q: 250                # Total power (W)
  a_f: 1.0e-4           # Front ellipsoid radius (m)
  a_r: 1.5e-4           # Rear ellipsoid radius (m)
  b: 0.8e-4             # y-axis radius (m)
  c: 1.2e-4             # z-axis radius (m)
  f_f: 0.8              # Front power fraction
  f_r: 1.2              # Rear power fraction (f_f + f_r must equal 2)
```

### Simulation Parameters
```yaml
simulation:
  duration: 0.001       # Simulation duration (s)
  output_interval: 0.0001 # Output interval (s)
  max_dt: 1.0e-4        # Maximum time step (s)
  solver_type: "cuda"   # "auto", "cpu", or "cuda"
```

### Domain Settings
```yaml
domain:
  size: 0.5e-3          # Domain size (m)
  points_per_dimension: 50 # Grid resolution
```

## Running the Simulation

1. Configure `config.yaml` with your desired parameters
2. Run the solver:
```bash
python solver.py
```

The solver will:
- Initialize the temperature field
- Run the simulation with progress updates
- Save VTK output files to `output/` directory

## Visualization

### 3D Interactive Visualization
```bash
python visualize.py
```
Features:
- Interactive time step slider
- Melt pool visualization
- Temperature color mapping
- Cross-section views

### 2D Slice Visualization
```bash
python visualize_slices.py [--mode x|y|both]
```
Options:
- `--mode x`: X-axis slices
- `--mode y`: Y-axis slices
- `--mode both`: Both X and Y slices (default)

## Output Files

The solver generates VTK files (`timestep_XXXX.vtr`) in the `output/` directory containing:
- 3D temperature field
- Grid coordinates
- Metadata from config

These can be opened in ParaView or other VTK-compatible visualization tools.

## Example Usage

1. Simulate laser heating with default parameters:
```bash
python solver.py
```

2. Visualize results in 3D:
```bash
python visualize.py
```

3. Generate X-slice animation:
```bash
python visualize_slices.py --mode x
```

## Troubleshooting

### Common Issues

1. **CUDA errors**:
   - Ensure CUDA Toolkit is installed
   - Try `solver_type: "cpu"` in config.yaml

2. **Visualization failures**:
   - Try reducing grid resolution
   - Ensure PyVista is properly installed

3. **Numerical instability**:
   - Reduce time step (`max_dt`)
   - Increase grid resolution

## License

MIT License - See LICENSE file for details.
