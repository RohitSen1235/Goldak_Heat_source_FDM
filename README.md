# 3D Heat Equation Solver with Gaussian Laser Source

A finite difference solver for 3D heat equation with Gaussian laser heating.

## Features
- Symmetrical cubic computational domain
- Gaussian laser heat source at top surface
- Configurable boundary conditions:
  - Top: Neumann (laser) + optional convective cooling
  - Other faces: Adiabatic/Fixed/Convective
- ParaView-compatible VTK output

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Edit `config.yaml` to set parameters
2. Run the solver:
```bash
python3 solver.py
```
3. View results in ParaView (open files in `output/` directory)

## Configuration
See `config.yaml` for:
- Material properties
- Laser parameters
- Boundary conditions
- Simulation settings
