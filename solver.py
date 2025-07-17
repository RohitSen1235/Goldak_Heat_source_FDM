import numpy as np
import yaml
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional

class BoundaryType(Enum):
    ADIABATIC = "adiabatic"
    FIXED = "fixed"
    CONVECTIVE = "convective"

@dataclass
class BoundaryCondition:
    type: BoundaryType
    fixed_temp: Optional[float] = None
    h: Optional[float] = None
    T_inf: Optional[float] = None

@dataclass
class TopSurfaceConfig:
    convective_cooling: bool = False
    h: float = 10.0
    T_inf: float = 300.0

@dataclass
class LaserConfig:
    power: float  # Watts
    spot_size: float  # meters
    position: tuple[float, float]  # (x,y) in meters
    reflectivity: float  # 0-1

@dataclass
class MaterialConfig:
    conductivity: float  # W/m-K
    density: float  # kg/m³
    specific_heat: float  # J/kg-K

@dataclass
class SimulationConfig:
    duration: float  # seconds
    output_interval: float  # seconds
    max_dt: float  # seconds

class HeatSolver3D:
    def __init__(self, config_file: str):
        self.load_config(config_file)
        self.initialize_grid()
        
    def load_config(self, config_file: str):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Type-safe configuration parsing
        def to_float(x):
            try:
                return float(x)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid numeric value: {x}")

        def to_bool(x):
            if isinstance(x, bool):
                return x
            if str(x).lower() in ('true', '1', 't', 'y', 'yes'):
                return True
            if str(x).lower() in ('false', '0', 'f', 'n', 'no'):
                return False
            raise ValueError(f"Invalid boolean value: {x}")

        # Parse material config
        material_conf = {
            'conductivity': to_float(config['material']['conductivity']),
            'density': to_float(config['material']['density']),
            'specific_heat': to_float(config['material']['specific_heat'])
        }

        # Parse laser config
        laser_conf = {
            'power': to_float(config['laser']['power']),
            'spot_size': to_float(config['laser']['spot_size']),
            'position': tuple(to_float(x) for x in config['laser']['position']),
            'reflectivity': to_float(config['laser']['reflectivity'])
        }

        # Parse simulation config
        sim_conf = {
            'duration': to_float(config['simulation']['duration']),
            'output_interval': to_float(config['simulation']['output_interval']),
            'max_dt': to_float(config['simulation']['max_dt'])
        }

        # Parse top surface config
        top_conf = {
            'convective_cooling': to_bool(config['top_surface']['convective_cooling']),
            'h': to_float(config['top_surface']['h']),
            'T_inf': to_float(config['top_surface']['T_inf'])
        }
        
        self.material = MaterialConfig(**material_conf)
        self.laser = LaserConfig(**laser_conf)
        self.simulation = SimulationConfig(**sim_conf)
        self.top_surface = TopSurfaceConfig(**top_conf)
        
        self.boundaries = {}
        for face, bc_config in config['boundaries'].items():
            self.boundaries[face] = BoundaryCondition(
                type=BoundaryType(bc_config['type']),
                fixed_temp=bc_config.get('fixed_temp'),
                h=bc_config.get('h'),
                T_inf=bc_config.get('T_inf')
            )
            
    def initialize_grid(self):
        self.N = 50  # Points per dimension
        self.L = 1e-3  # Domain size (1mm)
        self.dx = self.L / (self.N - 1)
        
        # Initialize temperature field (K)
        self.T = np.ones((self.N, self.N, self.N)) * 300.0  # Initial temp 300K
        
        # Calculate thermal diffusivity
        self.alpha = self.material.conductivity / (self.material.density * self.material.specific_heat)
        
    def gaussian_laser_source(self, x: float, y: float) -> float:
        """Calculate laser intensity at point (x,y)"""
        x0, y0 = self.laser.position
        w = self.laser.spot_size
        I0 = (2 * self.laser.power) / (np.pi * w**2)
        return I0 * np.exp(-2 * ((x-x0)**2 + (y-y0)**2) / w**2)
    
    def apply_boundary_conditions(self):
        """Apply all boundary conditions to the temperature field"""
        # Top surface (z = L)
        for i in range(self.N):
            for j in range(self.N):
                x = i * self.dx
                y = j * self.dx
                heat_flux = (1 - self.laser.reflectivity) * self.gaussian_laser_source(x, y)
                
                # Apply Neumann condition
                self.T[i,j,-1] = self.T[i,j,-2] + (heat_flux * self.dx / self.material.conductivity)
                
                # Optional convective cooling
                if self.top_surface.convective_cooling:
                    convective_term = (self.top_surface.h * self.dx / self.material.conductivity) * \
                                    (self.T[i,j,-1] - self.top_surface.T_inf)
                    self.T[i,j,-1] -= convective_term
        
        # Other boundaries (implementation depends on selected type)
        # ... will be implemented in next steps
    
    def solve(self):
        """Main solver loop"""
        t = 0.0
        output_count = 0
        
        while t < self.simulation.duration:
            dt = self.calculate_time_step()
            self.update_temperature(dt)
            t += dt
            
            if t >= output_count * self.simulation.output_interval:
                self.save_output(output_count)
                output_count += 1
                print(f"Saved output at t = {t:.3e}s ({(t/self.simulation.duration)*100:.1f}% complete)")
            else:
                print(f"Progress: t = {t:.3e}s ({(t/self.simulation.duration)*100:.1f}% complete)", end='\r')
    
    def calculate_time_step(self) -> float:
        """Calculate stable time step"""
        max_dt = (self.dx**2) / (6 * self.alpha)
        return min(max_dt, self.simulation.max_dt)
    
    def update_temperature(self, dt: float):
        """Update temperature field using explicit finite difference"""
        new_T = np.copy(self.T)
        
        # Update interior points
        for i in range(1, self.N-1):
            for j in range(1, self.N-1):
                for k in range(1, self.N-1):
                    d2T_dx2 = (self.T[i+1,j,k] - 2*self.T[i,j,k] + self.T[i-1,j,k]) / (self.dx**2)
                    d2T_dy2 = (self.T[i,j+1,k] - 2*self.T[i,j,k] + self.T[i,j-1,k]) / (self.dx**2)
                    d2T_dz2 = (self.T[i,j,k+1] - 2*self.T[i,j,k] + self.T[i,j,k-1]) / (self.dx**2)
                    
                    new_T[i,j,k] = self.T[i,j,k] + dt * self.alpha * (d2T_dx2 + d2T_dy2 + d2T_dz2)
        
        self.T = new_T
        self.apply_boundary_conditions()
    
    def save_output(self, step: int):
        """Save output in VTK format for ParaView"""
        import os
        from pyevtk.hl import gridToVTK
        
        # Create output directory if needed
        os.makedirs("output", exist_ok=True)
        
        # Create coordinates
        x = np.arange(0, self.L + self.dx, self.dx)
        y = np.arange(0, self.L + self.dx, self.dx)
        z = np.arange(0, self.L + self.dx, self.dx)
        
        # Save as VTK file
        filename = f"output/timestep_{step:04d}"
        gridToVTK(filename, x, y, z, cellData={"temperature": self.T})

if __name__ == "__main__":
    solver = HeatSolver3D("config.yaml")
    solver.solve()
