import numpy as np
import yaml
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
from numba import njit  # For JIT compilation

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
                    temp_diff = self.T[i,j,-1] - self.top_surface.T_inf
                    if not np.isnan(temp_diff) and np.isfinite(temp_diff):
                        convective_term = (self.top_surface.h * self.dx / self.material.conductivity) * temp_diff
                        self.T[i,j,-1] -= convective_term
        
        # Implement boundary conditions for all faces
        for face, bc in self.boundaries.items():
            if face == 'bottom':  # z=0
                for i in range(self.N):
                    for j in range(self.N):
                        if bc.type == BoundaryType.FIXED:
                            self.T[i,j,0] = bc.fixed_temp
                        elif bc.type == BoundaryType.ADIABATIC:
                            self.T[i,j,0] = self.T[i,j,1]
                        elif bc.type == BoundaryType.CONVECTIVE:
                            conv_term = (bc.h * self.dx / self.material.conductivity) * \
                                       (self.T[i,j,0] - bc.T_inf)
                            self.T[i,j,0] = self.T[i,j,1] + conv_term
            
            elif face == 'front':  # y=0
                for i in range(self.N):
                    for k in range(self.N):
                        if bc.type == BoundaryType.FIXED:
                            self.T[i,0,k] = bc.fixed_temp
                        elif bc.type == BoundaryType.ADIABATIC:
                            self.T[i,0,k] = self.T[i,1,k]
                        elif bc.type == BoundaryType.CONVECTIVE:
                            conv_term = (bc.h * self.dx / self.material.conductivity) * \
                                       (self.T[i,0,k] - bc.T_inf)
                            self.T[i,0,k] = self.T[i,1,k] + conv_term
            
            elif face == 'back':  # y=L
                for i in range(self.N):
                    for k in range(self.N):
                        if bc.type == BoundaryType.FIXED:
                            self.T[i,-1,k] = bc.fixed_temp
                        elif bc.type == BoundaryType.ADIABATIC:
                            self.T[i,-1,k] = self.T[i,-2,k]
                        elif bc.type == BoundaryType.CONVECTIVE:
                            conv_term = (bc.h * self.dx / self.material.conductivity) * \
                                       (self.T[i,-1,k] - bc.T_inf)
                            self.T[i,-1,k] = self.T[i,-2,k] + conv_term
            
            elif face == 'left':  # x=0
                for j in range(self.N):
                    for k in range(self.N):
                        if bc.type == BoundaryType.FIXED:
                            self.T[0,j,k] = bc.fixed_temp
                        elif bc.type == BoundaryType.ADIABATIC:
                            self.T[0,j,k] = self.T[1,j,k]
                        elif bc.type == BoundaryType.CONVECTIVE:
                            conv_term = (bc.h * self.dx / self.material.conductivity) * \
                                       (self.T[0,j,k] - bc.T_inf)
                            self.T[0,j,k] = self.T[1,j,k] + conv_term
            
            elif face == 'right':  # x=L
                for j in range(self.N):
                    for k in range(self.N):
                        if bc.type == BoundaryType.FIXED:
                            self.T[-1,j,k] = bc.fixed_temp
                        elif bc.type == BoundaryType.ADIABATIC:
                            self.T[-1,j,k] = self.T[-2,j,k]
                        elif bc.type == BoundaryType.CONVECTIVE:
                            conv_term = (bc.h * self.dx / self.material.conductivity) * \
                                       (self.T[-1,j,k] - bc.T_inf)
                            self.T[-1,j,k] = self.T[-2,j,k] + conv_term
    
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
        """Update temperature field using JIT-accelerated finite difference"""
        # Pre-calculate constants
        factor = dt * self.alpha / (self.dx**2)
        if factor <= 0 or not np.isfinite(factor):
            raise ValueError(f"Invalid factor value: {factor}")
        
        # Call Numba-optimized function
        self.T = _update_temperature_numba(self.T, factor, self.N)
        
        # Validate temperature values
        if np.isnan(self.T).any():
            nan_count = np.isnan(self.T).sum()
            raise RuntimeError(f"NaN values detected in temperature field ({nan_count} points)")
        if not np.isfinite(self.T).all():
            inf_count = (~np.isfinite(self.T)).sum()
            raise RuntimeError(f"Non-finite values detected in temperature field ({inf_count} points)")
        
        self.apply_boundary_conditions()
        
    def save_output(self, step: int):
        """Save output in VTK format for ParaView"""
        import os
        from pyevtk.hl import gridToVTK
        
        # Validate before saving
        if np.isnan(self.T).any():
            nan_count = np.isnan(self.T).sum()
            raise RuntimeError(f"Cannot save output - temperature field contains {nan_count} NaN values")
        if not np.isfinite(self.T).all():
            inf_count = (~np.isfinite(self.T)).sum()
            raise RuntimeError(f"Cannot save output - temperature field contains {inf_count} non-finite values")
        
        # Create output directory if needed
        os.makedirs("output", exist_ok=True)
        
        # Create coordinates
        x = np.arange(0, self.L + self.dx, self.dx)
        y = np.arange(0, self.L + self.dx, self.dx)
        z = np.arange(0, self.L + self.dx, self.dx)
        
        # Save as VTK file
        filename = f"output/timestep_{step:04d}"
        gridToVTK(filename, x, y, z, cellData={"temperature": self.T})

@njit
def _update_temperature_numba(T, factor, N):
    """Numba-optimized temperature update"""
    new_T = np.copy(T)
    
    # More stable computation avoiding large dx2_inv
    alpha = factor * 6.0
    if alpha <= 1e-10:
        alpha = 1e-10  # Prevent division by extremely small numbers
    
    # Update interior points with stability checks
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):
                # Compute neighbor average
                neighbor_avg = (T[i+1,j,k] + T[i-1,j,k] +
                              T[i,j+1,k] + T[i,j-1,k] +
                              T[i,j,k+1] + T[i,j,k-1]) / 6.0
                
                # Stable update using limited difference
                delta = (neighbor_avg - T[i,j,k]) * alpha
                if abs(delta) > 1000:  # Limit large changes
                    delta = np.sign(delta) * 1000
                new_T[i,j,k] = T[i,j,k] + delta
    
    return new_T



if __name__ == "__main__":
    solver = HeatSolver3D("config.yaml")
    solver.solve()
