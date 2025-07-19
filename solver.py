import numpy as np
import yaml
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
from numba import njit, prange, cuda  # For JIT compilation and parallel loops
from numba.cuda import is_available as cuda_is_available

@cuda.jit('void(float64[:,:,::1], float64[:,:,::1], float64, int64)')
def _update_temperature_cuda(T, new_T, factor, N):
    """CUDA kernel for parallel temperature field update using finite differences.
    
    Args:
        T: Input temperature field (3D numpy array)
        new_T: Output temperature field (3D numpy array)
        factor: Thermal diffusion factor (dt*alpha/dx^2)
        N: Grid dimension size (points per dimension)
    """

    i, j, k = cuda.grid(ndim=3)
    if 1 <= i < N-1 and 1 <= j < N-1 and 1 <= k < N-1:
        neighbor_avg = (T[i+1,j,k] + T[i-1,j,k] +
                      T[i,j+1,k] + T[i,j-1,k] +
                      T[i,j,k+1] + T[i,j,k-1]) / 6.0
        delta = (neighbor_avg - T[i,j,k]) * (factor * 6.0)
        if abs(delta) > 1000:
            # Manual sign calculation since np.sign isn't supported in CUDA
            delta = (1.0 if delta > 0 else -1.0) * 1000
        new_T[i,j,k] = T[i,j,k] + delta

class BoundaryType(Enum):
    """Enumeration of boundary condition types for heat transfer simulation."""
    ADIABATIC = "adiabatic"  # No heat flux boundary
    FIXED = "fixed"          # Fixed temperature boundary
    CONVECTIVE = "convective"  # Convective cooling boundary

@dataclass
class BoundaryCondition:
    """Dataclass representing a boundary condition configuration.
    
    Attributes:
        type: Boundary condition type (from BoundaryType enum)
        fixed_temp: Fixed temperature value (K) for FIXED type
        h: Heat transfer coefficient (W/m²K) for CONVECTIVE type
        T_inf: Ambient temperature (K) for CONVECTIVE type
    """
    type: BoundaryType
    fixed_temp: Optional[float] = None
    h: Optional[float] = None
    T_inf: Optional[float] = None

@dataclass
class TopSurfaceConfig:
    """Dataclass for top surface cooling configuration.
    
    Attributes:
        convective_cooling: Whether convective cooling is enabled
        h: Heat transfer coefficient (W/m²K)
        T_inf: Ambient temperature (K) for cooling
    """
    convective_cooling: bool = False
    h: float = 10.0
    T_inf: float = 300.0

@dataclass
class LaserConfig:
    """Dataclass for laser heat source configuration.
    
    Attributes:
        power: Laser power (W)
        spot_size: Beam diameter (m)
        position: (x,y) coordinates of beam center (m)
        reflectivity: Surface reflectivity coefficient (0-1)
    """
    power: float  # Watts
    spot_size: float  # meters
    position: tuple[float, float]  # (x,y) in meters
    reflectivity: float  # 0-1

@dataclass
class DoubleEllipsoidConfig:
    """Dataclass for double ellipsoid (Goldak) heat source configuration.
    
    Attributes:
        enabled: Whether this heat source is active
        Q: Total power (W)
        a_f: Front ellipsoid radius (m)
        a_r: Rear ellipsoid radius (m)
        b: y-axis radius (m)
        c: z-axis radius (m)
        f_f: Front power fraction (must satisfy f_f + f_r = 2)
        f_r: Rear power fraction
        position: (x,y) coordinates of heat source center (m)
    """
    enabled: bool
    Q: float  # Total power (W)
    a_f: float  # Front ellipsoid radius (m)
    a_r: float  # Rear ellipsoid radius (m)
    b: float  # y-axis radius (m)
    c: float  # z-axis radius (m)
    f_f: float  # Front fraction of power
    f_r: float  # Rear fraction of power
    position: tuple[float, float]  # (x,y) in meters

@dataclass
class MaterialConfig:
    """Dataclass for material thermal properties.
    
    Attributes:
        conductivity: Thermal conductivity (W/m-K)
        density: Material density (kg/m³)
        specific_heat: Specific heat capacity (J/kg-K)
    """
    conductivity: float  # W/m-K
    density: float  # kg/m³
    specific_heat: float  # J/kg-K

@dataclass
class SimulationConfig:
    """Dataclass for simulation time parameters.
    
    Attributes:
        duration: Total simulation time (s)
        output_interval: Time between output saves (s)
        max_dt: Maximum allowed time step (s) for stability
    """
    duration: float  # seconds
    output_interval: float  # seconds
    max_dt: float  # seconds

class HeatSolver3D:
    def __init__(self, config_file: str):
        """Initialize the 3D heat solver with configuration from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
            
        Raises:
            ValueError: If invalid solver_type is specified
        """
        self.load_config(config_file)
        self._cuda_available = cuda_is_available()
        if not self._cuda_available:
            print(f"Cuda solver requested but not available, falling back to cpu solver")

        self.solver_type = self.config['simulation'].get('solver_type', 'auto')

        if self.solver_type not in ('auto', 'cpu', 'cuda'):
            raise ValueError("Invalid solver_type, must be 'auto', 'cpu' or 'cuda'")
        
        self.initialize_grid()
        
    def load_config(self, config_file: str) -> None:
        """Load and validate configuration from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
            
        Raises:
            ValueError: If configuration contains invalid values
            FileNotFoundError: If config file doesn't exist
        """
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

        # Parse double ellipsoid config
        ellipsoid_conf = {
            'enabled': to_bool(config['double_ellipsoid']['enabled']),
            'Q': to_float(config['double_ellipsoid']['Q']),
            'a_f': to_float(config['double_ellipsoid']['a_f']),
            'a_r': to_float(config['double_ellipsoid'].get('a_r', config['double_ellipsoid']['a_f'])),
            'b': to_float(config['double_ellipsoid']['b']),
            'c': to_float(config['double_ellipsoid']['c']),
            'f_f': to_float(config['double_ellipsoid']['f_f']),
            'f_r': to_float(config['double_ellipsoid'].get('f_r', 2.0 - float(config['double_ellipsoid']['f_f']))),
            'position': tuple(to_float(x) for x in config['double_ellipsoid']['position'])
        }
        # Validate f_f + f_r = 2
        if not np.isclose(ellipsoid_conf['f_f'] + ellipsoid_conf['f_r'], 2.0, atol=1e-6):
            raise ValueError("f_f + f_r must equal 2.0 in double ellipsoid configuration")

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
        self.double_ellipsoid = DoubleEllipsoidConfig(**ellipsoid_conf)
        self.simulation = SimulationConfig(**sim_conf)
        self.top_surface = TopSurfaceConfig(**top_conf)
        self.config = config  # Store full config for later access
        
        self.boundaries = {}
        for face, bc_config in config['boundaries'].items():
            self.boundaries[face] = BoundaryCondition(
                type=BoundaryType(bc_config['type']),
                fixed_temp=bc_config.get('fixed_temp'),
                h=bc_config.get('h'),
                T_inf=bc_config.get('T_inf')
            )
            
    def initialize_grid(self) -> None:
        """Initialize the computational grid and temperature field.
        
        Sets up:
        - Grid dimensions (N) and spacing (dx)
        - Initial temperature field (300K)
        - Thermal diffusivity (alpha)
        - Laser position (centered)
        """
        self.N = self.config['domain']['points_per_dimension']  # Points per dimension
        self.L = self.config['domain']['size']  # Domain size in meters
        self.dx = self.L / (self.N - 1)
        
        # Update laser position to be centered
        center = self.L / 2
        self.laser.position = (center, center)
        print( f" Laser position : {self.laser.position}")
        # Initialize temperature field (K)
        self.T = np.ones((self.N, self.N, self.N)) * 300.0  # Initial temp 300K
        
        # Calculate thermal diffusivity
        self.alpha = self.material.conductivity / (self.material.density * self.material.specific_heat)
        
        
    def gaussian_laser_source(self, x: float, y: float) -> float:
        """Calculate Gaussian laser intensity at point (x,y).
        
        Args:
            x: X coordinate (m)
            y: Y coordinate (m)
            
        Returns:
            Laser intensity (W/m²) at given point
        """
        x0, y0 = self.laser.position
        w = self.laser.spot_size
        I0 = (2 * self.laser.power) / (np.pi * w**2)
        return I0 * np.exp(-2 * ((x-x0)**2 + (y-y0)**2) / w**2)

    def double_ellipsoid_source(self, x: float, y: float, z: float) -> float:
        """Calculate double ellipsoid heat source using Goldak model.
        
        Args:
            x: X coordinate (m)
            y: Y coordinate (m) 
            z: Z coordinate (m)
            
        Returns:
            Volumetric heat flux (W/m³) at given point
        """
        x0, y0 = self.double_ellipsoid.position
        x_rel = x - x0  # Relative x position
        y_rel = y- y0
        # Convert all lengths to mm for calculation (Goldak model expects mm)
        a_f_mm = self.double_ellipsoid.a_f * 1000
        a_r_mm = self.double_ellipsoid.a_r * 1000
        b_mm = self.double_ellipsoid.b * 1000
        c_mm = self.double_ellipsoid.c * 1000
        x_rel_mm = x_rel * 1000
        y_mm = (y - y0) * 1000
        z_mm = z * 1000
        
        if x_rel >= 0:  # Front half
            a = a_f_mm
            f = self.double_ellipsoid.f_f
        else:  # Rear half
            a = a_r_mm
            f = self.double_ellipsoid.f_r
            
        term1 = -3 * (x_rel_mm/a)**2
        term2 = -3 * (y_mm/b_mm)**2
        term3 = -3 * (z_mm/c_mm)**2
        
        # Calculate heat flux density in W/mm³
        numerator = 6 * np.sqrt(3) * f * self.double_ellipsoid.Q *self.config["laser"]["reflectivity"]
        denominator = np.pi * a * b_mm * c_mm
        q_mm3 = (numerator/denominator) * np.exp(term1 + term2 + term3)
        
        # Convert to W/m³ for consistency with other units
        return q_mm3 *1.0e9
    
    def apply_initial_heat_source(self) -> None:
        """Apply double ellipsoid heat source as initial condition.
        
        Modifies the temperature field by applying heat within the ellipsoid bounds.
        Limits maximum temperature increase to prevent numerical instability.
        """
        for i in range(1, self.N-1):
            for j in range(1, self.N-1):
                for k in range(1, self.N-1):
                    x = i * self.dx
                    y = j * self.dx
                    z = k * self.dx
                    # Only apply heat within ellipsoid bounds
                    if (((x-self.double_ellipsoid.position[0])**2/self.double_ellipsoid.a_f**2  <= 1) and 
                        ((y-self.double_ellipsoid.position[1])**2/self.double_ellipsoid.b**2 <= 1) and
                        (z**2/self.double_ellipsoid.c**2)) <= 1:
                        heat_flux = self.double_ellipsoid_source(x, y, z)
                        # Apply as initial condition with scaling factor
                        # Convert heat flux (W/m³) to temperature change (K)
                        # Using: ΔT = q * dt / (ρ * Cp) with dt = 1e-6s
                        scaling_factor = 2.0e-5  # Equivalent to first time step
                        self.T[i,j,k] += heat_flux * scaling_factor / (self.material.density * self.material.specific_heat)
                        # Limit maximum temperature increase to prevent instability
                        if self.T[i,j,k] > 300 + 3000:  # Max 3000K above ambient
                            self.T[i,j,k] = 300 + 3000
                            print(f"Warning: Temperature capped at 3300K at ({x:.2e}, {y:.2e}, {z:.2e})")

    def apply_boundary_conditions(self) -> None:
        """Apply all boundary conditions to the temperature field.
        
        Handles:
        - Top surface with optional convective cooling
        - All other faces (front, back, left, right, bottom)
        - Supports fixed, adiabatic and convective boundary types
        """

        # Top surface (z = L)
        for i in range(self.N):
            for j in range(self.N):
                x = i * self.dx
                y = j * self.dx
                if  self.double_ellipsoid.enabled:
                    # heat_flux = (1- self.laser.reflectivity) * self.double_ellipsoid_source(x, y, 0.0)
                    heat_flux = (1- self.laser.reflectivity) * double_ellipsoid_source(x, y, self.L/2 , 
                                                                                       self.laser.position, 
                                                                                       self.double_ellipsoid.a_f, 
                                                                                       self.double_ellipsoid.a_r, 
                                                                                       self.double_ellipsoid.b, 
                                                                                       self.double_ellipsoid.c, 
                                                                                       self.double_ellipsoid.f_f, 
                                                                                       self.double_ellipsoid.f_r, 
                                                                                       self.double_ellipsoid.Q, 
                                                                                       self.laser.reflectivity)
                else:
                    # heat_flux = (1- self.config["laser"]["reflectivity"]) *  self.gaussian_laser_source(x, y)
                    heat_flux = (1- self.laser.reflectivity) *  self.gaussian_laser_source(x, y)
                # Apply Neumann condition
                scaling_factor = 1.0e-1  # Equivalent to first time step

                self.T[i,j,-1] = self.T[i,j,-2] + (heat_flux * scaling_factor * self.dx / self.material.conductivity)
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
    
    def solve(self) -> None:
        """Run the main simulation loop.
        
        Iteratively:
        1. Calculates time step
        2. Updates temperature field
        3. Applies boundary conditions
        4. Saves output at specified intervals
        5. Prints progress updates
        """
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
        """Calculate maximum stable time step using CFL condition.
        
        Returns:
            Maximum stable time step (seconds) based on grid spacing and thermal diffusivity
        """
        max_dt = (self.dx**2) / (6 * self.alpha)
        # return min(max_dt, self.simulation.max_dt)
        return max_dt
    
    @staticmethod
    @njit(parallel=True)
    def _apply_heat_source_cpu(T, N, dx, position, a_f, a_r, b, c, f_f, f_r, Q, reflectivity, dt, density, specific_heat):
        """Numba-optimized heat source application"""
        for i in prange(1, N-1):
            for j in range(1, N-1):
                for k in range(1, N-1):
                    x = i * dx
                    y = j * dx
                    z = k * dx
                    # Only apply heat within ellipsoid bounds
                    if ((x-position[0])**2/a_f**2 <=1 and 
                        (y-position[1])**2/b**2 <=1 and
                        (z-0.0)**2/c**2) <= 1:
                        # Call the standalone double_ellipsoid_source function
                        heat_flux = (1 - reflectivity) * double_ellipsoid_source(
                            x, y, z, position, a_f, a_r, b, c, f_f, f_r, Q, reflectivity)
                        T[i,j,k] += heat_flux * dt / (density * specific_heat)
        return T

    def update_temperature(self, dt: float) -> None:
        """Update temperature field for one time step.
        
        Args:
            dt: Time step size (seconds)
            
        Raises:
            ValueError: If invalid time step or numerical instability detected
            RuntimeError: If NaN/infinite values appear in temperature field
        """
        # Pre-calculate constants
        factor = dt * self.alpha / (self.dx**2)
        if factor <= 0 or not np.isfinite(factor):
            raise ValueError(f"Invalid factor value: {factor}")
        
        # Select implementation based on config
        if self.solver_type == 'cpu' or not self._cuda_available:
            self.T = _update_temperature_cpu(self.T, factor, self.N)
        else:
            try:
                d_T = cuda.to_device(self.T)
                d_new_T = cuda.device_array_like(d_T)
                threads = (8, 8, 8)
                blocks = (
                    (self.N + threads[0] - 1) // threads[0],
                    (self.N + threads[1] - 1) // threads[1],
                    (self.N + threads[2] - 1) // threads[2]
                )
                _update_temperature_cuda[blocks, threads](d_T, d_new_T, float(factor), self.N)
                self.T = d_new_T.copy_to_host()
            except Exception as e:
                if self.solver_type == 'cuda':
                    print(f"WARNING: CUDA solver failed ({str(e)}) - falling back to CPU")
                else:
                    print("WARNING: CUDA failed - falling back to CPU")
                self.T = _update_temperature_cpu(self.T, factor, self.N)
        
        # Apply volumetric heat source if double ellipsoid is enabled
        if self.double_ellipsoid.enabled:
            # if self.solver_type == 'cpu' or not self._cuda_available:
            self.T = self._apply_heat_source_cpu(
                self.T, self.N, self.dx, 
                self.double_ellipsoid.position,
                self.double_ellipsoid.a_f,
                self.double_ellipsoid.a_r,
                self.double_ellipsoid.b,
                self.double_ellipsoid.c,
                self.double_ellipsoid.f_f,
                self.double_ellipsoid.f_r,
                self.double_ellipsoid.Q,
                self.laser.reflectivity,
                dt,
                self.material.density,
                self.material.specific_heat
            )
            # else:
            #     # TODO: Add CUDA version
            #     pass

        # Validate temperature values
        if np.isnan(self.T).any():
            nan_count = np.isnan(self.T).sum()
            raise RuntimeError(f"NaN values detected in temperature field ({nan_count} points)")
        if not np.isfinite(self.T).all():
            inf_count = (~np.isfinite(self.T)).sum()
            raise RuntimeError(f"Non-finite values detected in temperature field ({inf_count} points)")
        
        self.apply_boundary_conditions()
        
    def calculate_meltpool_dimensions(self, melting_temp: float = None) -> dict:
        """Calculate meltpool dimensions (length, width, depth) in micro meters.
        
        Args:
            melting_temp: Optional override of melting temperature (K)
            
        Returns:
            Dictionary with keys: length, width, depth (in micrometers)
        """
        if melting_temp is None:
            melting_temp = self.config['material']['melting_temperature']
        
        # Find all cells above melting temperature
        melt_cells = np.where(self.T > melting_temp)
        
        if len(melt_cells[0]) == 0:
            return {'length': 0, 'width': 0, 'depth': 0}
        
        # Get laser center in grid coordinates
        center_x, center_y = self.laser.position
        center_i = int(center_x / self.dx)
        center_j = int(center_y / self.dx)
        
        # Calculate max distances from center in each dimension
        i_coords = melt_cells[0]
        j_coords = melt_cells[1] 
        k_coords = melt_cells[2]
        
        length = (np.max(i_coords) - np.min(i_coords)) * self.dx
        width = (np.max(j_coords) - np.min(j_coords)) * self.dx
        depth = (np.max(k_coords) - np.min(k_coords)) * self.dx
        
        print({
            'length': float(length*1.0e6),
            'width': float(width*1.0e6),
            'depth': float(depth*1.0e6)
        })

        return {
            'length': length,
            'width': width,
            'depth': depth
        }

    def save_output(self, step: int) -> None:
        """Save current temperature field to VTK file.
        
        Args:
            step: Time step number used in filename
            
        Raises:
            RuntimeError: If temperature field contains invalid values
        """
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

@njit(fastmath=True)
def double_ellipsoid_source(x: float, y: float, z: float, 
                          position: tuple[float, float],
                          a_f: float, a_r: float, b: float, c: float,
                          f_f: float, f_r: float, Q: float, reflectivity: float) -> float:
    """Numba-compatible version of double ellipsoid heat source calculation.
    
    Args:
        x,y,z: Coordinates (m)
        position: (x,y) center of heat source (m)
        a_f: Front ellipsoid radius (m)
        a_r: Rear ellipsoid radius (m) 
        b: y-axis radius (m)
        c: z-axis radius (m)
        f_f: Front power fraction
        f_r: Rear power fraction
        Q: Total power (W)
        reflectivity: Surface reflectivity (0-1)
        
    Returns:
        Volumetric heat flux (W/m³) at given point
    """
    x0, y0 = position
    x_rel = x - x0  # Relative x position
    
    # Convert all lengths to mm for calculation (Goldak model expects mm)
    a_f_mm = a_f * 1000
    a_r_mm = a_r * 1000
    b_mm = b * 1000
    c_mm = c * 1000
    x_rel_mm = x_rel * 1000
    y_mm = (y - y0) * 1000
    z_mm = z * 1000
    
    if x_rel >= 0:  # Front half
        a = a_f_mm
        f = f_f
    else:  # Rear half
        a = a_r_mm
        f = f_r
        
    term1 = -3 * (x_rel_mm/a)**2
    term2 = -3 * (y_mm/b_mm)**2
    term3 = -3 * (z_mm/c_mm)**2
    
    # Calculate heat flux density in W/mm³
    numerator = 6 * np.sqrt(3) * f * Q * reflectivity
    denominator = np.pi * a * b_mm * c_mm
    q_mm3 = (numerator/denominator) * np.exp(term1 + term2 + term3)
    
    # Convert to W/m³ for consistency with other units
    return q_mm3 * 1.0e9

@njit(parallel=True, fastmath=True)
def _update_temperature_cpu(T: np.ndarray, factor: float, N: int) -> np.ndarray:
    """Numba-optimized CPU temperature field update.
    
    Args:
        T: Input temperature field (3D array)
        factor: Thermal diffusion factor (dt*alpha/dx^2)
        N: Grid dimension size
        
    Returns:
        Updated temperature field (3D array)
    """
    new_T = np.copy(T)
    
    # More stable computation avoiding large dx2_inv
    alpha = factor * 6.0
    if alpha <= 1e-10:
        alpha = 1e-10  # Prevent division by extremely small numbers
    
    # Update interior points with stability checks - parallelized
    for i in range(1, N-1):
        for j in range(1, N-1):
            # prange enables parallelization of the outermost loop
            for k in prange(1, N-1):
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
    solver.calculate_meltpool_dimensions()
