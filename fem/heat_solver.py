from dolfin import *
from typing import Tuple

class HeatEquationSolver:
    """
    Solves the 2D steady-state heat equation:
    -∇·(k∇u) = f with Dirichlet boundary conditions
    
    Args:
        k (float): Thermal conductivity
        f (float): Heat source term
        left_temp (float): Left boundary temperature
        right_temp (float): Right boundary temperature
        mesh_resolution (int): Mesh resolution (default=32)
    """
    
    def __init__(self, 
                 k: float = 1.0, 
                 f: float = 0.0, 
                 left_temp: float = 100.0, 
                 right_temp: float = 0.0,
                 mesh_resolution: int = 32):
        
        self.params = {
            'k': k,
            'f': f,
            'left_temp': left_temp,
            'right_temp': right_temp,
            'mesh_resolution': mesh_resolution
        }
        
        self.mesh = None
        self.solution = None
        
    def solve(self) -> Tuple[Mesh, Function]:
        """Solve the PDE and return (mesh, solution)"""
        self._validate_parameters()
        self._create_mesh()
        self._setup_problem()
        self._apply_boundary_conditions()
        self._solve_system()
        return self.mesh, self.solution
    
    def _validate_parameters(self):
        """Ensure physical parameters are valid"""
        if self.params['k'] <= 0:
            raise ValueError("Thermal conductivity must be positive")
        if self.params['mesh_resolution'] < 4:
            raise ValueError("Mesh resolution must be at least 4")
            
    def _create_mesh(self):
        """Create unit square mesh"""
        self.mesh = UnitSquareMesh(self.params['mesh_resolution'], 
                                 self.params['mesh_resolution'])
        
    def _setup_problem(self):
        """Setup variational formulation"""
        self.V = FunctionSpace(self.mesh, 'P', 1)
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        a = self.params['k'] * inner(grad(u), grad(v)) * dx
        L = self.params['f'] * v * dx
        
        self.a = a
        self.L = L
        
    def _apply_boundary_conditions(self):
        """Apply Dirichlet boundary conditions"""
        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], 0.0)
        
        class RightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], 1.0)
                
        self.bcs = [
            DirichletBC(self.V, Constant(self.params['left_temp']), LeftBoundary()),
            DirichletBC(self.V, Constant(self.params['right_temp']), RightBoundary())
        ]
        
    def _solve_system(self):
        """Solve the linear system"""
        u = Function(self.V)
        solve(self.a == self.L, u, self.bcs)
        self.solution = u
