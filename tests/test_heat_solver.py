import pytest
from fem.heat_solver import HeatEquationSolver

def test_basic_solution():
    solver = HeatEquationSolver(
        k=1.0,
        f=0.0,
        left_temp=100.0,
        right_temp=0.0,
        mesh_resolution=8
    )
    mesh, solution = solver.solve()
    
    assert mesh.num_vertices() == 81  # (8+1)^2
    assert solution.vector().size() == 81
    
def test_parameter_validation():
    with pytest.raises(ValueError):
        HeatEquationSolver(k=-1.0).solve()
        
    with pytest.raises(ValueError):
        HeatEquationSolver(mesh_resolution=3).solve()
