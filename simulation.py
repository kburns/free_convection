"""
Dedalus script for 3D free convection test.
"""

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx = Ly = Lz = 2
Nx = Ny = Nz = 512
Re = 1587
Ri = 397
Ro = 1.6
Pr = 1
b_noise = 1e-3
stop_sim_time = Ri  # ~400
nominal_dt = Lz / Nz  # ~4e-3
checkpoint_dt = stop_sim_time / 100  # ~4.0
slice_dt = checkpoint_dt / 10  # ~0.4
line_dt = slice_dt / 10  # ~0.04
scalar_dt = line_dt  # ~0.04
timestepper = "RK222"
cfl_params = {'initial_dt': nominal_dt,
              'max_dt': 10*nominal_dt,
              'cadence': 10,
              'safety': 1.0,
              'min_change': 0.5,
              'max_change': 1.5,
              'threshold': 0.05}

# Domain
start_init_time = time.time()
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', Nz, interval=(-Lz, 0), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

# Problem
problem = de.IVP(domain, variables=['p','b','u','v','w','bz','uz','vz','wz'], time='t')
problem.parameters['Re'] = Re
problem.parameters['Ri'] = Ri
problem.parameters['Ro'] = Ro
problem.parameters['Pr'] = Pr
problem.substitutions['Lap(A,Az)'] = "dx(dx(A)) + dy(dy(A)) + dz(Az)"
problem.substitutions['Adv(A,Az)'] = "u*dx(A) + v*dy(A) + w*Az"
problem.substitutions['hmean(A)'] = "integ(A, 'x', 'y')"
problem.substitutions['hvar(A)'] = "hmean((A - hmean(A))**2)"

problem.add_equation("dx(u) + dy(v) + wz = 0")
problem.add_equation("dt(b) - (1/Pr/Re)*Lap(b,bz)                    = - Adv(b,bz)")
problem.add_equation("dt(u) -    (1/Re)*Lap(u,uz) + dx(p) - (1/Ro)*v = - Adv(u,uz)")
problem.add_equation("dt(v) -    (1/Re)*Lap(v,vz) + dy(p) + (1/Ro)*u = - Adv(v,vz)")
problem.add_equation("dt(w) -    (1/Re)*Lap(w,wz) + dz(p) - b        = - Adv(w,wz)")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")

problem.add_bc("left(bz) = Ri")
problem.add_bc("left(uz) = 0")
problem.add_bc("left(vz) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(bz) = - Pr * Re")
problem.add_bc("right(uz) = 0")
problem.add_bc("right(vz) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0) and (ny == 0)")

# Build solver
solver = problem.build_solver(timestepper)
logger.info('Solver built')

# Initial conditions
z = domain.grid(2)
b = solver.state['b']
bz = solver.state['bz']
# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=23)
noise = b_noise * rand.standard_normal(gshape)[slices]
# Linear background + perturbations damped at walls
b['g'] = Ri * (z + noise)
b.differentiate('z', out=bz)

# Integration parameters
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
checkpoints = solver.evaluator.add_file_handler('data_checkpoints', sim_dt=checkpoint_dt, max_writes=1)
checkpoints.add_system(solver.state, layout='g')

slices = solver.evaluator.add_file_handler('data_slices', sim_dt=slice_dt, max_writes=100)
for var in problem.variables:
    slices.add_task("interp(%s, x='left')" %var)
    slices.add_task("interp(%s, z='left')" %var)
    slices.add_task("interp(%s, z='right')" %var)

lines = solver.evaluator.add_file_handler('data_lines', sim_dt=line_dt, max_writes=10000)
for var in problem.variables:
    lines.add_task("hmean(%s)" %var)
    lines.add_task("hvar(%s)" %var)

scalars = solver.evaluator.add_file_handler('data_scalars', sim_dt=scalar_dt, max_writes=1000000)
scalars.add_task("integ(b)", name="B")
scalars.add_task("integ(u*u + v*v + w*w)/2", name="KE")

# CFL
CFL = flow_tools.CFL(solver, **cfl_params)
CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("(u*u + v*v + w*w)/2", name='KE')

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max KE = %f' %flow.max('KE'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

