import numpy as np
from mpi4py import MPI
import h5py

import osiris_interface as oi
import pic_calculations as pic
import triangle_calculations as tri
import utilities
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if (len(sys.argv)!=7 and len(sys.argv)!=8):
    if (rank==0):
        print('Usage:\n    mpirun -n <size> python deposit_parallel.py <simulation_folder> <species> <deposit_n_x> <deposit_n_y> <deposit_n_z> <zoom> [<t>]')
    sys.exit()

simulation_folder = sys.argv[1]
species = sys.argv[2]
deposit_n_x = int(sys.argv[3])
deposit_n_y = int(sys.argv[4])
deposit_n_z = int(sys.argv[5])
zoom = float(sys.argv[6])

raw_folder = simulation_folder + '/MS/RAW/' + species + '/'
output_folder = simulation_folder + '/' + str(deposit_n_x) + 'x' + str(deposit_n_y) + 'x' + str(deposit_n_z) + '/'

t_array = oi.get_HIST_time(simulation_folder)

if (len(sys.argv)==8):
    t = int(sys.argv[7])
    timesteps = [t]
else:
    n_t = len(t_array)
    timesteps = range(n_t)

for t in timesteps:
    if (rank==0):
        t_start = MPI.Wtime()
        print('Starting timestep ' + str(t))

#    pic.save_cic_fields_parallel(comm, species, t, raw_folder, output_folder, deposit_n_x, deposit_n_y)
    tri.save_triangle_fields_parallel_3d(comm, species, t, raw_folder, output_folder, deposit_n_x, deposit_n_y, deposit_n_z, zoom)

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Total time for timestep:')
        print(t_elapsed)

