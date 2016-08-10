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

if (len(sys.argv)!=5):
    if (rank==0):
        print('Usage:\n    mpirun -n <size> python power_spectrum.py <simulation_folder> <species> <n_k_x> <n_k_y>')
    sys.exit()

simulation_folder = sys.argv[1]
species = sys.argv[2]
n_k_x = int(sys.argv[3])
n_k_y = int(sys.argv[4])

raw_folder = simulation_folder + '/MS/RAW/' + species + '/'
output_folder = simulation_folder + '/'

t_array = oi.get_HIST_time(simulation_folder)
timesteps = range(len(t_array))

for t in timesteps:
    if (rank==0):
        t_start = MPI.Wtime()
        print('Starting timestep ' + str(t))

    pic.calculate_power_spectrum(comm, species, t, raw_folder, output_folder, n_k_x, n_k_y)

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Total time for timestep:')
        print(t_elapsed)

