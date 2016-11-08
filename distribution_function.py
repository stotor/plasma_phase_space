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

if (len(sys.argv)!=3):
    if (rank==0):
        print('Usage:\n    mpirun -n <size> python distribution_function.py <simulation_folder> <species>')
    sys.exit()

simulation_folder = sys.argv[1]
species = sys.argv[2]
sample_times = [0]
sample_locations = [[5.5875, 23.3375],
                    [4.7875, 25.6875]]

raw_folder = simulation_folder + '/MS/RAW/' + species + '/'
output_folder = simulation_folder + '/distribution_functions/'

for t in sample_times:
    if (rank==0):
        t_start = MPI.Wtime()
        print('Starting t = ' + str(t))

    tri.distribution_function_2d(comm, species, t, raw_folder, output_folder, sample_locations)
        
    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Total time:')
        print(t_elapsed)

