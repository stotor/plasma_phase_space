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

if (len(sys.argv)!=4):
    if (rank==0):
        print('Usage:\n    mpirun -n <size> python distribution_function.py <simulation_folder> <species> <t>')
    sys.exit()

simulation_folder = sys.argv[1]
species = sys.argv[2]
t = 4
sample_locations = [[12.8, 12.8],[12.8,13.0]]

raw_folder = simulation_folder + '/MS/RAW/' + species + '/'
output_folder = simulation_folder + '/distribution_functions/'

if (rank==0):
    t_start = MPI.Wtime()
    print('Starting')

tri.distribution_function_2d(comm, species, t, raw_folder, output_folder, sample_locations)

if (rank==0):
    t_end = MPI.Wtime()
    t_elapsed = t_end - t_start
    print('Total time;')
    print(t_elapsed)

