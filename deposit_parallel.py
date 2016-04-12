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

if (not (len(sys.argv)!=5 or len(sys.argv)!=6)):
    if (rank==0):
        print('Usage:\n    mpirun -n <size> python deposit_parallel.py <simulation_folder> <species> <deposit_n_x> <deposit_n_y> [<n_t>]')
    sys.exit()

simulation_folder = sys.argv[1]
species = sys.argv[2]
deposit_n_x = int(sys.argv[3])
deposit_n_y = int(sys.argv[4])

raw_sorted_folder = simulation_folder + '/MS/RAW_SORTED/' + species + '/'
output_folder = simulation_folder + '/' + str(deposit_n_x) + 'x' + str(deposit_n_y) + '/'

t_array = oi.get_HIST_time(simulation_folder)

if (len(sys.argv)==6):
    n_t = int(sys.argv[5])
else:
    n_t = len(t_array)

for t in range(n_t):
    if (rank==0):
        print(t)
        t_start = MPI.Wtime()

    pic.save_cic_fields_parallel(comm, species, t, raw_sorted_folder, output_folder, deposit_n_x, deposit_n_y)
    tri.save_triangle_fields_parallel(comm, species, t, raw_sorted_folder, output_folder, deposit_n_x, deposit_n_y)

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print(t_elapsed)

