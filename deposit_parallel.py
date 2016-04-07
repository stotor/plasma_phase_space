import numpy as np
from mpi4py import MPI
import h5py

import osiris_interface as oi
import pic_calculations as pic
import triangle_calculations as tri
import utilities

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

simulation_folder = '/Users/stotor/Desktop/weibel_5e-1/'
raw_sorted_folder = simulation_folder + '/MS/RAW_SORTED/electrons_a/'
output_folder = simulation_folder + '/400x400/'
species = 'electrons_a'

t_array = oi.get_HIST_time(simulation_folder)
n_t = len(t_array)

deposit_n_x = 400
deposit_n_y = 400

for t in range(0, n_t, 25):
    if (rank==0):
        print(t)
        t_start = MPI.Wtime()

    pic.save_cic_fields_parallel(comm, species, t, raw_sorted_folder, output_folder, deposit_n_x, deposit_n_y)
    tri.save_triangle_fields_parallel(comm, species, t, raw_sorted_folder, output_folder, deposit_n_x, deposit_n_y)

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print(t_elapsed)

