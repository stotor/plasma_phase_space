import numpy as np
from mpi4py import MPI
import h5py

import osiris_interface as oi
import pic_calculations as pic
import triangle_calculations as tri
import utilities

comm = MPI.COMM_WORLD

simulation_folder = '/Users/stotor/Desktop/weibel_5e-1/'
output_folder = simulation_folder + '/test_parallel/'
species = 'electrons_a'

t_array = oi.get_HIST_time(simulation_folder)
n_t = len(t_array)
t = 100

pic.save_cic_fields_parallel(comm, species, t, simulation_folder, output_folder)
tri.save_triangle_fields_parallel(comm, species, t, simulation_folder, output_folder)
