from mpi4py import MPI

import osiris_interface as oi
import utilities
import sys

if (len(sys.argv)!=3):
    print('Usage:\n    python sort_raw_serial.py <simulation_folder> <species>')
    sys.exit()

simulation_folder = sys.argv[1]
species = sys.argv[2]

input_folder = simulation_folder + '/MS/RAW/' + species + '/'
output_folder = simulation_folder + '/MS/RAW_SORTED/' + species + '/'
utilities.ensure_folder_exists(output_folder)

t_array = oi.get_HIST_time(simulation_folder)
n_t = len(t_array)

for t in range(n_t):
    print(t)
    t_start = MPI.Wtime()

    input_filename = input_folder + 'RAW-' + species + '-' + str(t).zfill(6) + '.h5'
    output_filename = output_folder + 'RAW-' + species + '-' + str(t).zfill(6) + '.h5'
    oi.save_raw_sorted_serial(input_filename, output_filename)

    t_end = MPI.Wtime()
    t_elapsed = t_end - t_start
    print(t_elapsed)

    
