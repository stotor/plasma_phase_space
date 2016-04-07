from mpi4py import MPI

import osiris_interface as oi
import utilities

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Parameters to be supplied for each run
simulation_folder = '/Users/stotor/Desktop/weibel_5e-1/'

input_folder = '/Users/stotor/Desktop/weibel_5e-1/MS/RAW/electrons_a/'
output_folder = '/Users/stotor/Desktop/weibel_5e-1/MS/RAW_SORTED/electrons_a/'

species = 'electrons_a'

utilities.ensure_folder_exists(output_folder)

t_array = oi.get_HIST_time(simulation_folder)

for t in range(0,len(t_array), 25):
    if (rank==0):
        print(t)
        t_start = MPI.Wtime()

    input_filename = input_folder + 'RAW-' + species + '-' + str(t).zfill(6) + '.h5'
    output_filename = output_folder + 'RAW-' + species + '-' + str(t).zfill(6) + '.h5'
    oi.save_raw_sorted_parallel(comm, input_filename, output_filename)

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print(t_elapsed)

    
