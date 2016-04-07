from mpi4py import MPI

import sort_raw_parallel

comm = MPI.COMM_WORLD

# Parameters to be supplied for each run
base_folder = '/Users/stotor/Desktop/weibel_5e-1/MS/RAW/electrons_a/'
species = 'electrons_a'
n_t = 143

for t in [100]:#range(n_t):
    print(t)
    input_filename = base_folder + 'RAW-' + species + '-' + str(t).zfill(6) + '.h5'
    output_filename = base_folder + 'SORTED_RAW-' + species + '-' + str(t).zfill(6) + '.h5'
    sort_raw_parallel.save_raw_sorted_parallel(comm, input_filename, output_filename)
