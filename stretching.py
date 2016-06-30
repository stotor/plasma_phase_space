# Routines for calculating triangle quantities such as area

import numpy as np
from mpi4py import MPI
import h5py
import sys

import utilities
import osiris_interface as oi
import ship
import extend

def length_in_box(a, b, l_max):
    """Account for periodic boundaries by choosing the shortest possible distance between two points along a given axis."""
    length = b - a
    if (length > (l_max / 2.0)):
        length = length - l_max
    if (length < (-1.0 * l_max / 2.0)):
        length = length + l_max
    return length

def calculate_axis_separations(v_ll, v_lr, v_ul, v_ur, l_x, l_y):
    vertical_y = length_in_box(v_ll[0], v_ul[0], l_x)
    vertical_x = length_in_box(v_ll[1], v_ul[1], l_y)
    horizontal_y = length_in_box(v_ll[0], v_lr[0], l_x)
    horizontal_x = length_in_box(v_ll[1], v_lr[1], l_y)
    diagonal_a_y = length_in_box(v_ll[0], v_ur[0], l_x)
    diagonal_a_x = length_in_box(v_ll[1], v_ur[1], l_y)
    diagonal_b_y = length_in_box(v_ul[0], v_lr[0], l_x)
    diagonal_b_x = length_in_box(v_ul[1], v_lr[1], l_y)

    separations_x = []
    separations_x.append(vertical_x)
    separations_x.append(horizontal_x)
    separations_x.append(diagonal_a_x)
    separations_x.append(diagonal_b_x)

    separations_y = []
    separations_y.append(vertical_y)
    separations_y.append(horizontal_y)
    separations_y.append(diagonal_a_y)
    separations_y.append(diagonal_b_y)

    return [separations_y, separations_x]

def get_axis_separation_distribution(particle_positions_extended, l_x, l_y):
    n_l_y = particle_positions_extended.shape[0]-1
    n_l_x = particle_positions_extended.shape[1]-1

    separations_x = []
    separations_y = []

    for y in range(n_l_y):
        for x in range(n_l_x):
            v_ll = particle_positions_extended[y, x]
            v_lr = particle_positions_extended[y, x+1]
            v_ul = particle_positions_extended[y+1, x]
            v_ur = particle_positions_extended[y+1, x+1]

            separations_cell = calculate_axis_separations(v_ll, v_lr, v_ul, v_ur, l_x, l_y)

            separations_y.extend(separations_cell[0])
            separations_x.extend(separations_cell[1])

    return [separations_y, separations_x]

def calculate_tracer_separation_2d(comm, raw_filename, bin_edges):

    f_input = h5py.File(raw_filename, 'r', driver='mpio', comm=comm)

    n_proc_x = f_input.attrs['PAR_NODE_CONF'][0]
    n_proc_y = f_input.attrs['PAR_NODE_CONF'][1]
    n_cell_x = f_input.attrs['NX'][0]

    cartcomm = comm.Create_cart([n_proc_y, n_proc_x], periods=[True,True])
    rank = cartcomm.Get_rank()
    size = cartcomm.Get_size()
    
    n_p_total = f_input['x1'].shape[0]
    n_ppp = n_p_total / size
    n_ppp_x = utilities.int_nth_root(n_ppp, 2)
    n_ppp_y = n_ppp_x

    axis = np.zeros([2,2])

    axis[0,0] = f_input.attrs['XMIN'][0]
    axis[0,1] = f_input.attrs['XMAX'][0]
    axis[1,0] = f_input.attrs['XMIN'][1]
    axis[1,1] = f_input.attrs['XMAX'][1]
    
    l_x = axis[0,1] - axis[0,0]
    l_y = axis[1,1] - axis[1,0]

    dx = l_x / n_cell_x

    time = f_input.attrs['TIME']

    # Get particle data for this processor's lagrangian subdomain
    particle_positions = ship.ship_particle_data(cartcomm, f_input, 2)[0]

    f_input.close()

    particle_positions = particle_positions.reshape(n_ppp_y, n_ppp_x, 2)
    particle_positions_extended = extend.extend_lagrangian_quantity_2d(cartcomm, particle_positions)

    [separations_y, separations_x] = get_axis_separation_distribution(particle_positions_extended, l_x, l_y)
    separations_x = np.abs(separations_x)
    separations_y = np.abs(separations_y)

    max_x = np.amax(separations_x)
    min_x = np.amin(separations_x)
    mean_x = np.mean(separations_x)

    max_y = np.amax(separations_y)
    min_y = np.amin(separations_y)
    mean_y = np.mean(separations_y)
    
    # Histogram separation distributions
    hist_x = np.histogram(separations_x, bin_edges)[0]
    hist_y = np.histogram(separations_y, bin_edges)[0]

    hist_x_total = np.zeros_like(hist_x)
    hist_y_total = np.zeros_like(hist_y)

    max_x_global = np.zeros(1, dtype='double')
    min_x_global = np.zeros(1, dtype='double')
    mean_x_global = np.zeros(1, dtype='double')

    max_y_global = np.zeros(1, dtype='double')
    min_y_global = np.zeros(1, dtype='double')
    mean_y_global = np.zeros(1, dtype='double')

    # Sum reduction of histograms
    cartcomm.Reduce([hist_x, MPI.DOUBLE], [hist_x_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([hist_y, MPI.DOUBLE], [hist_y_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

    cartcomm.Reduce([max_x, MPI.DOUBLE], [max_x_global, MPI.DOUBLE], op = MPI.MAX, root = 0)
    cartcomm.Reduce([min_x, MPI.DOUBLE], [min_x_global, MPI.DOUBLE], op = MPI.MIN, root = 0)
    cartcomm.Reduce([mean_x, MPI.DOUBLE], [mean_x_global, MPI.DOUBLE], op = MPI.SUM, root = 0)
    if (rank==0):
        mean_x_global = mean_x_global / float(size)

    cartcomm.Reduce([max_y, MPI.DOUBLE], [max_y_global, MPI.DOUBLE], op = MPI.MAX, root = 0)
    cartcomm.Reduce([min_y, MPI.DOUBLE], [min_y_global, MPI.DOUBLE], op = MPI.MIN, root = 0)
    cartcomm.Reduce([mean_y, MPI.DOUBLE], [mean_y_global, MPI.DOUBLE], op = MPI.SUM, root = 0)
    if (rank==0):
        mean_y_global = mean_y_global / float(size)

    statistics = {}
    if (rank==0):
        statistics['max_x'] = max_x_global
        statistics['min_x'] = min_x_global
        statistics['mean_x'] = mean_x_global
        statistics['max_y'] = max_y_global
        statistics['min_y'] = min_y_global
        statistics['mean_y'] = mean_y_global
        statistics['l_x'] = l_x
        statistics['dx'] = dx
        
    return [hist_y_total, hist_x_total, statistics]

# Main program

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if (len(sys.argv)!=6):
    if (rank==0):
        print('Usage:\n    mpirun -n <size> python stretching.py <simulation_folder> <species> <dist_min> <dist_max> <n_bins>')
    sys.exit()

simulation_folder = sys.argv[1]
species = sys.argv[2]
dist_min = float(sys.argv[3])
dist_max = float(sys.argv[4])
n_bins = int(sys.argv[5])

bin_edges = np.logspace(np.log10(dist_min), np.log10(dist_max), (n_bins+1))

raw_folder = simulation_folder + '/MS/RAW/' + species + '/'

t_array = oi.get_HIST_time(simulation_folder)
n_t = len(t_array)
timesteps = range(n_t)

separation_x_evolution = np.zeros([n_bins, n_t])
separation_y_evolution = np.zeros([n_bins, n_t])

max_x = np.zeros(n_t, dtype = 'double')
min_x = np.zeros(n_t, dtype = 'double')
mean_x = np.zeros(n_t, dtype = 'double')
max_y = np.zeros(n_t, dtype = 'double')
min_y = np.zeros(n_t, dtype = 'double')
mean_y = np.zeros(n_t, dtype = 'double')

for t in timesteps:
    if (rank==0):
        t_start = MPI.Wtime()
        print('Starting timestep ' + str(t))

    raw_filename = raw_folder + "/RAW-" + species + "-" + str(t).zfill(6) + ".h5"
    [hist_y_total, hist_x_total, statistics] = calculate_tracer_separation_2d(comm, raw_filename, bin_edges)

    if (rank==0):
        separation_x_evolution[:,t] = hist_x_total
        separation_y_evolution[:,t] = hist_y_total
        max_x[t] = statistics['max_x']
        min_x[t] = statistics['min_x']
        mean_x[t] = statistics['mean_x']
        max_y[t] = statistics['max_y']
        min_y[t] = statistics['min_y']
        mean_y[t] = statistics['mean_y']

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Total time for timestep:')
        print(t_elapsed)

if (rank==0):
    save_folder = './separation_distribution/'
    utilities.ensure_folder_exists(save_folder)
    save_filename = save_folder + species + '.h5'
    h5f = h5py.File(save_filename, 'w')
    h5f.create_dataset('separation_x_evolution', data=separation_x_evolution)
    h5f.create_dataset('separation_y_evolution', data=separation_y_evolution)
    h5f.create_dataset('time', data=t_array)
    h5f.create_dataset('bin_edges', data=bin_edges)
    h5f.create_dataset('max_x', data=max_x)
    h5f.create_dataset('min_x', data=min_x)
    h5f.create_dataset('mean_x', data=mean_x)
    h5f.create_dataset('max_y', data=max_y)
    h5f.create_dataset('min_y', data=min_y)
    h5f.create_dataset('mean_y', data=mean_y)
    h5f.create_dataset('l_x', data=statistics['l_x'])
    h5f.create_dataset('dx', data=statistics['dx'])

    h5f.close()
