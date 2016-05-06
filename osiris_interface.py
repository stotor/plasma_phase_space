# Routines for loading data from OSIRIS

import numpy as np
from mpi4py import MPI
import h5py

def get_RAW(folder, species, t):
    """Load raw data, returns h5py File object."""
    filename = folder + "/MS/RAW/" + species + "/RAW-" + species + "-" + str(t).zfill(6) + ".h5"
    raw = h5py.File(filename)
    return raw

def get_field(folder, field_type, field_name, t, species=0):
    filename = 0
    if (field_type=="FLD"):
        filename = folder + "/MS/FLD/" + field_name + "/" + field_name + "-" + str(t).zfill(6) + ".h5"
    elif (field_type=="DENSITY" or field_type=="UDIST"):
        filename = folder + "/MS/" + field_type + "/" + species + "/" + field_name + "/" + field_name + "-" + species + "-" + str(t).zfill(6) + ".h5"
    file_h5 = h5py.File(filename)
    field = file_h5[field_name].value
    file_h5.close()
    return field

def get_field_extent(folder, field_type, field_name, t, species=0):
    """Returns: field_extent = [x1_min, x1_max, x2_min, x2_max]"""
    filename = 0
    if (field_type=="FLD"):
        filename = folder + "/MS/FLD/" + field_name + "/" + field_name + "-" + str(t).zfill(6) + ".h5"
    elif (field_type=="DENSITY" or field_type=="UDIST"):
        filename = folder + "/MS/" + field_type + "/" + species + "/" + field_name + "/" + field_name + "-" + species + "-" + str(t).zfill(6) + ".h5"
    file_h5 = h5py.File(filename)
    x1_min = file_h5["AXIS"]["AXIS1"][0]
    x1_max = file_h5["AXIS"]["AXIS1"][1]
    x2_min = file_h5["AXIS"]["AXIS2"][0]
    x2_max = file_h5["AXIS"]["AXIS2"][1]
    field_extent = [x1_min, x1_max, x2_min, x2_max]
    file_h5.close()
    return field_extent

def get_HIST_fld_ene(folder, field_name):
    energy = np.loadtxt(folder + "/HIST/fld_ene", skiprows=1)
    field_column = {'b1': 2, 'b2': 3, 'b3': 4, 'e1': 5, 'e2': 6, 'e3': 7}
    column = field_column[field_name]
    return energy[:, column]

def get_HIST_time(folder):
    energy = np.loadtxt(folder + "/HIST/fld_ene", skiprows=1)
    t_array = energy[:,1]
    return t_array

def get_HIST_timesteps(folder):
    energy = np.loadtxt(folder + "/HIST/fld_ene", skiprows=1)
    t_array = energy[:,0]
    return t_array

def create_position_array(raw, i_start, i_end):
    n_particles = i_end - i_start
    position = np.zeros([n_particles, 2])
    for i in range(2):
        position[:, i] = raw["x" + str(i+1)][i_start:i_end]
    return position

def create_velocity_array(raw, i_start, i_end):
    n_particles = i_end - i_start
    velocity = np.zeros([n_particles, 3])
    for i in range(3):
        velocity[:, i] = raw["p" + str(i+1)][i_start:i_end]
    gamma = np.sqrt(1.0 + np.sum(velocity**2, axis=1))
    velocity[:,0] = velocity[:,0] / gamma
    velocity[:,1] = velocity[:,1] / gamma
    velocity[:,2] = velocity[:,2] / gamma
    return velocity

def create_momentum_array(raw, i_start, i_end):
    n_particles = i_end - i_start
    momentum = np.zeros([n_particles, 3])
    for i in range(3):
        momentum[:, i] = raw["p" + str(i+1)][i_start:i_end]
    return momentum

def create_position_array_raw_unsorted(raw, my_particle_indices, lagrangian_sorting_keys):
    n_particles = len(my_particle_indices)
    position = np.zeros([n_particles, 2])
    for i in range(2):
        position[:, i] = raw["x" + str(i+1)][my_particle_indices][lagrangian_sorting_keys]
    return position

def create_velocity_array_raw_unsorted(raw, my_particle_indices, lagrangian_sorting_keys):
    n_particles = len(my_particle_indices)
    velocity = np.zeros([n_particles, 3])
    for i in range(3):
        velocity[:, i] = raw["p" + str(i+1)][my_particle_indices][lagrangian_sorting_keys]
    gamma = np.sqrt(1.0 + np.sum(velocity**2, axis=1))
    velocity[:,0] = velocity[:,0] / gamma
    velocity[:,1] = velocity[:,1] / gamma
    velocity[:,2] = velocity[:,2] / gamma
    return velocity

def create_momentum_array_raw_unsorted(raw, my_particle_indices, lagrangian_sorting_keys):
    n_particles = len(my_particle_indices)
    momentum = np.zeros([n_particles, 3])
    for i in range(3):
        momentum[:, i] = raw["p" + str(i+1)][my_particle_indices][lagrangian_sorting_keys]
    return momentum

def osiris_tag_to_lagrangian(osiris_id, n_cell_proc_x, n_cell_proc_y, n_ppc_x, n_ppc_y):
    """Converts the OSIRIS particle ID within a processor, tag[i,1], into the 
    Lagrangian coordinate within that processors subdomain."""

    # Convert to zero based indexing
    osiris_id = osiris_id - 1
    
    n_ppc = n_ppc_x * n_ppc_y

    cell = osiris_id / n_ppc
    cell_x = cell / (n_cell_proc_y)
    cell_y = cell - (cell_x * n_cell_proc_y)

    sub_cell = osiris_id - cell * n_ppc
    sub_cell_x = sub_cell / n_ppc_y
    sub_cell_y = sub_cell - (sub_cell_x * n_ppc_y)

    lagrangian_x = cell_x * n_ppc_x + sub_cell_x
    lagrangian_y = cell_y * n_ppc_y + sub_cell_y

    # Lagrangian coordinate, with x increasing fastest
    lagrangian = lagrangian_y * n_cell_proc_x * n_ppc_x + lagrangian_x
    
    return lagrangian

def save_raw_sorted_serial(input_filename, output_filename):
    # Load raw data to be sorted
    f_input = h5py.File(input_filename, 'r')

    n_cell_x = f_input.attrs['NX'][0]
    n_cell_y = f_input.attrs['NX'][1]
    
    n_p_total = f_input['x1'].shape[0]
    n_ppc = n_p_total / (n_cell_x * n_cell_y)
    n_ppc_x = int(np.sqrt(n_ppc))
    n_ppc_y = n_ppc_x

    n_proc_x = f_input.attrs['PAR_NODE_CONF'][0]
    n_proc_y = f_input.attrs['PAR_NODE_CONF'][1]
    n_cell_proc_x = n_cell_x / n_proc_x
    n_cell_proc_y = n_cell_y / n_proc_y
    n_ppp = (n_ppc_x * n_ppc_y) * (n_cell_proc_x * n_cell_proc_y)
    
    x1 = f_input['x1'][:]
    x2 = f_input['x2'][:]
    p1 = f_input['p1'][:]
    p2 = f_input['p2'][:]
    p3 = f_input['p3'][:]
    tag = f_input['tag'][:]

    keys = f_input.attrs.keys()
    values = f_input.attrs.values()

    f_input.close()

    # Get memory location for saving
    processor = tag[:,0] - 1 # Convert to zero based indexing
    lagrangian_id = osiris_tag_to_lagrangian(tag[:,1], 
                                             n_cell_proc_x, 
                                             n_cell_proc_y, 
                                             n_ppc_x, 
                                             n_ppc_y)

    memory_index = processor * n_ppp + lagrangian_id
    sorted_indices = np.argsort(memory_index)

    x1 = x1[sorted_indices]
    x2 = x2[sorted_indices]
    p1 = p1[sorted_indices]
    p2 = p2[sorted_indices]
    p3 = p3[sorted_indices]
    processor = processor[sorted_indices]
    lagrangian_id = lagrangian_id[sorted_indices]

    # Save sorted raw data
    f_output = h5py.File(output_filename, 'w')
    
    for i in range(len(keys)):
        f_output.attrs.create(keys[i], values[i])

    f_output.create_dataset('x1', (n_p_total,), dtype='float32')
    f_output.create_dataset('x2', (n_p_total,), dtype='float32') 
    f_output.create_dataset('p1', (n_p_total,), dtype='float32') 
    f_output.create_dataset('p2', (n_p_total,), dtype='float32') 
    f_output.create_dataset('p3', (n_p_total,), dtype='float32') 
    f_output.create_dataset('processor', (n_p_total,), dtype='int32')
    f_output.create_dataset('lagrangian_id', (n_p_total,), dtype='int32')
    
    f_output['x1'][:] = x1[:]
    f_output['x2'][:] = x2[:]
    f_output['p1'][:] = p1[:]
    f_output['p2'][:] = p2[:]
    f_output['p3'][:] = p3[:]
    f_output['processor'][:] = processor[:]
    f_output['lagrangian_id'][:] = lagrangian_id[:]
    f_output.close()

    return

# def save_raw_sorted_parallel(comm, input_filename, output_filename):
#     rank = comm.Get_rank()

#     # Load raw data to be sorted
#     f_input = h5py.File(input_filename, 'r', driver='mpio', comm=comm)

#     n_cell_x = f_input.attrs['NX'][0]
#     n_cell_y = f_input.attrs['NX'][1]
    
#     n_p_total = f_input['x1'].shape[0]
#     n_ppc = n_p_total / (n_cell_x * n_cell_y)
#     n_ppc_x = int(np.sqrt(n_ppc))
#     n_ppc_y = n_ppc_x

#     n_proc_x = f_input.attrs['PAR_NODE_CONF'][0]
#     n_proc_y = f_input.attrs['PAR_NODE_CONF'][1]
#     n_cell_proc_x = n_cell_x / n_proc_x
#     n_cell_proc_y = n_cell_y / n_proc_y
#     n_ppp = (n_ppc_x * n_ppc_y) * (n_cell_proc_x * n_cell_proc_y)

#     my_particle_indices = np.where(
    
#     x1 = f_input['x1'][:]
#     x2 = f_input['x2'][:]
#     p1 = f_input['p1'][:]
#     p2 = f_input['p2'][:]
#     p3 = f_input['p3'][:]
#     tag = f_input['tag'][:]

#     keys = f_input.attrs.keys()
#     values = f_input.attrs.values()

#     f_input.close()

#     # Get memory location for saving
#     processor = tag[:,0] - 1 # Convert to zero based indexing
#     lagrangian_id = osiris_tag_to_lagrangian(tag[:,1], 
#                                              n_cell_proc_x, 
#                                              n_cell_proc_y, 
#                                              n_ppc_x, 
#                                              n_ppc_y)

#     memory_index = processor * n_ppp + lagrangian_id
#     sorted_indices = np.argsort(memory_index)

#     x1 = x1[sorted_indices]
#     x2 = x2[sorted_indices]
#     p1 = p1[sorted_indices]
#     p2 = p2[sorted_indices]
#     p3 = p3[sorted_indices]
#     processor = processor[sorted_indices]
#     lagrangian_id = lagrangian_id[sorted_indices]

#     # Save sorted raw data
#     f_output = h5py.File(output_filename, 'w')
    
#     for i in range(len(keys)):
#         f_output.attrs.create(keys[i], values[i])

#     f_output.create_dataset('x1', (n_p_total,), dtype='float32')
#     f_output.create_dataset('x2', (n_p_total,), dtype='float32') 
#     f_output.create_dataset('p1', (n_p_total,), dtype='float32') 
#     f_output.create_dataset('p2', (n_p_total,), dtype='float32') 
#     f_output.create_dataset('p3', (n_p_total,), dtype='float32') 
#     f_output.create_dataset('processor', (n_p_total,), dtype='int32')
#     f_output.create_dataset('lagrangian_id', (n_p_total,), dtype='int32')
    
#     f_output['x1'][:] = x1[:]
#     f_output['x2'][:] = x2[:]
#     f_output['p1'][:] = p1[:]
#     f_output['p2'][:] = p2[:]
#     f_output['p3'][:] = p3[:]
#     f_output['processor'][:] = processor[:]
#     f_output['lagrangian_id'][:] = lagrangian_id[:]
#     f_output.close()

#     return
