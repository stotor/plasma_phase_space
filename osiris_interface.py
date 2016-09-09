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
    if ((field_name[-5:]=='-savg') or (field_name[-5:]=='-tavg')):
        field_name = field_name[:-5]
    field = file_h5[field_name][:]
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

def momentum_to_velocity(particle_momentum):
    n_particles = particle_momentum.shape[0]
    particle_velocities = np.zeros([n_particles, 3])

    gamma = np.sqrt(1.0 + np.sum(particle_momentum**2, axis=1))
    particle_velocities[:,0] = particle_momentum[:,0] / gamma
    particle_velocities[:,1] = particle_momentum[:,1] / gamma
    particle_velocities[:,2] = particle_momentum[:,2] / gamma
    return particle_velocities

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

def osiris_tag_to_lagrangian_3d(osiris_id, n_cell_x, n_cell_y, n_cell_z, n_ppc_x, n_ppc_y, n_ppc_z):
    # Convert to zero based indexing
    osiris_id = osiris_id - 1
    
    n_ppc = n_ppc_x * n_ppc_y * n_ppc_z

    cell = osiris_id // n_ppc
    cell_x = cell // (n_cell_y * n_cell_z)
    cell_y = (cell - cell_x * (n_cell_y * n_cell_z)) // n_cell_z
    cell_z = cell - cell_x * (n_cell_y * n_cell_z) - cell_y * n_cell_z

    sub_cell = osiris_id - cell * n_ppc
    sub_cell_x = sub_cell // (n_ppc_y * n_ppc_z)
    sub_cell_y = (sub_cell - sub_cell_x * (n_ppc_y * n_ppc_z)) // n_ppc_z
    sub_cell_z = sub_cell - sub_cell_x * (n_ppc_y * n_ppc_z) - sub_cell_y * n_ppc_z

    lagrangian_x = cell_x * n_ppc_x + sub_cell_x
    lagrangian_y = cell_y * n_ppc_y + sub_cell_y
    lagrangian_z = cell_z * n_ppc_z + sub_cell_z
    
    # Lagrangian coordinate, with x increasing fastest
    lagrangian = lagrangian_z * (n_cell_x * n_ppc_x * n_cell_y * n_ppc_y) + lagrangian_y * (n_cell_x * n_ppc_x) + lagrangian_x
    
    return lagrangian

def factor_field(field, n_ave):
    field = field.reshape(field.shape[0]//n_ave, n_ave, field.shape[1]//n_ave, n_ave)
    field = np.average(field, axis=3)
    field = np.average(field, axis=1)
    return field

def calculate_power_spectrum(field):
    field = np.fft.fft2(field)
    field = field * np.conj(field)
    field = field.real
    field[np.where(field < 10**(-16))] = 0
    return field
