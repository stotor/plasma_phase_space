# Routines for loading data from OSIRIS

import h5py
import numpy as np

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
    energy = np.loadtxt(folder + "HIST/fld_ene", skiprows=1)
    t_array = energy[:,1]
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
