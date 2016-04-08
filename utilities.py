# Miscellaneous utility functions

import numpy as np
import h5py
import subprocess
import os
import glob

def create_pdf(name):
    subprocess.call(["convert", name + "-*.png", name + ".pdf"])
    subprocess.call(["rm"] + glob.glob(name + "-*.png"))
    return

def ensure_folder_exists(directory, rank=0):
    if (rank==0):
        if (not os.path.exists(directory)):
            os.makedirs(directory)
    return

def save_density_field(folder, field_name, species, t, data):
    save_folder = folder + '/' + species + '/' + field_name + '/'
    ensure_folder_exists(save_folder)
    filename = save_folder + field_name + '-' + species + '-' + str(t).zfill(6) + '.h5'
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset(field_name, data=data)
    h5f.close()
    return

def load_density_field(folder, field_name_list, species_list, t):
    if (type(species_list)!=list):
        species_list = [species_list]
    if (type(field_name_list)!=list):
        field_name_list = [field_name_list]
    field = None
    for species in species_list:
            for field_name in field_name_list:
                save_folder = folder + '/' + species + '/' + field_name + '/'
                filename = save_folder + field_name + '-' + species + '-' + str(t).zfill(6) + '.h5'
                h5f = h5py.File(filename, 'r')
                data = h5f[field_name][:]
                h5f.close()
                if (field==None):
                    field = data
                else:
                    field = field + data
    return field
