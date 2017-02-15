# Miscellaneous utility functions

import numpy as np
import h5py
import subprocess
import os
import glob

t_axis_string = '$\omega_{pe} t$'

x_axis_string = '$x / (c / \omega_{pe})$'
y_axis_string = '$y / (c / \omega_{pe})$'
z_axis_string = '$z / (c / \omega_{pe})$'

kx_axis_string = '$k_{x} / (2 \pi / L_{x})$'
ky_axis_string = '$k_{y} / (2 \pi / L_{y})$'
kz_axis_string = '$k_{z} / (2 \pi / L_{z})$'

def create_time_string(t_array, t):
    return '$\omega_{pe}t = ' + str(round(t_array[t], 2)) + '$'

def create_pdf(name):
    subprocess.call(["convert", name + "-*.png", name + ".pdf"])
    subprocess.call(["rm"] + glob.glob(name + "-*.png"))
    return

def combine_pdfs(name):
    # pdftk *.pdf cat output newfile.pdf
    subprocess.call(["pdftk"] + glob.glob(name + "-*.pdf") +
                    [ "cat", "output", name + ".pdf"])
    subprocess.call(["rm"] + glob.glob(name + "-*.pdf"))
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

def save_density_field_attrs(folder, field_name, species, t, time, data, axis):
    save_folder = folder + '/' + species + '/' + field_name + '/'
    ensure_folder_exists(save_folder)
    filename = save_folder + field_name + '-' + species + '-' + str(t).zfill(6) + '.h5'
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset(field_name, data=data)
    if field_name[:6] == 'charge':
        h5f[field_name].attrs['LONG_NAME'] = r'\rho'
        h5f[field_name].attrs['UNITS'] = 'e \omega_p^3/ c^3'
    elif field_name[0] == 'j':
        h5f[field_name].attrs['LONG_NAME'] = r'j_{' + field_name[1] + '}'
        h5f[field_name].attrs['UNITS'] = 'n_{0}'
    elif field_name[:3] == 'ufl':
        h5f[field_name].attrs['LONG_NAME'] = r'u_{fl' + field_name[3] + '}'
        h5f[field_name].attrs['UNITS'] = 'c'
    elif field_name[:3] == 'uth':
        h5f[field_name].attrs['LONG_NAME'] = r'u_{th' + field_name[3] + '}'
        h5f[field_name].attrs['UNITS'] = 'c'
    elif field_name[:7] == 'streams':
        h5f[field_name].attrs['LONG_NAME'] = 'streams'
        h5f[field_name].attrs['UNITS'] = '\#'
    else:
        h5f[field_name].attrs['LONG_NAME'] = r'\rho'
        h5f[field_name].attrs['UNITS'] = 'e \omega_p^3/ c^3'
    
    axis = np.array(axis)
    dim = axis.shape[0]
    for i in range(dim):
        h5f.create_dataset('AXIS/AXIS' + str(i+1), data=axis[i,:])
        h5f['AXIS/AXIS' + str(i+1)].attrs['LONG_NAME'] = 'x_' + str(i+1)
        h5f['AXIS/AXIS' + str(i+1)].attrs['NAME'] = 'x' + str(i+1)
        h5f['AXIS/AXIS' + str(i+1)].attrs['TYPE'] = 'linear'
        h5f['AXIS/AXIS' + str(i+1)].attrs['UNITS'] = 'c / \omega_p'
        
    h5f.attrs['ITER'] = t
    h5f.attrs['NAME'] = field_name
    h5f.attrs['TIME'] = time
    h5f.attrs['TIME UNITS'] = '1 / \omega_p'
    h5f.attrs['TYPE'] = 'grid'
    h5f.attrs['XMAX'] = axis[:,1]
    h5f.attrs['XMIN'] = axis[:,0]
    
    # Below are attributes that OSIRIS usually prints with a density
    # file but do not seem to be necessary to get the IDL routines working
    #h5f.attrs['NX'] = [64,64,64]
    #h5f.attrs['PAR_NODE_CONF'] = [2,2,2]
    #h5f.attrs['PAR_NX_X1'] = [32,32]
    #h5f.attrs['PAR_NX_X2'] = [32,32]
    #h5f.attrs['PAR_NX_X3'] = [32,32]
    #h5f.attrs['PERIODIC'] = [1,1,1]
    #h5f.attrs['MOVE C'] = [0,0,0]
    #h5f.attrs['DT'] = 1.0
    
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

def int_nth_root(val, n):
    root = int(val**(1.0/n))
    while (root**n!=val):
        root = root + 1
    return root
