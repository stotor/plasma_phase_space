# Routines for PIC calculations such as charge and current deposits.

import numpy as np
import math
import ctypes
from mpi4py import MPI
import h5py

import osiris_interface as oi
import utilities

def deposit_particle_2d_0(position, field, charge, n_x, n_y, dx):
    """NGP deposit onto a 2D field.  Assumes dx = dy, and that x_min = y_min = 0.0."""
    x = position[1] / dx - 0.5
    y = position[0] / dx - 0.5
    
    i_lower = math.floor(x)
    j_lower = math.floor(y)
    
    if ((x-i_lower)<0.5):
        i_ngp = i_lower
    else:
        i_ngp = i_lower+1
        
    if ((y-j_lower)<0.5):
        j_ngp = j_lower
    else:
        j_ngp = j_lower+1
    
    field[j_ngp%n_y, i_ngp%n_x] += charge
    
    return field

def deposit_particle_2d_1(position, field, charge, n_x, n_y, dx):
    """CIC deposit onto a 2D field.  Assumes dx = dy, and that x_min = y_min = 0.0."""
    x = position[1] / dx - 0.5
    y = position[0] / dx - 0.5
    
    # Indices of the lower left gridpoint of the cell the particle is in
    i_ll = math.floor(x)
    j_ll = math.floor(y)

    dx = x - (i_ll + 0.5)
    dy = y - (j_ll + 0.5)

    wx = np.zeros(2, dtype='double')
    wy = np.zeros_like(wx)

    wx[0] = 0.5 - dx
    wx[1] = 0.5 + dx

    wy[0] = 0.5 - dy
    wy[1] = 0.5 + dy

    for i in range(2):
        for j in range(2):
            field[(j_ll+j)%n_y, (i_ll+i)%n_x] += charge * wx[i] * wy[j]

    return field

def deposit_particle_2d_2(position, field, charge, n_x, n_y, dx):
    """Second order deposit onto a 2D field.  Assumes dx = dy, and that x_min = y_min = 0.0."""
    # Normalize position to grid spacing
    x = position[1] / dx - 0.5
    y = position[0] / dx - 0.5
    
    i_ll = math.floor(x)
    j_ll = math.floor(y)
    
    if ((x-i_ll)<0.5):
        i_ngp = i_ll
    else:
        i_ngp = i_ll+1
        
    if ((y-j_ll)<0.5):
        j_ngp = j_ll
    else:
        j_ngp = j_ll+1

    dx = x - i_ngp
    dy = y - j_ngp

    wx = np.zeros(3, dtype='double')
    wy = np.zeros_like(wx)

    wx[0] = (1.0 - 2.0 * dx)**2 / 8.0
    wx[1] = 0.75 - dx**2
    wx[2] = (1.0 + 2.0 * dx)**2 / 8.0
    
    wy[0] = (1.0 - 2.0 * dy)**2 / 8.0
    wy[1] = 0.75 - dy**2
    wy[2] = (1.0 + 2.0 * dy)**2 / 8.0

    for i in range(3):
        for j in range(3):    
            field[(j_ngp-1+j)%n_y, (i_ngp-1+i)%n_x] += charge * wx[i] * wy[j]
    
    return field

def deposit_particle_2d_3(position, field, charge, n_x, n_y, dx):
    """Second order deposit onto a 2D field.  Assumes dx = dy, and that x_min = y_min = 0.0."""
    # Normalize position to grid spacing
    x = position[1] / dx - 0.5
    y = position[0] / dx - 0.5
                  
    i_ll = math.floor(x)
    j_ll = math.floor(y)
    
    if ((x-i_ll)<0.5):
        i_ngp = i_ll
    else:
        i_ngp = i_ll+1
        
    if ((y-j_ll)<0.5):
        j_ngp = j_ll
    else:
        j_ngp = j_ll+1

    dx = x - (i_ll+0.5)
    dy = y - (j_ll+0.5)

    wx = np.zeros(4, dtype='double')
    wy = np.zeros_like(wx)

    wx[0] = -(-0.5 + dx)**3 / 6.0
    wx[1] = (4.0 - 6.0 * (0.5 + dx)**2 + 3.0 * (0.5 + dx)**3) / 6.0
    wx[2] = (23.0 + 30.0 * dx - 12.0 * dx**2 - 24.0 * dx**3) / 48.0
    wx[3] = (0.5 + dx)**3 / 6.0

    wy[0] = -(-0.5 + dy)**3 / 6.0
    wy[1] = (4.0 - 6.0 * (0.5 + dy)**2 + 3.0 * (0.5 + dy)**3) / 6.0
    wy[2] = (23.0 + 30.0 * dy - 12.0 * dy**2 - 24.0 * dy**3) / 48.0
    wy[3] = (0.5 + dy)**3 / 6.0
    
    for i in range(4):
        for j in range(4):
            field[(j_ll-1+j)%n_y, (i_ll-1+i)%n_x] += charge * wx[i] * wy[j]
    
    return field

def deposit_particle_2d_4(position, field, charge, n_x, n_y, dx):
    """Second order deposit onto a 2D field.  Assumes dx = dy, and that x_min = y_min = 0.0."""
    # Normalize position to grid spacing
    x = position[1] / dx - 0.5
    y = position[0] / dx - 0.5
    
    i_lower = math.floor(x)
    j_lower = math.floor(y)
    
    if ((x-i_lower)<0.5):
        i_ngp = i_lower
    else:
        i_ngp = i_lower+1
        
    if ((y-j_lower)<0.5):
        j_ngp = j_lower
    else:
        j_ngp = j_lower+1

    dx = x - i_ngp
    dy = y - j_ngp

    wx = np.zeros(5, dtype='double')
    wy = np.zeros_like(wx)

    wx[0] = (1.0 - 2.0 * dx)**4 / 384.0
    wx[1] = (19.0 - 44.0 * dx + 24.0 * dx**2 + 16.0 * dx**3 - 16.0 * dx**4) / 96.0
    wx[2] = 0.5989583333333334 - (5.0 * dx**2) / 8.0 + dx**4 / 4.0
    wx[3] = (19.0 + 44.0 * dx + 24.0 * dx**2 - 16.0 * dx**3 - 16.0 * dx**4) / 96.0
    wx[4] = (1.0 + 2.0 * dx)**4 / 384.0

    wy[0] = (1.0 - 2.0 * dy)**4 / 384.0
    wy[1] = (19.0 - 44.0 * dy + 24.0 * dy**2 + 16.0 * dy**3 - 16.0 * dy**4) / 96.0
    wy[2] = 0.5989583333333334 - (5.0 * dy**2) / 8.0 + dy**4 / 4.0
    wy[3] = (19.0 + 44.0 * dy + 24.0 * dy**2 - 16.0 * dy**3 - 16.0 * dy**4) / 96.0
    wy[4] = (1.0 + 2.0 * dy)**4 / 384.0

    for i in range(5):
        for j in range(5):
            field[(j_ngp-2+j)%n_y, (i_ngp-2+i)%n_x] += charge * wx[i] * wy[j]
    
    return field

# Not working yet
def deposit_cic_3d_particle(position, field, charge, n_x, n_y, n_z, dx):
    """CIC deposit onto a 3D field.  Assumes dx = dy, and that x_min = y_min = 0.0."""
    # Shift to align the cell centers with the pixel centers
    x = position[2] / dx - 0.5
    y = position[1] / dx - 0.5
    z = position[0] / dx - 0.5
    
    # Indices of the lower corner gridpoint of the cubic cell the particle is in
    x_l = math.floor(x)
    y_l = math.floor(y)
    z_l = math.floor(y)
    
    x_rel = x - x_lower
    y_rel = y - y_lower
    z_rel = z - z_lower
    
    field[z_l%n_z, y_l%n_y, x_l%n_x] += charge * (1.0 - x_rel) * (1.0 - y_rel) * (1.0 - z_rel)
    field[(z_l+1)%n_z, y_l%n_y, x_l%n_x] += charge * (1.0 - x_rel) * (1.0 - y_rel) * z_rel

    field[z_l%n_z, (y_l+1)%n_y, x_l%n_x] += charge * (1.0 - x_rel) * y_rel * (1.0 - z_rel)
    field[(z_l+1)%n_z, (y_l+1)%n_y, x_l%n_x] += charge * (1.0 - x_rel) * y_rel * z_rel

    field[z_l%n_z, y_l%n_y, (x_l+1)%n_x] += charge * x_rel * (1.0 - y_rel) * (1.0 - z_rel)
    field[(z_l+1)%n_z, y_l%n_y, (x_l+1)%n_x] += charge * x_rel * (1.0 - y_rel) * z_rel

    field[z_l%n_z, (y_l+1)%n_y, (x_l+1)%n_x] += charge * x_rel * y_rel * (1.0 - z_rel)
    field[(z_l+1)%n_z, (y_l+1)%n_y, (x_l+1)%n_x] += charge * x_rel * y_rel * z_rel
    
    return field

def get_ngp(x_normalized):
    x_lower = int(math.floor(x_normalized))
    if (x_normalized - x_lower > 0.5):
        x_ngp = x_lower + 1
    else:
        x_ngp = x_lower
    return x_ngp

def deposit_species(particle_positions, field, particle_charges, n_x, n_y, dx, deposit_type):
    n_p = len(particle_positions)
    if (deposit_type=='ngp'):
        for i in range(n_p):
            position = particle_positions[i,:]
            charge = particle_charges[i]
            deposit_particle_2d_0(position, field, charge, n_x, n_y, dx)
    elif (deposit_type=='cic'):
        for i in range(n_p):
            position = particle_positions[i,:]
            charge = particle_charges[i]
            deposit_particle_2d_1(position, field, charge, n_x, n_y, dx)
    elif (deposit_type=='quadratic'):
        for i in range(n_p):
            position = particle_positions[i,:]
            charge = particle_charges[i]
            deposit_particle_2d_2(position, field, charge, n_x, n_y, dx)
    elif (deposit_type=='cubic'):
        for i in range(n_p):
            position = particle_positions[i,:]
            charge = particle_charges[i]
            deposit_particle_2d_3(position, field, charge, n_x, n_y, dx)
    elif (deposit_type=='quartic'):
        for i in range(n_p):
            position = particle_positions[i,:]
            charge = particle_charges[i]
            deposit_particle_2d_4(position, field, charge, n_x, n_y, dx)
    return field

def deposit_species_ctypes(particle_positions, field, particle_charges,
                           n_x, n_y, dx, deposit_type):
    lib = ctypes.cdll['/Users/stotor/Desktop/plasma_phase_space/analysis_v2/pic_deposits/pic_deposits.so']
    deposit_species = lib['deposit_species']

    n_p = particle_positions.shape[0]
    
    if (deposit_type=='ngp'):
        order = 0
    elif (deposit_type=='cic'):
        order = 1
    elif (deposit_type=='quadratic'):
        order = 2
    elif (deposit_type=='cubic'):
        order = 3
    elif (deposit_type=='quartic'):
        order = 4
    cell_width = dx

    c_double_p = ctypes.POINTER(ctypes.c_double)
    
    particle_positions_c = particle_positions.ctypes.data_as(c_double_p)
    field_c = field.ctypes.data_as(c_double_p)
    particle_charges_c = particle_charges.ctypes.data_as(c_double_p)
    n_p_c = ctypes.c_int(n_p)
    n_x_c = ctypes.c_int(n_x)
    n_y_c = ctypes.c_int(n_y)
    order_c = ctypes.c_int(order)
    cell_width_c = ctypes.c_double(cell_width)

    deposit_species(particle_positions_c, field_c, particle_charges_c, n_x_c, n_y_c,
                    n_p_c, cell_width_c, order_c)

    return 


def calculate_power_spectrum(comm, species, t, raw_folder, output_folder, n_k_x, n_k_y):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load raw data to be deposited
    input_filename = raw_folder + "/RAW-" + species + "-" + str(t).zfill(6) + ".h5"

    f_input = h5py.File(input_filename, 'r', driver='mpio', comm=comm)

    n_p_total = f_input['x1'].shape[0]
    n_ppp = n_p_total / size
    
    l_x = f_input.attrs['XMAX'][0] - f_input.attrs['XMIN'][0]
    l_y = f_input.attrs['XMAX'][1] - f_input.attrs['XMIN'][1]

    i_start = rank * n_ppp
    i_end = (rank + 1) * n_ppp

    # Get position array
    particle_positions = oi.create_position_array(f_input, i_start, i_end)

    time = f_input.attrs['TIME']

    f_input.close()

    # Calculate exact fourier transform
    n_k = n_k_x
    
    k_x = np.append(np.arange(0, (n_k/2), 1), np.arange(-(n_k/2), 0, 1))
    k_y = np.append(np.arange(0, (n_k/2), 1), np.arange(-(n_k/2), 0, 1))
    
    k_x = np.tile(k_x, n_k).reshape(n_k, n_k) * 2.0 * np.pi / l_x
    k_y = np.repeat(k_y, n_k).reshape(n_k, n_k) * 2.0 * np.pi / l_y

    chunk_size = 64
    indices = np.append(range(0, n_ppp, chunk_size), n_ppp)

    for i in range(len(indices)-1):
        start = indices[i]
        end = indices[i+1]

        phase = 1j * k_x[None,:,:] * particle_positions[start:end,0][:,None,None] \
                + 1j * k_y[None,:,:] * particle_positions[start:end,1][:,None,None]
        if (i==0):
            ft = np.sum(np.exp(phase), axis=0)
        else:
            ft += np.sum(np.exp(phase), axis=0)

    ft_r = np.copy(ft.real)
    ft_i = np.copy(ft.imag)

    # Reduce fourier transform
    ft_r_total = np.zeros_like(ft_r)
    ft_i_total = np.zeros_like(ft_i)

    comm.Reduce([ft_r, MPI.DOUBLE], [ft_r_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    comm.Reduce([ft_i, MPI.DOUBLE], [ft_i_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

    # Calculate power spectrum
    if (rank==0):
        ps = ft_r_total**2 + ft_i_total**2

    # Save power spectrum
    if (rank==0):
        save_folder = output_folder + '/power_spectrum/' + species + '/'
        utilities.ensure_folder_exists(save_folder)
        filename = save_folder + 'power_spectrum-' + species + '-' + str(t).zfill(6) + '.h5'
        h5f = h5py.File(filename, 'w')
        h5f.create_dataset('k_x', data=k_x)
        h5f.create_dataset('k_y', data=k_y)
        h5f.create_dataset('ps', data=ps)
        h5f.create_dataset('ft_r', data=ft_r_total)
        h5f.create_dataset('ft_i', data=ft_i_total)
        h5f.attrs['n_p_total'] = n_p_total
        h5f.attrs['l_x'] = l_x
        h5f.attrs['l_y'] = l_y
        h5f.attrs['time'] = time
        h5f.close()

    return




