# Routines for PIC calculations such as charge and current deposits.

import numpy as np
import math
from mpi4py import MPI
import h5py

import osiris_interface as oi
import utilities

def deposit_ngp_particle(position, field, charge, n_x, n_y, dx):
    """NGP deposit onto a 2D field.  Assumes dx = dy, and that x_min = y_min = 0.0."""
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
    
    field[j_ngp%n_y, i_ngp%n_x] += charge
    
    return field

def deposit_cic_particle(position, field, charge, n_x, n_y, dx):
    """CIC deposit onto a 2D field.  Assumes dx = dy, and that x_min = y_min = 0.0."""
    # Shift to align the cell centers with the pixel centers
    x = position[1] / dx - 0.5
    y = position[0] / dx - 0.5
    
    # Indices of the lower left gridpoint of the cell the particle is in
    i_ll = math.floor(x)
    j_ll = math.floor(y)
    
    x = x - i_ll
    y = y - j_ll
    
    field[j_ll%n_y, i_ll%n_x] += (1.0 - x) * (1.0 - y) * charge
    field[j_ll%n_y, (i_ll+1)%n_x] += x * (1.0 - y) * charge
    field[(j_ll+1)%n_y, (i_ll+1)%n_x] += x * y * charge
    field[(j_ll+1)%n_y, i_ll%n_x] += (1.0 - x) * y * charge
    
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

def deposit_cic_2x_particle(position, field, charge, n_x, n_y, dx):
    """2X CIC deposit onto a 2D field.  Assumes dx = dy, and that x_min = y_min = 0.0."""
    # Shift to align the cell centers with the pixel centers
    x_normalized = position[0] / dx - 0.5
    y_normalized = position[1] / dx - 0.5
    
    # Indices of the lower left gridpoint of the cell the particle is in
    x_ngp = get_ngp(x_normalized)
    y_ngp = get_ngp(y_normalized)
    
    x_rel = x_normalized - x_ngp
    y_rel = y_normalized - y_ngp

    fraction_x = [0.5 * (0.5 - x_rel), 0.5, 0.5 * (0.5 + x_rel)]
    fraction_y = [0.5 * (0.5 - y_rel), 0.5, 0.5 * (0.5 + y_rel)]
    
    field[(y_ngp-1)%n_y, (x_ngp-1)%n_x] += charge * fraction_x[0] * fraction_y[0]
    field[y_ngp%n_y, (x_ngp-1)%n_x] += charge * fraction_x[0] * fraction_y[1]
    field[(y_ngp+1)%n_y, (x_ngp-1)%n_x] += charge * fraction_x[0] * fraction_y[2]

    field[(y_ngp-1)%n_y, x_ngp%n_x] += charge * fraction_x[1] * fraction_y[0]
    field[y_ngp%n_y, x_ngp%n_x] += charge * fraction_x[1] * fraction_y[1]
    field[(y_ngp+1)%n_y, x_ngp%n_x] += charge * fraction_x[1] * fraction_y[2]

    field[(y_ngp-1)%n_y, (x_ngp+1)%n_x] += charge * fraction_x[2] * fraction_y[0]
    field[y_ngp%n_y, (x_ngp+1)%n_x] += charge * fraction_x[2] * fraction_y[1]
    field[(y_ngp+1)%n_y, (x_ngp+1)%n_x] += charge * fraction_x[2] * fraction_y[2]
    
    return field

def deposit_cic_4x_particle(position, field, charge, n_x, n_y, dx):
    """4X CIC deposit onto a 2D field.  Assumes dx = dy, and that x_min = y_min = 0.0."""
    # Shift to align the cell centers with the pixel centers
    x_normalized = position[0] / dx - 0.5
    y_normalized = position[1] / dx - 0.5
    
    # Indices of the lower left gridpoint of the cell the particle is in
    x_ngp = get_ngp(x_normalized)
    y_ngp = get_ngp(y_normalized)
    
    x_rel = x_normalized - x_ngp
    y_rel = y_normalized - y_ngp

    fraction_x = [0.25 * (0.5 - x_rel), 0.25, 0.25, 0.25, 0.25 * (0.5 + x_rel)]
    fraction_y = [0.25 * (0.5 - y_rel), 0.25, 0.25, 0.25, 0.25 * (0.5 + y_rel)]
    
    field[(y_ngp-2)%n_y, (x_ngp-2)%n_x] += charge * fraction_x[0] * fraction_y[0]
    field[(y_ngp-1)%n_y, (x_ngp-2)%n_x] += charge * fraction_x[0] * fraction_y[1]
    field[y_ngp%n_y, (x_ngp-2)%n_x] += charge * fraction_x[0] * fraction_y[2]
    field[(y_ngp+1)%n_y, (x_ngp-2)%n_x] += charge * fraction_x[0] * fraction_y[3]
    field[(y_ngp+2)%n_y, (x_ngp-2)%n_x] += charge * fraction_x[0] * fraction_y[4]

    field[(y_ngp-2)%n_y, (x_ngp-1)%n_x] += charge * fraction_x[1] * fraction_y[0]
    field[(y_ngp-1)%n_y, (x_ngp-1)%n_x] += charge * fraction_x[1] * fraction_y[1]
    field[y_ngp%n_y, (x_ngp-1)%n_x] += charge * fraction_x[1] * fraction_y[2]
    field[(y_ngp+1)%n_y, (x_ngp-1)%n_x] += charge * fraction_x[1] * fraction_y[3]
    field[(y_ngp+2)%n_y, (x_ngp-1)%n_x] += charge * fraction_x[1] * fraction_y[4]

    field[(y_ngp-2)%n_y, x_ngp%n_x] += charge * fraction_x[2] * fraction_y[0]
    field[(y_ngp-1)%n_y, x_ngp%n_x] += charge * fraction_x[2] * fraction_y[1]
    field[y_ngp%n_y, x_ngp%n_x] += charge * fraction_x[2] * fraction_y[2]
    field[(y_ngp+1)%n_y, x_ngp%n_x] += charge * fraction_x[2] * fraction_y[3]
    field[(y_ngp+2)%n_y, x_ngp%n_x] += charge * fraction_x[2] * fraction_y[4]

    field[(y_ngp-2)%n_y, (x_ngp+1)%n_x] += charge * fraction_x[3] * fraction_y[0]
    field[(y_ngp-1)%n_y, (x_ngp+1)%n_x] += charge * fraction_x[3] * fraction_y[1]
    field[y_ngp%n_y, (x_ngp+1)%n_x] += charge * fraction_x[3] * fraction_y[2]
    field[(y_ngp+1)%n_y, (x_ngp+1)%n_x] += charge * fraction_x[3] * fraction_y[3]
    field[(y_ngp+2)%n_y, (x_ngp+1)%n_x] += charge * fraction_x[3] * fraction_y[4]

    field[(y_ngp-2)%n_y, (x_ngp+2)%n_x] += charge * fraction_x[4] * fraction_y[0]
    field[(y_ngp-1)%n_y, (x_ngp+2)%n_x] += charge * fraction_x[4] * fraction_y[1]
    field[y_ngp%n_y, (x_ngp+2)%n_x] += charge * fraction_x[4] * fraction_y[2]
    field[(y_ngp+1)%n_y, (x_ngp+2)%n_x] += charge * fraction_x[4] * fraction_y[3]
    field[(y_ngp+2)%n_y, (x_ngp+2)%n_x] += charge * fraction_x[4] * fraction_y[4]
    
    return field

def deposit_cic_refined_particle(position, field, charge, n_x, n_y, dx, refine):
    """Refined CIC deposit onto a 2D field.  Assumes dx = dy, and that x_min = y_min = 0.0."""
    # Shift to align the cell centers with the pixel centers
    x_normalized = position[0] / dx - 0.5
    y_normalized = position[1] / dx - 0.5
    
    # Indices of the lower left gridpoint of the cell the particle is in
    x_ngp = get_ngp(x_normalized)
    y_ngp = get_ngp(y_normalized)
    
    x_rel = x_normalized - x_ngp
    y_rel = y_normalized - y_ngp

    fraction_x = np.zeros(refine + 1, dtype='double')
    fraction_x[:] = 1.0 / refine
    fraction_x[0] = fraction_x[0] * (0.5 - x_rel)
    fraction_x[-1] = fraction_x[-1] * (0.5 + x_rel)

    fraction_y = np.zeros(refine + 1, dtype='double')
    fraction_y[:] = 1.0 / refine
    fraction_y[0] = fraction_y[0] * (0.5 - y_rel)
    fraction_y[-1] = fraction_y[-1] * (0.5 + y_rel)

    charge_fraction = np.outer(fraction_y, fraction_x) * charge

    for y in range(refine+1):
        for x in range(refine+1):
            field[(y_ngp-refine/2+y)%n_y, (x_ngp-refine/2+x)%n_x] += charge_fraction[y,x]
    
    return field


def deposit_species(particle_positions, field, particle_charges, n_x, n_y, dx, type):
    n_p = len(particle_positions)
    if (type=='ngp'):
        for i in range(n_p):
            position = particle_positions[i,:]
            charge = particle_charges[i]
            deposit_ngp_particle(position, field, charge, n_x, n_y, dx)
    elif (type=='cic'):
        for i in range(n_p):
            position = particle_positions[i,:]
            charge = particle_charges[i]
            deposit_cic_particle(position, field, charge, n_x, n_y, dx)
    return field

def save_pic_fields_parallel(comm, species, t, raw_folder, output_folder, deposit_n_x, deposit_n_y):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load raw data to be deposited
    input_filename = raw_folder + "/RAW-" + species + "-" + str(t).zfill(6) + ".h5"

    f_input = h5py.File(input_filename, 'r', driver='mpio', comm=comm)

    n_proc_x = f_input.attrs['PAR_NODE_CONF'][0]
    n_proc_y = f_input.attrs['PAR_NODE_CONF'][1]
    n_cell_x = f_input.attrs['NX'][0]
    n_cell_y = f_input.attrs['NX'][1]
    
    n_cell_proc_x = n_cell_x / n_proc_x
    n_cell_proc_y = n_cell_y / n_proc_y

    n_p_total = f_input['x1'].shape[0]
    n_ppc = n_p_total / (n_cell_x * n_cell_y)
    n_ppc_x = int(np.sqrt(n_ppc))
    n_ppc_y = n_ppc_x

    # Number of particles per processor
    n_ppp = (n_ppc_x * n_ppc_y) * (n_cell_proc_x * n_cell_proc_y)
    n_ppp_x = int(np.sqrt(n_ppp))
    n_ppp_y = n_ppp_x
    
    l_x = f_input.attrs['XMAX'][0] - f_input.attrs['XMIN'][0]
    l_y = f_input.attrs['XMAX'][1] - f_input.attrs['XMIN'][1]
    dx = l_x / float(n_cell_x)

    i_start = rank * n_ppp
    i_end = (rank + 1) * n_ppp

    # Get position array
    # Get velocity array
    particle_positions = oi.create_position_array(f_input, i_start, i_end)
    particle_velocities = oi.create_velocity_array(f_input, i_start, i_end)
    particle_momentum = oi.create_momentum_array(f_input, i_start, i_end)

    axis = np.zeros([2,2], dtype='double')
    
    axis[0,0] = f_input.attrs['XMIN'][0]
    axis[0,1] = f_input.attrs['XMAX'][0]
    axis[1,0] = f_input.attrs['XMIN'][1]
    axis[1,1] = f_input.attrs['XMAX'][1]
    time = f_input.attrs['TIME']

    f_input.close()
    
    # Deposit species
    deposit_dx = l_x / float(deposit_n_x)
    deposit_n_ppc = n_p_total / float(deposit_n_x * deposit_n_y)
    particle_charge = -1.0 / deposit_n_ppc
    particle_charges = particle_charge * np.ones(n_ppp)

    for deposit_type in ['ngp', 'cic']:

        rho = np.zeros(deposit_n_x * deposit_n_y).reshape(deposit_n_y, deposit_n_x)
        rho = deposit_species(particle_positions, rho, particle_charges, deposit_n_x, deposit_n_y, deposit_dx, deposit_type)

        j1 = np.zeros_like(rho)
        particles_j1 = particle_charges * particle_velocities[:,0]
        j1 = deposit_species(particle_positions, j1, particles_j1, deposit_n_x, deposit_n_y, deposit_dx, deposit_type)

        j2 = np.zeros_like(rho)
        particles_j2 = particle_charges * particle_velocities[:,1]
        j2 = deposit_species(particle_positions, j2, particles_j2,  deposit_n_x, deposit_n_y, deposit_dx, deposit_type)

        j3 = np.zeros_like(rho)
        particles_j3 = particle_charges * particle_velocities[:,2]
        j3 = deposit_species(particle_positions, j3, particles_j3, deposit_n_x, deposit_n_y, deposit_dx, deposit_type)

        ufl1 = np.zeros_like(rho)
        particles_ufl1 = particle_charges * particle_momentum[:,0]
        ufl1 = deposit_species(particle_positions, ufl1, particles_ufl1, deposit_n_x, deposit_n_y, deposit_dx, deposit_type)
        
        ufl2 = np.zeros_like(rho)
        particles_ufl2 = particle_charges * particle_momentum[:,1]
        ufl2 = deposit_species(particle_positions, ufl2, particles_ufl2, deposit_n_x, deposit_n_y, deposit_dx, deposit_type)

        ufl3 = np.zeros_like(rho)
        particles_ufl3 = particle_charges * particle_momentum[:,2]
        ufl3 = deposit_species(particle_positions, ufl3, particles_ufl3, deposit_n_x, deposit_n_y, deposit_dx, deposit_type)
    
        # Reduce deposited fields
        rho_total = np.zeros_like(rho)
        j1_total = np.zeros_like(rho)
        j2_total = np.zeros_like(rho)
        j3_total = np.zeros_like(rho)
        ufl1_total = np.zeros_like(rho)
        ufl2_total = np.zeros_like(rho)
        ufl3_total = np.zeros_like(rho)
        comm.Reduce([rho, MPI.DOUBLE], [rho_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
        comm.Reduce([j1, MPI.DOUBLE], [j1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
        comm.Reduce([j2, MPI.DOUBLE], [j2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
        comm.Reduce([j3, MPI.DOUBLE], [j3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
        comm.Reduce([ufl1, MPI.DOUBLE], [ufl1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
        comm.Reduce([ufl2, MPI.DOUBLE], [ufl2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
        comm.Reduce([ufl3, MPI.DOUBLE], [ufl3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

        # Save final field
        if (rank==0):
            utilities.save_density_field_attrs(output_folder, 'charge-' + deposit_type, species, t, time, rho_total, axis)
            utilities.save_density_field_attrs(output_folder, 'j1-' + deposit_type, species, t, time, j1_total, axis)
            utilities.save_density_field_attrs(output_folder, 'j2-' + deposit_type, species, t, time, j2_total, axis)
            utilities.save_density_field_attrs(output_folder, 'j3-' + deposit_type, species, t, time, j3_total, axis)
        
            zero_indices = np.where(rho_total==0.0)
            rho_total[zero_indices] = 10.0**16
            ufl1_total = ufl1_total / rho_total
            ufl2_total = ufl2_total / rho_total
            ufl3_total = ufl3_total / rho_total
            ufl1_total[zero_indices] = 0.0
            ufl2_total[zero_indices] = 0.0
            ufl3_total[zero_indices] = 0.0
            utilities.save_density_field_attrs(output_folder, 'ufl1-' + deposit_type, species, t, time, ufl1_total, axis)
            utilities.save_density_field_attrs(output_folder, 'ufl2-' + deposit_type, species, t, time, ufl2_total, axis)
            utilities.save_density_field_attrs(output_folder, 'ufl3-' + deposit_type, species, t, time, ufl3_total, axis)

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




