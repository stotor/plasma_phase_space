# Routines for calculating triangle quantities such as area

import numpy as np
from mpi4py import MPI
import h5py
import sys

import utilities
import osiris_interface as oi

pypsi_path = '/Users/stotor/Desktop/plasma_phase_space/PyPSI'                                                                       
sys.path.insert(0, pypsi_path)
import PyPSI as psi

def length_in_box(a, b, l_max):
    """Account for periodic boundaries by choosing the shortest possible distance between two points along a given axis."""
    length = b - a
    if (length > (l_max / 2.0)):
        length = length - l_max
    if (length < (-1.0 * l_max / 2.0)):
        length = length + l_max
    return length

def get_triangle_area(vertices, l_x, l_y):
    a = vertices[0]
    b = vertices[1]
    c = vertices[2]
    ab_x = length_in_box(a[0], b[0], l_x)
    ab_y = length_in_box(a[1], b[1], l_y)
    ac_x = length_in_box(a[0], c[0], l_x)
    ac_y = length_in_box(a[1], c[1], l_y)
    area = 0.5 * (ab_x * ac_y - ab_y * ac_x)
    area = abs(area)
    return area

def get_triangle_vertices_ll(lagrangian_quantity_extended, id_x, id_y):
    vertex_a = lagrangian_quantity_extended[id_y, id_x, :]
    vertex_b = lagrangian_quantity_extended[id_y+1, id_x, :]
    vertex_c = lagrangian_quantity_extended[id_y, id_x+1, :]
        
    vertices = np.array([vertex_a, vertex_b, vertex_c])
    
    return vertices

def get_triangle_vertices_ur(lagrangian_quantity_extended, id_x, id_y):
    vertex_a = lagrangian_quantity_extended[id_y+1, id_x, :]
    vertex_b = lagrangian_quantity_extended[id_y, id_x+1, :]
    vertex_c = lagrangian_quantity_extended[id_y+1, id_x+1, :]
        
    vertices = np.array([vertex_a, vertex_b, vertex_c])
    
    return vertices

def get_triangle_vertices_ul(lagrangian_quantity_extended, id_x, id_y):
    vertex_a = lagrangian_quantity_extended[id_y, id_x, :]
    vertex_b = lagrangian_quantity_extended[id_y+1, id_x, :]
    vertex_c = lagrangian_quantity_extended[id_y+1, id_x+1, :]
        
    vertices = np.array([vertex_a, vertex_b, vertex_c])
    
    return vertices

def get_triangle_vertices_lr(lagrangian_quantity_extended, id_x, id_y):
    vertex_a = lagrangian_quantity_extended[id_y, id_x, :]
    vertex_b = lagrangian_quantity_extended[id_y+1, id_x+1, :]
    vertex_c = lagrangian_quantity_extended[id_y, id_x+1, :]
        
    vertices = np.array([vertex_a, vertex_b, vertex_c])
    
    return vertices

def get_triangle_areas(particle_positions, n_l_x, n_l_y, l_x, l_y):
    area = []
    # Loop over rows
    for j in range(n_l_y):
        # Loop over square cells in a row
        for i in range(n_l_x):
            # Get indices of lower left and upper right triangles
            vertices_ll = get_triangle_vertices_ll(particle_positions, i + j * n_l_x, n_l_x, n_l_y)
            vertices_ur = get_triangle_vertices_ur(particle_positions, i + j * n_l_x, n_l_x, n_l_y)
            # Calculate triangle area
            area.append([get_triangle_area(vertices_ll, l_x, l_y)])
            area.append([get_triangle_area(vertices_ur, l_x, l_y)])
    return area

def get_triangles_array(lagrangian_quantity_extended):
    n_l_y = lagrangian_quantity_extended.shape[0] - 1
    n_l_x = lagrangian_quantity_extended.shape[1] - 1
    dim = lagrangian_quantity_extended.shape[2]

    n_particles = n_l_x * n_l_y
    n_triangles = n_particles * 2

    triangles_array = np.zeros([n_triangles, 3, dim])
    for id_y in range(n_l_y):
        for id_x in range(n_l_x):
            i = id_x + id_y * n_l_x
            # Get indices of lower left and upper right triangles
            vertices_ll = get_triangle_vertices_ll(lagrangian_quantity_extended, id_x, id_y)
            vertices_ur = get_triangle_vertices_ur(lagrangian_quantity_extended, id_x, id_y)
            triangles_array[2*i,:,:] = vertices_ll
            triangles_array[2*i+1,:,:] = vertices_ur
    return triangles_array

def get_triangle_areas_array(pos, l_x, l_y):
    n_tri = pos.shape[0]
    areas = np.zeros(n_tri)
    for i in range(n_tri):
        vertices = pos[i,:,:]
        areas[i] = get_triangle_area(vertices, l_x, l_y)
    return areas

def get_max_axis_separation_cell(v_ll, v_lr, v_ul, v_ur, l_x, l_y):
    vertical_x = length_in_box(v_ll[0], v_ul[0], l_x)
    vertical_y = length_in_box(v_ll[1], v_ul[1], l_y)
    horizontal_x = length_in_box(v_ll[0], v_lr[0], l_x)
    horizontal_y = length_in_box(v_ll[1], v_lr[1], l_y)
    diagonal_a_x = length_in_box(v_ll[0], v_ur[0], l_x)
    diagonal_a_y = length_in_box(v_ll[1], v_ur[1], l_y)
    diagonal_b_x = length_in_box(v_ul[0], v_lr[0], l_x)
    diagonal_b_y = length_in_box(v_ul[1], v_lr[1], l_y)

    max_axis_separation = np.amax([vertical_x, vertical_y, 
                                  horizontal_x, horizontal_y, 
                                  diagonal_a_x, diagonal_a_y,
                                  diagonal_b_x, diagonal_b_y])

    return max_axis_separation

def get_max_axis_separation_species(particle_positions, n_l_x, n_l_y, l_x, l_y):
    max_axis_separation = 0.0
    # Loop over rows
    for j in range(n_l_y):
        # Loop over square cells in a row
        for i in range(n_l_x):
            cell = i + j * n_l_x
            index_ll = cell
            index_lr = cell+1
            index_ul = cell+n_l_x
            index_ur = cell+n_l_x+1

            v_ll = particle_positions[index_ll]
            v_lr = particle_positions[index_lr]
            v_ul = particle_positions[index_ul]
            v_ur = particle_positions[index_ur]

            max_axis_separation = max(max_axis_separation, 
                                      get_max_axis_separation_cell(v_ll, v_lr, v_ul, v_ur, l_x, l_y))

    return max_axis_separation

def get_axis_separations_triangle(v_ll, v_lr, v_ul, v_ur, l_x, l_y):
    vertical_x = length_in_box(v_ll[0], v_ul[0], l_x)
    vertical_y = length_in_box(v_ll[1], v_ul[1], l_y)
    horizontal_x = length_in_box(v_ll[0], v_lr[0], l_x)
    horizontal_y = length_in_box(v_ll[1], v_lr[1], l_y)
    diagonal_a_x = length_in_box(v_ll[0], v_ur[0], l_x)
    diagonal_a_y = length_in_box(v_ll[1], v_ur[1], l_y)
    diagonal_b_x = length_in_box(v_ul[0], v_lr[0], l_x)
    diagonal_b_y = length_in_box(v_ul[1], v_lr[1], l_y)

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

    return [separations_x, separations_y]

def get_axis_separation_distribution_species(particle_positions, n_l_x, n_l_y, l_x, l_y):
    separations_x = []
    separations_y = []
    # Loop over rows
    for j in range(n_l_y):
        # Loop over square cells in a row
        for i in range(n_l_x):
            cell = i + j * n_l_x
            index_ll = cell
            index_lr = cell+1
            index_ul = cell+n_l_x
            index_ur = cell+n_l_x+1

            v_ll = particle_positions[index_ll]
            v_lr = particle_positions[index_lr]
            v_ul = particle_positions[index_ul]
            v_ur = particle_positions[index_ur]

            separations_cell = get_axis_separations_triangle(v_ll, v_lr, v_ul, v_ur, l_x, l_y)
            separations_x.extend(separations_cell[0])
            separations_y.extend(separations_cell[1])

    return [separations_x, separations_y]

def extend_lagrangian_quantity(cartcomm, lagrangian_quantity):
    rank = cartcomm.Get_rank()
    dim = lagrangian_quantity.shape[2]

    sendtag = 0
    recvtag = 0

    source_dest = cartcomm.Shift(0,-1)
    source = source_dest[0]
    dest = source_dest[1]
    first_row_send = np.array(lagrangian_quantity[0,:,:], copy=True)
    last_row_recv = np.zeros(first_row_send.shape)
    cartcomm.Sendrecv(first_row_send, dest, sendtag, last_row_recv, source, recvtag)

    source_dest = cartcomm.Shift(1,-1)
    source = source_dest[0]
    dest = source_dest[1]
    first_column_send = np.array(lagrangian_quantity[:,0,:], copy=True)
    last_column_recv = np.zeros(first_column_send.shape)
    cartcomm.Sendrecv(first_column_send, dest, sendtag, last_column_recv, source, recvtag)

    coords = cartcomm.Get_coords(rank)
    upper_right_coords = [(coords[0]+1), (coords[1]+1)]
    lower_left_coords = [(coords[0]-1), (coords[1]-1)]
    source = cartcomm.Get_cart_rank(upper_right_coords)
    dest = cartcomm.Get_cart_rank(lower_left_coords)
    first_cell_send = np.array(lagrangian_quantity[0,0,:], copy=True)
    last_cell_recv = np.zeros(first_cell_send.shape)
    cartcomm.Sendrecv(first_cell_send, dest, sendtag, last_cell_recv, source, recvtag)

    last_row_recv = np.append(last_row_recv, last_cell_recv.reshape(1,dim), axis = 0)
    lagrangian_quantity = np.append(lagrangian_quantity, last_column_recv.reshape(last_column_recv.shape[0],1,dim), axis=1)
    lagrangian_quantity = np.append(lagrangian_quantity, last_row_recv.reshape(1,last_row_recv.shape[0],dim), axis=0)

    lagrangian_quantity_extended = np.array(lagrangian_quantity, copy=True)

    return lagrangian_quantity_extended

def save_triangle_fields_parallel(comm, species, t, simulation_folder, output_folder):
    # Load raw data to be deposited
    input_filename = simulation_folder + "/MS/RAW/" + species + "/RAW-" + species + "-" + str(t).zfill(6) + ".h5"

    f_input = h5py.File(input_filename, 'r', driver='mpio', comm=comm)

    n_proc_x = f_input.attrs['PAR_NODE_CONF'][0]
    n_proc_y = f_input.attrs['PAR_NODE_CONF'][1]
    n_cell_x = f_input.attrs['NX'][0]
    n_cell_y = f_input.attrs['NX'][1]

    cartcomm = comm.Create_cart([n_proc_y, n_proc_x], periods=[True,True])
    rank = cartcomm.Get_rank()
    size = cartcomm.Get_size()
    
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

    f_input.close()

    # Add ghost column and row
    particle_positions = particle_positions.reshape(n_ppp_y, n_ppp_x, 2)
    particle_velocities = particle_velocities.reshape(n_ppp_y, n_ppp_x, 3)
    
    # Extend to get the missing triangle vertices
    particle_positions_extended = extend_lagrangian_quantity(cartcomm, particle_positions)
    particle_velocities_extended = extend_lagrangian_quantity(cartcomm, particle_velocities)

    # Deposit using psi
    # Create triangles arrays
    pos = get_triangles_array(particle_positions_extended)
    vel = get_triangles_array(particle_velocities_extended)

    particle_charge = -1.0 / n_ppc
    ntri = n_ppp*2
    charge = particle_charge * np.ones(ntri) / 2.0

    # Parameters for PSI
    grid = (n_cell_x, n_cell_y)
    window = ((0.0, 0.0), (l_x, l_y))
    box = window

    fields = {'m': None, 'v': None} 
    psi.elementMesh(fields, pos, np.array(vel[:,:,:2], copy=True), charge, grid=grid, window=window, box=box, periodic=True)
    rho = fields['m']
    j1 = (fields['v'][:,:,0] * fields['m'])
    j2 = (fields['v'][:,:,1] * fields['m'])
    # Repeat for j3
    fields = {'m': None,'v': None} 
    psi.elementMesh(fields, pos, np.array(vel[:,:,1:], copy=True), charge, grid=grid, window=window, box=box, periodic=True)
    j3 = (fields['v'][:,:,1] * fields['m'])
    
    # Reduce deposited fields
    rho_total = np.zeros(n_cell_x * n_cell_y).reshape(n_cell_y, n_cell_x)
    j1_total = np.zeros(n_cell_x * n_cell_y).reshape(n_cell_y, n_cell_x)
    j2_total = np.zeros(n_cell_x * n_cell_y).reshape(n_cell_y, n_cell_x)
    j3_total = np.zeros(n_cell_x * n_cell_y).reshape(n_cell_y, n_cell_x)
    cartcomm.Reduce([rho, MPI.DOUBLE], [rho_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j1, MPI.DOUBLE], [j1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j2, MPI.DOUBLE], [j2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j3, MPI.DOUBLE], [j3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    rho_total = np.array(rho_total.transpose(), copy=True) 
    j1_total = np.array(j1_total.transpose(), copy=True) 
    j2_total = np.array(j2_total.transpose(), copy=True) 
    j3_total = np.array(j3_total.transpose(), copy=True) 

    # Save final field
    if (rank==0):
        utilities.save_density_field(output_folder, 'charge-ed', species, t, rho_total)
        utilities.save_density_field(output_folder, 'j1-ed', species, t, j1_total)
        utilities.save_density_field(output_folder, 'j2-ed', species, t, j2_total)
        utilities.save_density_field(output_folder, 'j3-ed', species, t, j3_total)

    return
