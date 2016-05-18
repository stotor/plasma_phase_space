# Routines for calculating triangle quantities such as area

import numpy as np
from mpi4py import MPI
import h5py
import sys

import PyPSI as psi

import utilities
import osiris_interface as oi
import ship

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

def extend_lagrangian_quantity_2d(cartcomm, l_quant):
    rank = cartcomm.Get_rank()

    l_quant_extended = np.zeros([l_quant.shape[0]+1,l_quant.shape[1]+1,l_quant.shape[2]], dtype='double')

    l_quant_extended[:-1,:-1,:] = l_quant

    sendtag = 0
    recvtag = 0

    source_dest = cartcomm.Shift(1,-1)
    source = source_dest[0]
    dest = source_dest[1]
    x1_face_send = np.array(l_quant_extended[:-1,0,:], copy=True)
    x1_face_recv = np.zeros_like(x1_face_send)
    cartcomm.Sendrecv(x1_face_send, dest, sendtag, x1_face_recv, source, recvtag)
    l_quant_extended[:-1,-1,:] = x1_face_recv

    source_dest = cartcomm.Shift(0,-1)
    source = source_dest[0]
    dest = source_dest[1]
    x2_face_send = np.array(l_quant_extended[0,:,:], copy=True)
    x2_face_recv = np.zeros_like(x2_face_send)
    cartcomm.Sendrecv(x2_face_send, dest, sendtag, x2_face_recv, source, recvtag)
    l_quant_extended[-1,:,:] = x2_face_recv

    return l_quant_extended

def extend_lagrangian_quantity_3d(cartcomm, l_quant):
    rank = cartcomm.Get_rank()

    l_quant_extended = np.zeros([l_quant.shape[0]+1,l_quant.shape[1]+1,l_quant.shape[2]+1,l_quant.shape[3]], 
                                dtype='double')

    l_quant_extended[:-1,:-1,:-1,:] = l_quant

    sendtag = 0
    recvtag = 0

    source_dest = cartcomm.Shift(2,-1)
    source = source_dest[0]
    dest = source_dest[1]
    x1_face_send = np.array(l_quant_extended[:-1,:-1,0,:], copy=True)
    x1_face_recv = np.zeros_like(x1_face_send)
    cartcomm.Sendrecv(x1_face_send, dest, sendtag, x1_face_recv, source, recvtag)
    l_quant_extended[:-1,:-1,-1,:] = x1_face_recv

    source_dest = cartcomm.Shift(1,-1)
    source = source_dest[0]
    dest = source_dest[1]
    x2_face_send = np.array(l_quant_extended[:-1,0,:,:], copy=True)
    x2_face_recv = np.zeros_like(x2_face_send)
    cartcomm.Sendrecv(x2_face_send, dest, sendtag, x2_face_recv, source, recvtag)
    l_quant_extended[:-1,-1,:,:] = x2_face_recv

    source_dest = cartcomm.Shift(0,-1)
    source = source_dest[0]
    dest = source_dest[1]
    x3_face_send = np.array(l_quant_extended[0,:,:,:], copy=True)
    x3_face_recv = np.zeros_like(x3_face_send)
    cartcomm.Sendrecv(x3_face_send, dest, sendtag, x3_face_recv, source, recvtag)
    l_quant_extended[-1,:,:,:] = x3_face_recv

    return l_quant_extended

def save_triangle_fields_parallel_2d(comm, species, t, raw_folder, output_folder, deposit_n_x, deposit_n_y):
    # Load raw data to be deposited
    input_filename = raw_folder + "/RAW-" + species + "-" + str(t).zfill(6) + ".h5"

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
    n_ppc_x = utilities.int_nth_root(n_ppc, 2)
    n_ppc_y = n_ppc_x

    # Number of particles per processor
    n_ppp = n_ppc * n_cell_proc_x * n_cell_proc_y
    n_ppp_x = utilities.int_nth_root(n_ppp, 2)
    n_ppp_y = n_ppp_x

    axis = np.zeros([2,2], dtype='double')
    
    axis[0,0] = f_input.attrs['XMIN'][0]
    axis[0,1] = f_input.attrs['XMAX'][0]
    axis[1,0] = f_input.attrs['XMIN'][1]
    axis[1,1] = f_input.attrs['XMAX'][1]

    if (rank==0):
        t_start = MPI.Wtime()

    # Get particle data for this processor's lagrangian subdomain
    [particle_positions, particle_momentum] = ship.ship_particle_data(cartcomm, f_input, 2)

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Time for particle data shipping:')
        print(t_elapsed)

    time = f_input.attrs['TIME']

    f_input.close()

    particle_velocities = oi.momentum_to_velocity(particle_momentum)

    # Add ghost column and row
    particle_positions = particle_positions.reshape(n_ppp_y, n_ppp_x, 2)
    particle_velocities = particle_velocities.reshape(n_ppp_y, n_ppp_x, 3)
    
    # Extend to get the missing triangle vertices
    if (rank==0):
        t_start = MPI.Wtime()

    particle_positions_extended = extend_lagrangian_quantity_2d(cartcomm, particle_positions)
    particle_velocities_extended = extend_lagrangian_quantity_2d(cartcomm, particle_velocities)

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Time for adding ghost zones:')
        print(t_elapsed)

    # Deposit using psi
    # Create triangles arrays
    pos = get_triangles_array(particle_positions_extended)
    vel = get_triangles_array(particle_velocities_extended)

    deposit_n_ppc = n_p_total / float(deposit_n_x * deposit_n_y)
    particle_charge = -1.0 / deposit_n_ppc

    ntri = pos.shape[0]
    charge = particle_charge * np.ones(ntri) / 2.0

    # Parameters for PSI
    grid = (deposit_n_x, deposit_n_y)
    window = ((axis[0,0], axis[1,0]), (axis[0,1], axis[1,1]))
    box = window

    if (rank==0):
        t_start = MPI.Wtime()

    fields = {'m': None, 'v': None} 
    psi.elementMesh(fields, np.array(pos, copy=True), np.array(vel[:,:,:2], copy=True), charge, grid=grid, window=window, box=box, periodic=True)
    rho = fields['m']
    j3 = (fields['v'][:,:,0] * fields['m'])
    j2 = (fields['v'][:,:,1] * fields['m'])
    # Repeat for j3
    fields = {'m': None,'v': None} 
    psi.elementMesh(fields, np.array(pos, copy=True), np.array(vel[:,:,1:], copy=True), charge, grid=grid, window=window, box=box, periodic=True)
    j1 = (fields['v'][:,:,1] * fields['m'])

    cartcomm.barrier()
    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Time for deposit:')
        print(t_elapsed)

    # Reduce deposited fields
    rho_total = np.zeros(deposit_n_y * deposit_n_x).reshape(deposit_n_y, deposit_n_x)
    j1_total = np.zeros(deposit_n_y * deposit_n_x).reshape(deposit_n_y, deposit_n_x)
    j2_total = np.zeros(deposit_n_y * deposit_n_x).reshape(deposit_n_y, deposit_n_x)
    j3_total = np.zeros(deposit_n_y * deposit_n_x).reshape(deposit_n_y, deposit_n_x)

    if (rank==0):
        t_start = MPI.Wtime()

    cartcomm.Reduce([rho, MPI.DOUBLE], [rho_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j1, MPI.DOUBLE], [j1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j2, MPI.DOUBLE], [j2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j3, MPI.DOUBLE], [j3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Time for parallel field reduction:')
        print(t_elapsed)

    if (rank==0):
        t_start = MPI.Wtime()

    # Save final field
    if (rank==0):
        utilities.save_density_field_attrs(output_folder, 'charge-ed', species, t, time, rho_total, axis)
        utilities.save_density_field_attrs(output_folder, 'j1-ed', species, t, time, j1_total, axis)
        utilities.save_density_field_attrs(output_folder, 'j2-ed', species, t, time, j2_total, axis)
        utilities.save_density_field_attrs(output_folder, 'j3-ed', species, t, time, j3_total, axis)

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Time for saving field:')
        print(t_elapsed)

    return

def save_triangle_fields_parallel_3d(comm, species, t, raw_folder, output_folder, deposit_n_x, deposit_n_y, deposit_n_z, zoom):
    # Load raw data to be deposited
    input_filename = raw_folder + "/RAW-" + species + "-" + str(t).zfill(6) + ".h5"

    f_input = h5py.File(input_filename, 'r', driver='mpio', comm=comm)

    n_proc_x = f_input.attrs['PAR_NODE_CONF'][0]
    n_proc_y = f_input.attrs['PAR_NODE_CONF'][1]
    n_proc_z = f_input.attrs['PAR_NODE_CONF'][2]

    cartcomm = comm.Create_cart([n_proc_z, n_proc_y, n_proc_x], periods=[True,True,True])

    rank = cartcomm.Get_rank()
    size = cartcomm.Get_size()

    n_cell_x = f_input.attrs['NX'][0]
    n_cell_y = f_input.attrs['NX'][1]
    n_cell_z = f_input.attrs['NX'][2]
    
    n_cell_proc_x = n_cell_x / n_proc_x
    n_cell_proc_y = n_cell_y / n_proc_y
    n_cell_proc_z = n_cell_z / n_proc_z

    n_p_total = f_input['x1'].shape[0]
    n_ppc = n_p_total / (n_cell_x * n_cell_y * n_cell_z)
    n_ppc_x = utilities.int_nth_root(n_ppc, 3)
    n_ppc_y = n_ppc_x
    n_ppc_z = n_ppc_x

    deposit_n_ppc = float(n_p_total) / (deposit_n_x * deposit_n_y * deposit_n_z)

    # Number of particles per processor
    n_ppp = n_ppc * n_cell_proc_x * n_cell_proc_y * n_cell_proc_z
    n_ppp_x = utilities.int_nth_root(n_ppp, 3)
    n_ppp_y = n_ppp_x
    n_ppp_z = n_ppp_x

    
    axis = np.zeros([3,2], dtype='double')
    
    axis[0,0] = f_input.attrs['XMIN'][0]
    axis[0,1] = f_input.attrs['XMAX'][0]
    axis[1,0] = f_input.attrs['XMIN'][1]
    axis[1,1] = f_input.attrs['XMAX'][1]
    axis[2,0] = f_input.attrs['XMIN'][2]
    axis[2,1] = f_input.attrs['XMAX'][2]

    axis_zoom = axis / zoom

    # Get particle data for this processor's lagrangian subdomain
    if (rank==0):
        t_start = MPI.Wtime()
    [particle_positions, particle_momentum] = ship.ship_particle_data(cartcomm, f_input, 3)
    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Time for particle data shipping:')
        print(t_elapsed)

    time = f_input.attrs['TIME']

    f_input.close()

    particle_velocities = oi.momentum_to_velocity(particle_momentum)

    # Add ghost column and row
    particle_positions = particle_positions.reshape(n_ppp_z, n_ppp_y, n_ppp_x, 3)
    particle_velocities = particle_velocities.reshape(n_ppp_z, n_ppp_y, n_ppp_x, 3)
    if (rank==0):
        t_start = MPI.Wtime()
    particle_positions_extended = extend_lagrangian_quantity_3d(cartcomm, particle_positions)
    particle_velocities_extended = extend_lagrangian_quantity_3d(cartcomm, particle_velocities)
    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Time for adding ghost zones:')
        print(t_elapsed)

    # Deposit using psi
    #particle_charge = -1.0 / n_ppp

    # Parameters for PSI
    grid = (deposit_n_z, deposit_n_y, deposit_n_x)
    box = ((axis[2,0], axis[1,0], axis[0,0]), (axis[2,1], axis[1,1], axis[0,1]))
    window = ((axis_zoom[2,0], axis_zoom[1,0], axis_zoom[0,0]), (axis_zoom[2,1], axis_zoom[1,1], axis_zoom[0,1]))

    fields = {'m': None , 'v': None}
    tol = 1000
    if (rank==0):
        t_start = MPI.Wtime()

    for pos, vel, mass, block, nblocks in psi.elementBlocksFromGrid(particle_positions_extended, particle_velocities_extended, order=1, periodic=False):
        mass = mass * (-1.0 * n_ppp) / float(deposit_n_ppc) * zoom**3
        psi.elementMesh(fields, pos, vel, mass, tol=tol, window=window, grid=grid, periodic=True, box=box)

    cartcomm.barrier()
    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Time for deposit:')
        print(t_elapsed)

    rho = fields['m']
    j1 = (fields['v'][:,:,:,2] * fields['m'])
    j2 = (fields['v'][:,:,:,1] * fields['m'])
    j3 = (fields['v'][:,:,:,0] * fields['m'])
    
    if (rank==0):
        t_start = MPI.Wtime()

    # Reduce deposited fields
    rho_total = np.zeros(deposit_n_z * deposit_n_y * deposit_n_x).reshape([deposit_n_z, deposit_n_y, deposit_n_x])
    j1_total = np.zeros_like(rho_total)
    j2_total = np.zeros_like(rho_total)
    j3_total = np.zeros_like(rho_total)

    if (rank==0):
        t_start = MPI.Wtime()

    cartcomm.Reduce([rho, MPI.DOUBLE], [rho_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j1, MPI.DOUBLE], [j1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j2, MPI.DOUBLE], [j2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j3, MPI.DOUBLE], [j3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Time for parallel field reduction:')
        print(t_elapsed)

    # Save final field
    if (rank==0):
        t_start = MPI.Wtime()

    if (rank==0):
        utilities.save_density_field_attrs(output_folder, 'charge-ed', species, t, time, rho_total, axis_zoom)
        utilities.save_density_field_attrs(output_folder, 'j1-ed', species, t, time, j1_total, axis_zoom)
        utilities.save_density_field_attrs(output_folder, 'j2-ed', species, t, time, j2_total, axis_zoom)
        utilities.save_density_field_attrs(output_folder, 'j3-ed', species, t, time, j3_total, axis_zoom)

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Time for saving field:')
        print(t_elapsed)

    return

