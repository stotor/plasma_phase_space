# Routines for calculating triangle quantities such as area

import numpy as np
from mpi4py import MPI
import h5py
import sys

import PyPSI as psi

import utilities
import osiris_interface as oi
import ship
import extend

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

    particle_positions_extended = extend.extend_lagrangian_quantity_2d(cartcomm, particle_positions)
    particle_velocities_extended = extend.extend_lagrangian_quantity_2d(cartcomm, particle_velocities)

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
    particle_positions_extended = extend.extend_lagrangian_quantity_3d(cartcomm, particle_positions)
    particle_velocities_extended = extend.extend_lagrangian_quantity_3d(cartcomm, particle_velocities)
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

def distribution_function_2d(comm, species, t, raw_folder, output_folder, sample_locations):
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

    # Add ghost column and row
    particle_positions = particle_positions.reshape(n_ppp_y, n_ppp_x, 2)
    particle_momentum = particle_momentum.reshape(n_ppp_y, n_ppp_x, 3)
    
    # Extend to get the missing triangle vertices
    if (rank==0):
        t_start = MPI.Wtime()

    particle_positions_extended = extend.extend_lagrangian_quantity_2d(cartcomm, particle_positions)
    particle_momentum_extended = extend.extend_lagrangian_quantity_2d(cartcomm, particle_momentum)

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Time for adding ghost zones:')
        print(t_elapsed)

    # Create triangles arrays
    pos = get_triangles_array(particle_positions_extended)
    momentum = get_triangles_array(particle_momentum_extended)

    # Open output file
    if (rank==0):
        utilities.ensure_folder_exists(output_folder)
        filename = output_folder + 'distribution_functions-' + species + '-' + str(t).zfill(6) + '.h5'
        h5f = h5py.File(filename, 'w')

    # Loop over sample locations and calculate distribution function
    for i in range(len(sample_locations)):
        sample_x = sample_locations[i][1]
        sample_y = sample_locations[i][0]
        # For each triangle:
        # Account for periodic boundaries (can be neglected if I don't choose points near the boundary)
        # Calculate area of triangle    
        det = (pos[:,1,0]-pos[:,2,0])*(pos[:,0,1]-pos[:,2,1])+(pos[:,2,1]-pos[:,1,1])*(pos[:,0,0]-pos[:,2,0])
        # Calculate barycentric coordinates
        l1 = ((pos[:,1,0]-pos[:,2,0])*(sample_x-pos[:,2,1])+(pos[:,2,1]-pos[:,1,1])*(sample_y-pos[:,2,0])) / det
        l2 = ((pos[:,2,0]-pos[:,0,0])*(sample_x-pos[:,2,1])+(pos[:,0,1]-pos[:,2,1])*(sample_y-pos[:,2,0])) / det
        l3 = 1.0 - l1 - l2
        # Find triangles that contain the sample location, including edges and vertices
        sample_inside = (l1<=1.0)*(l1>=0.0)*(l2<=1.0)*(l2>=0.0)*(l3<=1.0)*(l3>=0.0)
        sample_indices = np.where(sample_inside==True)[0]
    
        l1_sample = l1[sample_indices]
        l2_sample = l2[sample_indices]
        l3_sample = l3[sample_indices]
        momentum_sample = momentum[sample_indices]

        # Interpolate three momentum components to sample location
        px_sample = l1_sample * momentum_sample[:,0,0] + l2_sample * momentum_sample[:,1,0] + l3_sample * momentum_sample[:,2,0]
        py_sample = l1_sample * momentum_sample[:,0,1] + l2_sample * momentum_sample[:,1,1] + l3_sample * momentum_sample[:,2,1]
        pz_sample = l1_sample * momentum_sample[:,0,2] + l2_sample * momentum_sample[:,1,2] + l3_sample * momentum_sample[:,2,2]
        area_sample = np.abs(det[sample_indices])

        # Send this processors number of streams to root
        n_streams = np.zeros(size, dtype='int32')
        n_streams[rank] = len(sample_indices)

        n_streams_total = np.zeros_like(n_streams)
        cartcomm.Reduce([n_streams, MPI.INT], [n_streams_total, MPI.INT], op = MPI.SUM, root = 0)
        
        # Scatter start indices and broadcast total number of entries
        total_streams = np.zeros(1, dtype='int32')
        start_indices = np.zeros(size, dtype='int32')
        if (rank==0):
            end_indices = np.cumsum(n_streams_total, dtype='int32')
            total_streams = end_indices[-1]
            start_indices = np.roll(end_indices, 1)
            start_indices[0] = 0

        cartcomm.Bcast([total_streams, MPI.INT])
        start_index = np.empty(1, dtype='int32')
        cartcomm.Scatter([start_indices, MPI.INT], [start_index, MPI.INT])

        end_index = start_index + len(sample_indices)
    
        # Reduce distributions
        px_dist = np.zeros(total_streams, dtype='float64')
        py_dist = np.zeros(total_streams, dtype='float64')
        pz_dist = np.zeros(total_streams, dtype='float64')
        area_dist = np.zeros(total_streams, dtype='float64')
        px_dist[start_index:end_index] = px_sample
        py_dist[start_index:end_index] = py_sample
        pz_dist[start_index:end_index] = pz_sample
        area_dist[start_index:end_index] = area_sample

        px_dist_total = np.zeros_like(px_dist)
        py_dist_total = np.zeros_like(py_dist)
        pz_dist_total = np.zeros_like(pz_dist)
        area_dist_total = np.zeros_like(area_dist)

        cartcomm.Reduce([px_dist, MPI.DOUBLE], [px_dist_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
        cartcomm.Reduce([py_dist, MPI.DOUBLE], [py_dist_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
        cartcomm.Reduce([pz_dist, MPI.DOUBLE], [pz_dist_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
        cartcomm.Reduce([area_dist, MPI.DOUBLE], [area_dist_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

        # Save distribution functions
        if (rank==0):
            h5f.create_dataset('sample_' + str(i) + '/px', data=px_dist_total)
            h5f.create_dataset('sample_' + str(i) + '/py', data=py_dist_total)
            h5f.create_dataset('sample_' + str(i) + '/pz', data=pz_dist_total)
            h5f.create_dataset('sample_' + str(i) + '/area', data=area_dist_total)
            h5f['sample_' + str(i)].attrs['x'] = sample_x
            h5f['sample_' + str(i)].attrs['y'] = sample_y
            h5f['sample_' + str(i)].attrs['time'] = time

    if (rank==0):
            h5f.close()

    return
