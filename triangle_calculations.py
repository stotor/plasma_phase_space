"""Routines for calculating triangle quantities such as area """

import numpy as np
from mpi4py import MPI
import h5py
import sys

import PyPSI as psi

import pic_calculations as pic
import utilities
import osiris_interface as oi
import ship
import extend

def calculate_triangle_areas(pos, l_x, l_y):
    # Shift triangle vertices to account for periodic boundaries
    max_x = np.amax(pos[:,:,1], axis=1)
    max_y = np.amax(pos[:,:,0], axis=1)
    
    shift_x = (max_x[:,None] - pos[:,:,1]) > (l_x/2.0)
    shift_y = (max_y[:,None] - pos[:,:,0]) > (l_y/2.0)

    pos_shifted = np.zeros_like(pos)
    
    pos_shifted[:,:,0] = pos[:,:,0] + shift_y * l_y
    pos_shifted[:,:,1] = pos[:,:,1] + shift_x * l_x
    
    det = (pos_shifted[:,1,0]-pos_shifted[:,2,0])*(pos_shifted[:,0,1]-pos_shifted[:,2,1])+(pos_shifted[:,2,1]-pos_shifted[:,1,1])*(pos_shifted[:,0,0]-pos_shifted[:,2,0])
    area = 0.5 * np.abs(det)
    
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

def deposit_triangles_current(cartcomm, pos, vel, charge, grid, window, box, output_folder, species, t, time, axis):
    rank = cartcomm.Get_rank()

    fields = {'m': None, 'v': None} 
    psi.elementMesh(fields, pos, np.array(vel[:,:,:2], copy=True), charge, grid=grid, window=window, box=box, periodic=True)
    rho = fields['m']
    j3 = (fields['v'][:,:,0] * fields['m'])
    j2 = (fields['v'][:,:,1] * fields['m'])
    # Repeat for j1
    fields = {'m': None,'v': None} 
    psi.elementMesh(fields, pos, np.array(vel[:,:,1:], copy=True), charge, grid=grid, window=window, box=box, periodic=True)
    j1 = (fields['v'][:,:,1] * fields['m'])
    
    # Calcualate cell averaged number of streams
    fields = {'m': None}
    psi.elementMesh(fields, pos, np.array(vel[:,:,1:], copy=True), charge, grid=grid, window=window, box=box, periodic=True,
                    weight='volume')
    streams = fields['m']

    # Reduce deposited fields
    rho_total = np.zeros_like(rho)
    j1_total = np.zeros_like(rho)
    j2_total = np.zeros_like(rho)
    j3_total = np.zeros_like(rho)
    streams_total = np.zeros_like(rho)

    cartcomm.Reduce([rho, MPI.DOUBLE], [rho_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j1, MPI.DOUBLE], [j1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j2, MPI.DOUBLE], [j2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j3, MPI.DOUBLE], [j3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([streams, MPI.DOUBLE], [streams_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank==0):
        utilities.save_density_field_attrs(output_folder, 'charge-ed' , species, t, time, rho_total, axis)
        utilities.save_density_field_attrs(output_folder, 'j1-ed', species, t, time, j1_total, axis)
        utilities.save_density_field_attrs(output_folder, 'j2-ed', species, t, time, j2_total, axis)
        utilities.save_density_field_attrs(output_folder, 'j3-ed', species, t, time, j3_total, axis)

        deposit_dx = (axis[0,1] - axis[0,0]) / float(grid[0])
        streams_total = streams_total / deposit_dx**2
        utilities.save_density_field_attrs(output_folder, 'streams-ed', species, t, time, streams_total, axis)
    return

def deposit_sic_3d(cartcomm, pos, vel, charge, grid, window, box, output_folder, species, t, time, axis):
    rank = cartcomm.Get_rank()

    fields = {'m': None, 'v': None} 
    psi.elementMesh(fields, pos, np.array(vel[:,:,:2], copy=True), charge, grid=grid, window=window, box=box, periodic=True)
    rho = fields['m']
    j3 = (fields['v'][:,:,0] * fields['m'])
    j2 = (fields['v'][:,:,1] * fields['m'])
    # Repeat for j1
    fields = {'m': None,'v': None} 
    psi.elementMesh(fields, pos, np.array(vel[:,:,1:], copy=True), charge, grid=grid, window=window, box=box, periodic=True)
    j1 = (fields['v'][:,:,1] * fields['m'])
    
    # Calcualate cell averaged number of streams
    fields = {'m': None}
    psi.elementMesh(fields, pos, np.array(vel[:,:,1:], copy=True), charge, grid=grid, window=window, box=box, periodic=True,
                    weight='volume')
    streams = fields['m']

    # Reduce deposited fields
    rho_total = np.zeros_like(rho)
    j1_total = np.zeros_like(rho)
    j2_total = np.zeros_like(rho)
    j3_total = np.zeros_like(rho)
    streams_total = np.zeros_like(rho)

    cartcomm.Reduce([rho, MPI.DOUBLE], [rho_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j1, MPI.DOUBLE], [j1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j2, MPI.DOUBLE], [j2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j3, MPI.DOUBLE], [j3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([streams, MPI.DOUBLE], [streams_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank==0):
        utilities.save_density_field_attrs(output_folder, 'charge-ed' , species, t, time, rho_total, axis)
        utilities.save_density_field_attrs(output_folder, 'j1-ed', species, t, time, j1_total, axis)
        utilities.save_density_field_attrs(output_folder, 'j2-ed', species, t, time, j2_total, axis)
        utilities.save_density_field_attrs(output_folder, 'j3-ed', species, t, time, j3_total, axis)

        deposit_dx = (axis[0,1] - axis[0,0]) / float(grid[0])
        streams_total = streams_total / deposit_dx**2
        utilities.save_density_field_attrs(output_folder, 'streams-ed', species, t, time, streams_total, axis)
    return


def deposit_triangles_current_points(cartcomm, pos, vel, charge, area, grid, window, box, output_folder, species, t, time, axis):
    rank = cartcomm.Get_rank()

    vel_dummy = np.zeros_like(vel[:,:,:2])

    fields = {'m': None} 
    psi.elementMesh(fields, pos, vel_dummy, charge, grid=grid, window=window, box=box, periodic=True, sampling='point')
    rho = fields['m']

    dx = (axis[0,1] - axis[0,0]) / float(grid[0])
    weight = np.abs(area) / (dx*dx)

    fields = {'m': None} 
    psi.elementMesh(fields, pos, vel_dummy, weight, grid=grid, window=window, box=box, periodic=True, sampling='point')
    streams = fields['m']
    
    # Reduce deposited fields
    rho_total = np.zeros_like(rho)
    streams_total = np.zeros_like(streams)

    cartcomm.Reduce([rho, MPI.DOUBLE], [rho_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([streams, MPI.DOUBLE], [streams_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank==0):
        utilities.save_density_field_attrs(output_folder, 'charge-ed-p' , species, t, time, rho_total, axis)
        utilities.save_density_field_attrs(output_folder, 'streams-ed-p' , species, t, time, streams_total, axis)
    return


def deposit_triangles_momentum(cartcomm, pos, mom, charge, grid, window, box, output_folder, species, t, time, axis):
    rank = cartcomm.Get_rank()

    # Momentum fields
    fields = {'m': None, 'v': None}
    psi.elementMesh(fields, pos, np.array(mom[:,:,:2], copy=True), charge, grid=grid, window=window, box=box, periodic=True)
    rho = fields['m']
    ufl3 = np.array(fields['v'][:,:,0] * fields['m'])
    ufl2 = np.array(fields['v'][:,:,1] * fields['m'])
    # Repeat for ufl1
    fields = {'m': None, 'v': None}
    psi.elementMesh(fields, pos, np.array(mom[:,:,1:], copy=True), charge, grid=grid, window=window, box=box, periodic=True)
    ufl1 = np.array(fields['v'][:,:,1] * fields['m'])

    rho_total = np.zeros_like(rho)
    ufl1_total = np.zeros_like(rho)
    ufl2_total = np.zeros_like(rho)
    ufl3_total = np.zeros_like(rho)
    cartcomm.Reduce([rho, MPI.DOUBLE], [rho_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl1, MPI.DOUBLE], [ufl1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl2, MPI.DOUBLE], [ufl2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl3, MPI.DOUBLE], [ufl3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank==0):        
        ufl1_total = ufl1_total / rho_total
        ufl2_total = ufl2_total / rho_total
        ufl3_total = ufl3_total / rho_total
        utilities.save_density_field_attrs(output_folder, 'ufl1-ed', species, t, time, ufl1_total, axis)
        utilities.save_density_field_attrs(output_folder, 'ufl2-ed', species, t, time, ufl2_total, axis)
        utilities.save_density_field_attrs(output_folder, 'ufl3-ed', species, t, time, ufl3_total, axis)

    return

def deposit_triangles_p2p1(cartcomm, mom, charge, grid, window, box, output_folder, species, t, time, axis):
    if t == 0:
        return
    
    rank = cartcomm.Get_rank()

    mom = np.array(mom[:,:,1:], copy=True)

    vel_dummy = np.zeros_like(mom)
    
    fields = {'m': None}
    psi.elementMesh(fields, mom, vel_dummy, charge, grid=grid, window=window, box=box, periodic=False)
    p2p1 = fields['m']
    p2p1_total = np.zeros_like(p2p1)

    cartcomm.Reduce([p2p1, MPI.DOUBLE], [p2p1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank==0):        
        utilities.save_density_field_attrs(output_folder, 'p2p1-ed', species, t, time, p2p1_total, axis)

    return

def deposit_particles_p2p1_ctypes(cartcomm, mom, charge, grid, window, box, output_folder, species, t, time, axis, deposit):
    # This assumes periodic boundaries so, the deposit box must be large enough
    rank = cartcomm.Get_rank()

    dx = (axis[0,1] - axis[0,0]) / float(grid[0])
    n_y = grid[0]
    n_x = grid[1]
    mom = np.array(mom[:,1:], copy=True)

    mom[:, 0] = mom[:, 0] - axis[0, 0]
    mom[:, 1] = mom[:, 1] - axis[1, 0]

    p2p1 = np.zeros(grid[0] * grid[1], dtype='float64').reshape(grid[0], grid[1])
    
    pic.deposit_species_ctypes(mom, p2p1, charge, n_x, n_y, dx, deposit)
    
    p2p1_total = np.zeros_like(p2p1)
    cartcomm.Reduce([p2p1, MPI.DOUBLE], [p2p1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank==0):        
        utilities.save_density_field_attrs(output_folder, 'p2p1-' + deposit, species, t, time, p2p1_total, axis)

    return


def deposit_triangles_dispersion(cartcomm, pos, mom, charge, grid, window, box, output_folder, species, t, time, axis):
    rank = cartcomm.Get_rank()

    # Momentum fields
    fields = {'m': None, 'v': None, 'vv': None}
    psi.elementMesh(fields, pos, np.array(mom[:,:,:2], copy=True), charge, grid=grid, window=window, box=box, periodic=True)
    rho = fields['m']
    uth3 = np.array((fields['vv'][:,:,0,0] + fields['v'][:,:,0]**2) * fields['m'])
    uth2 = np.array((fields['vv'][:,:,1,1] + fields['v'][:,:,1]**2) * fields['m'])
    ufl3 = np.array(fields['v'][:,:,0] * fields['m'])
    ufl2 = np.array(fields['v'][:,:,1] * fields['m'])
    # Repeat for ufl1
    fields = {'m': None, 'v': None, 'vv': None}
    psi.elementMesh(fields, pos, np.array(mom[:,:,1:], copy=True), charge, grid=grid, window=window, box=box, periodic=True)
    uth1 = np.array((fields['vv'][:,:,1,1] + fields['v'][:,:,1]**2) * fields['m'])
    ufl1 = np.array(fields['v'][:,:,1] * fields['m'])

    rho_total = np.zeros_like(rho)
    ufl1_total = np.zeros_like(rho)
    ufl2_total = np.zeros_like(rho)
    ufl3_total = np.zeros_like(rho)
    uth1_total = np.zeros_like(rho)
    uth2_total = np.zeros_like(rho)
    uth3_total = np.zeros_like(rho)    
    cartcomm.Reduce([rho, MPI.DOUBLE], [rho_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl1, MPI.DOUBLE], [ufl1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl2, MPI.DOUBLE], [ufl2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl3, MPI.DOUBLE], [ufl3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([uth1, MPI.DOUBLE], [uth1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([uth2, MPI.DOUBLE], [uth2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([uth3, MPI.DOUBLE], [uth3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank==0):        
        ufl1_total = ufl1_total / rho_total
        ufl2_total = ufl2_total / rho_total
        ufl3_total = ufl3_total / rho_total
        utilities.save_density_field_attrs(output_folder, 'ufl1-ed', species, t, time, ufl1_total, axis)
        utilities.save_density_field_attrs(output_folder, 'ufl2-ed', species, t, time, ufl2_total, axis)
        utilities.save_density_field_attrs(output_folder, 'ufl3-ed', species, t, time, ufl3_total, axis)

        # Eliminate negative numbers due to machine precision effects
        uth1_total = uth1_total / rho_total - ufl1_total**2
        uth2_total = uth2_total / rho_total - ufl2_total**2
        uth3_total = uth3_total / rho_total - ufl3_total**2
            
        uth1_total[np.where(uth1_total<0.0)] = 0.0
        uth2_total[np.where(uth2_total<0.0)] = 0.0
        uth3_total[np.where(uth3_total<0.0)] = 0.0
            
        uth1_total = np.sqrt(uth1_total)
        uth2_total = np.sqrt(uth2_total)
        uth3_total = np.sqrt(uth3_total)

        utilities.save_density_field_attrs(output_folder, 'uth1-ed', species, t, time, uth1_total, axis)
        utilities.save_density_field_attrs(output_folder, 'uth2-ed', species, t, time, uth2_total, axis)
        utilities.save_density_field_attrs(output_folder, 'uth3-ed', species, t, time, uth3_total, axis)

    return

def deposit_triangles_momentum_lagrangian(cartcomm, pos, mom, charge, grid, window, box, output_folder, species, t, time, axis):
    rank = cartcomm.Get_rank()

    # Momentum fields
    fields = {'m': None, 'v': None}
    psi.elementMesh(fields, pos, np.array(mom[:,:,:2], copy=True), charge, grid=grid, window=window, box=box, periodic=True)
    rho = fields['m']
    ufl3 = np.array(fields['v'][:,:,0] * fields['m'])
    ufl2 = np.array(fields['v'][:,:,1] * fields['m'])
    # Repeat for ufl1
    fields = {'m': None, 'v': None}
    psi.elementMesh(fields, pos, np.array(mom[:,:,1:], copy=True), charge, grid=grid, window=window, box=box, periodic=True)
    ufl1 = np.array(fields['v'][:,:,1] * fields['m'])

    # Reduce deposited fields
    rho_total = np.zeros_like(rho) 
    ufl1_total = np.zeros_like(rho)
    ufl2_total = np.zeros_like(rho)
    ufl3_total = np.zeros_like(rho)

    cartcomm.Reduce([rho, MPI.DOUBLE], [rho_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl1, MPI.DOUBLE], [ufl1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl2, MPI.DOUBLE], [ufl2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl3, MPI.DOUBLE], [ufl3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

    if (rank==0):        
        ufl1_total = ufl1_total / rho_total
        ufl2_total = ufl2_total / rho_total
        ufl3_total = ufl3_total / rho_total
        utilities.save_density_field_attrs(output_folder, 'ufl1-ed-l', species, t, time, ufl1_total, axis)
        utilities.save_density_field_attrs(output_folder, 'ufl2-ed-l', species, t, time, ufl2_total, axis)
        utilities.save_density_field_attrs(output_folder, 'ufl3-ed-l', species, t, time, ufl3_total, axis)

    return

def deposit_particles(cartcomm, pos, mom, vel, charge, grid, window, box,
                      output_folder, species, t, time, axis, deposit_type):
    rank = cartcomm.Get_rank()

    dx = (axis[0,1] - axis[0,0]) / float(grid[0])
    n_y = grid[0]
    n_x = grid[1]

    rho = np.zeros(grid[0] * grid[1], dtype='float64').reshape(grid[0], grid[1])
    j1 = np.zeros_like(rho)
    j2 = np.zeros_like(rho)
    j3 = np.zeros_like(rho)
    ufl1 = np.zeros_like(rho)
    ufl2 = np.zeros_like(rho)
    ufl3 = np.zeros_like(rho)

    # Current fields
    pic.deposit_species(pos, rho, charge, n_x, n_y, dx, deposit_type)

    pic.deposit_species(pos, j1, charge * vel[:, 2], n_x, n_y, dx, deposit_type)
    pic.deposit_species(pos, j2, charge * vel[:, 1], n_x, n_y, dx, deposit_type)
    pic.deposit_species(pos, j3, charge * vel[:, 0], n_x, n_y, dx, deposit_type)

    pic.deposit_species(pos, ufl1, charge * mom[:, 2], n_x, n_y, dx, deposit_type)
    pic.deposit_species(pos, ufl2, charge * mom[:, 1], n_x, n_y, dx, deposit_type)
    pic.deposit_species(pos, ufl3, charge * mom[:, 0], n_x, n_y, dx, deposit_type)
    
    # Reduce deposited fields
    rho_total = np.zeros_like(rho)
    j1_total = np.zeros_like(rho)
    j2_total = np.zeros_like(rho)
    j3_total = np.zeros_like(rho)
    ufl1_total = np.zeros_like(rho)
    ufl2_total = np.zeros_like(rho)
    ufl3_total = np.zeros_like(rho)

    cartcomm.Reduce([rho, MPI.DOUBLE], [rho_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j1, MPI.DOUBLE], [j1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j2, MPI.DOUBLE], [j2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j3, MPI.DOUBLE], [j3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl1, MPI.DOUBLE], [ufl1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl2, MPI.DOUBLE], [ufl2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl3, MPI.DOUBLE], [ufl3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

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

def deposit_particles_ctypes(cartcomm, pos, mom, vel, charge, grid, window, box,
                             output_folder, species, t, time, axis, deposit_type):
    rank = cartcomm.Get_rank()

    dx = (axis[0,1] - axis[0,0]) / float(grid[0])
    n_y = grid[0]
    n_x = grid[1]

    rho = np.zeros(grid[0] * grid[1], dtype='float64').reshape(grid[0], grid[1])
    j1 = np.zeros_like(rho)
    j2 = np.zeros_like(rho)
    j3 = np.zeros_like(rho)
    ufl1 = np.zeros_like(rho)
    ufl2 = np.zeros_like(rho)
    ufl3 = np.zeros_like(rho)

    # Current fields
    pic.deposit_species_ctypes(pos, rho, charge, n_x, n_y, dx, deposit_type)

    pic.deposit_species_ctypes(pos, j1, charge * vel[:, 2], n_x, n_y, dx, deposit_type)
    pic.deposit_species_ctypes(pos, j2, charge * vel[:, 1], n_x, n_y, dx, deposit_type)
    pic.deposit_species_ctypes(pos, j3, charge * vel[:, 0], n_x, n_y, dx, deposit_type)

    pic.deposit_species_ctypes(pos, ufl1, charge * mom[:, 2], n_x, n_y, dx, deposit_type)
    pic.deposit_species_ctypes(pos, ufl2, charge * mom[:, 1], n_x, n_y, dx, deposit_type)
    pic.deposit_species_ctypes(pos, ufl3, charge * mom[:, 0], n_x, n_y, dx, deposit_type)
    
    # Reduce deposited fields
    rho_total = np.zeros_like(rho)
    j1_total = np.zeros_like(rho)
    j2_total = np.zeros_like(rho)
    j3_total = np.zeros_like(rho)
    ufl1_total = np.zeros_like(rho)
    ufl2_total = np.zeros_like(rho)
    ufl3_total = np.zeros_like(rho)

    cartcomm.Reduce([rho, MPI.DOUBLE], [rho_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j1, MPI.DOUBLE], [j1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j2, MPI.DOUBLE], [j2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([j3, MPI.DOUBLE], [j3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl1, MPI.DOUBLE], [ufl1_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl2, MPI.DOUBLE], [ufl2_total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    cartcomm.Reduce([ufl3, MPI.DOUBLE], [ufl3_total, MPI.DOUBLE], op = MPI.SUM, root = 0)

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

def save_triangle_fields_parallel_2d(comm, species, t, raw_folder,
                                     output_folder, deposit_n_x, deposit_n_y):
    input_filename = raw_folder + '/RAW-' + species + '-' + str(t).zfill(6) + '.h5'

    f_input = h5py.File(input_filename, 'r', driver='mpio', comm=comm)

    n_proc_x = f_input.attrs['PAR_NODE_CONF'][0]
    n_proc_y = f_input.attrs['PAR_NODE_CONF'][1]

    cartcomm = comm.Create_cart([n_proc_y, n_proc_x], periods=[True,True])

    rank = cartcomm.Get_rank()
    coords = cartcomm.Get_coords(rank)

    n_p_total = f_input['x1'].shape[0]
    n_ppp = n_p_total / (n_proc_x * n_proc_y)
    n_ppp_x = utilities.int_nth_root(n_ppp, 2)
    n_ppp_y = n_ppp_x

    axis = np.zeros([2,2], dtype='double')    
    axis[0,0] = f_input.attrs['XMIN'][0]
    axis[0,1] = f_input.attrs['XMAX'][0]
    axis[1,0] = f_input.attrs['XMIN'][1]
    axis[1,1] = f_input.attrs['XMAX'][1]

    l_x = axis[0,1] - axis[0,0]
    l_y = axis[1,1] - axis[1,0]

    axis_lagrangian = np.zeros_like(axis)
    axis_lagrangian[0,0] = 0.0
    axis_lagrangian[0,1] = n_proc_y * n_ppp_y
    axis_lagrangian[1,0] = 0.0
    axis_lagrangian[1,1] = n_proc_x * n_ppp_x

    axis_p2p1 = np.zeros_like(axis)
    axis_p2p1[0,0] = -1.0
    axis_p2p1[0,1] = 1.0
    axis_p2p1[1,0] = -0.5
    axis_p2p1[1,1] = 1.5

    if (rank==0):
        t_start = MPI.Wtime()

    # Get particle data for this processor's lagrangian subdomain
    [particle_positions, particle_momentum] = ship.ship_particle_data(cartcomm,
                                                                      f_input, 2)

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Time for particle data shipping:')
        print(t_elapsed)

    time = f_input.attrs['TIME']

    f_input.close()

    # Parameters for PSI
    window = ((axis[0,0], axis[1,0]), (axis[0,1], axis[1,1]))
    box = window
    
    window_lagrangian = ((axis_lagrangian[0,0], axis_lagrangian[1,0]), (axis_lagrangian[0,1], axis_lagrangian[1,1]))
    box_lagrangian = window_lagrangian

    window_p2p1 = ((axis_p2p1[0,0], axis_p2p1[1,0]), (axis_p2p1[0,1], axis_p2p1[1,1]))
    box_p2p1 = window_p2p1
    
    particle_velocities = oi.momentum_to_velocity(particle_momentum)

    # Next do triangle deposits
    particle_positions = particle_positions.reshape(n_ppp_y, n_ppp_x, 2)
    particle_velocities = particle_velocities.reshape(n_ppp_y, n_ppp_x, 3)
    particle_momentum = particle_momentum.reshape(n_ppp_y, n_ppp_x, 3)
    
    particle_positions_extended = extend.extend_lagrangian_quantity_2d(cartcomm, particle_positions)
    particle_velocities_extended = extend.extend_lagrangian_quantity_2d(cartcomm, particle_velocities)
    particle_momentum_extended = extend.extend_lagrangian_quantity_2d(cartcomm, particle_momentum)

    n_l_x = n_ppp_x + 1
    n_l_y = n_ppp_y + 1

    lagrangian_positions_extended = np.zeros([n_l_y, n_l_x, 2], dtype='float64')
    lagrangian_positions_extended[:,:,0] = np.repeat(np.arange(0, n_l_y, 1), n_l_x).reshape(n_l_y, n_l_x) + coords[0] * n_ppp_y
    lagrangian_positions_extended[:,:,1] = np.tile(np.arange(0, n_l_x, 1), n_l_y).reshape(n_l_y, n_l_x) + coords[1] * n_ppp_x

    step_list = [1]
    refine_list = [1]

    for step in step_list:
        # First do particle deposits
        deposit_n_ppc = n_p_total / float(deposit_n_x * deposit_n_y * step**2)
        particle_charge = -1.0 / deposit_n_ppc
        pos_particles = np.array(particle_positions_extended[:-1:step, :-1:step, :].reshape(n_ppp_y * n_ppp_x / float(step**2), 2), copy=True)
        mom_particles = np.array(particle_momentum_extended[:-1:step, :-1:step, :].reshape(n_ppp_y * n_ppp_x / float(step**2), 3), copy=True)
        vel_particles = np.array(particle_velocities_extended[:-1:step, :-1:step, :].reshape(n_ppp_y * n_ppp_x / float(step**2), 3), copy=True)
        
        for refine in refine_list:
            sub_folder = '/' + str(step) + '/' + str(deposit_n_x*refine) + 'x' + str(deposit_n_y*refine) + '/'
            charge = particle_charge * np.ones(n_ppp / step**2) * refine**2
            grid = (deposit_n_x*refine, deposit_n_y*refine)
            
            for deposit in ['ngp', 'cic', 'quadratic', 'cubic', 'quartic']:
                if (rank==0):
                    t_start = MPI.Wtime()

#                deposit_particles_ctypes(cartcomm, pos_particles, mom_particles, vel_particles, charge, grid, window,
#                                      box, output_folder + sub_folder, species, t,
#                                      time, axis, deposit_type=deposit)
                deposit_particles_p2p1_ctypes(cartcomm, mom_particles, charge, grid, window_p2p1,
                                       box_p2p1, output_folder + sub_folder,
                                       species, t, time, axis_p2p1, deposit=deposit)
                if (rank==0):
                    t_end = MPI.Wtime()
                    t_elapsed = t_end - t_start
                    print('Time for ' + deposit.upper() + ' deposit:')
                    print(t_elapsed)

        # Second, do triangle deposits
        pos = np.array(get_triangles_array(particle_positions_extended[::step,::step,:]), copy=True)
        pos_lagrangian = np.array(get_triangles_array(lagrangian_positions_extended[::step,::step,:]), copy=True)
        vel = np.array(get_triangles_array(particle_velocities_extended[::step,::step,:]), copy=True)
        mom = np.array(get_triangles_array(particle_momentum_extended[::step,::step,:]), copy=True)

        ntri = pos.shape[0]
        area = calculate_triangle_areas(pos, l_x, l_y)
                    
        for refine in refine_list:
            if (rank==0):
                t_start = MPI.Wtime()

            sub_folder = '/' + str(step) + '/' + str(deposit_n_x*refine) + 'x' + str(deposit_n_y*refine) + '/'
            charge = (particle_charge * np.ones(ntri) / 2.0) * refine**2
            grid = (deposit_n_x*refine, deposit_n_y*refine)
        
#            deposit_triangles_current(cartcomm, pos, vel, charge, grid, window, box,
#                                      output_folder + sub_folder, species, t, time,
#                                      axis)
#            deposit_triangles_momentum(cartcomm, pos, mom, charge, grid, window,
#                                       box, output_folder + sub_folder, species, t,
#                                       time, axis)
#            deposit_triangles_momentum_lagrangian(cartcomm, pos_lagrangian, mom,
#                                                  charge, grid, window_lagrangian,
#                                                  box_lagrangian,
#                                                  output_folder + sub_folder,
#                                                  species, t, time, axis_lagrangian)
#            deposit_triangles_current_points(cartcomm, pos, vel, charge, area, grid,
#                                             window, box,
#                                             output_folder + sub_folder, species, t,
#                                             time, axis)
#            deposit_triangles_p2p1(cartcomm, mom, charge, grid, window_p2p1,
#                                   box_p2p1, output_folder + sub_folder, species, t,
#                                   time, axis_p2p1)

            if (rank==0):
                t_end = MPI.Wtime()
                t_elapsed = t_end - t_start
                print('Time for triangle deposit:')
                print(t_elapsed)

    return

def save_triangle_fields_parallel_3d_new(comm, species, t, raw_folder,
                                     output_folder, deposit_n_x, deposit_n_y,
                                     deposit_n_z):
    input_filename = raw_folder + '/RAW-' + species + '-' + str(t).zfill(6) + '.h5'

    f_input = h5py.File(input_filename, 'r', driver='mpio', comm=comm)

    n_proc_x = f_input.attrs['PAR_NODE_CONF'][0]
    n_proc_y = f_input.attrs['PAR_NODE_CONF'][1]
    n_proc_z = f_input.attrs['PAR_NODE_CONF'][2]

    cartcomm = comm.Create_cart([n_proc_z, n_proc_y, n_proc_x], periods=[True, True, True])

    rank = cartcomm.Get_rank()
    coords = cartcomm.Get_coords(rank)

    n_p_total = f_input['x1'].shape[0]
    n_ppp = n_p_total / (n_proc_x * n_proc_y * n_proc_z)
    n_ppp_x = utilities.int_nth_root(n_ppp, 3)
    n_ppp_y = n_ppp_x
    n_ppp_z = n_ppp_x

    axis = np.zeros([2,2], dtype='double')    
    axis[0,0] = f_input.attrs['XMIN'][0]
    axis[0,1] = f_input.attrs['XMAX'][0]
    axis[1,0] = f_input.attrs['XMIN'][1]
    axis[1,1] = f_input.attrs['XMAX'][1]
    axis[2,0] = f_input.attrs['XMIN'][2]
    axis[2,1] = f_input.attrs['XMAX'][2]

    l_x = axis[0,1] - axis[0,0]
    l_y = axis[1,1] - axis[1,0]
    l_z = axis[2,1] - axis[2,0]


    if (rank==0):
        t_start = MPI.Wtime()

    # Get particle data for this processor's lagrangian subdomain
    [particle_positions, particle_momentum] = ship.ship_particle_data(cartcomm,
                                                                      f_input,
                                                                      3)

    if (rank==0):
        t_end = MPI.Wtime()
        t_elapsed = t_end - t_start
        print('Time for particle data shipping:')
        print(t_elapsed)

    time = f_input.attrs['TIME']

    f_input.close()

    # Parameters for PSI
    window = ((axis[0,0], axis[1,0], axis[2,0]), (axis[0,1], axis[1,1], axis[2,1]))
    box = window
    
    particle_velocities = oi.momentum_to_velocity(particle_momentum)

    # Next do triangle deposits
    particle_positions = particle_positions.reshape(n_ppp_y, n_ppp_x, 3)
    particle_velocities = particle_velocities.reshape(n_ppp_y, n_ppp_x, 3)
    particle_momentum = particle_momentum.reshape(n_ppp_y, n_ppp_x, 3)
    
    particle_positions_extended = extend.extend_lagrangian_quantity_2d(cartcomm, particle_positions)
    particle_velocities_extended = extend.extend_lagrangian_quantity_2d(cartcomm, particle_velocities)
    particle_momentum_extended = extend.extend_lagrangian_quantity_2d(cartcomm, particle_momentum)

    refine_list = [1]

    # First do particle deposits
    deposit_n_ppc = n_p_total / float(deposit_n_x * deposit_n_y * deposit_n_z)
    particle_charge = -1.0 / deposit_n_ppc
    pos_particles = np.array(particle_positions_extended[:-1:, :-1:, :].reshape(n_ppp_y * n_ppp_x, 3), copy=True)
    mom_particles = np.array(particle_momentum_extended[:-1:, :-1:, :].reshape(n_ppp_y * n_ppp_x, 3), copy=True)
    vel_particles = np.array(particle_velocities_extended[:-1:, :-1:, :].reshape(n_ppp_y * n_ppp_x, 3), copy=True)
        
    for refine in refine_list:
        sub_folder = '/' + str(deposit_n_x*refine) + 'x' + str(deposit_n_y*refine) + '/'
        charge = particle_charge * np.ones(n_ppp) * refine**3
        grid = (deposit_n_x*refine, deposit_n_y*refine, deposit_n_z*refine)
            
        for deposit in ['ngp', 'cic']:
            if (rank==0):
                t_start = MPI.Wtime()
                
            deposit_particles_psi(cartcomm, pos_particles, mom_particles, vel_particles, charge, grid, window,
                                  box, output_folder + sub_folder, species, t,
                                  time, axis, deposit=deposit)
            if (rank==0):
                t_end = MPI.Wtime()
                t_elapsed = t_end - t_start
                print('Time for ' + deposit.upper() + ' deposit:')
                print(t_elapsed)

    # Second, do triangle deposits
    pos = np.array(get_triangles_array(particle_positions_extended[:,:,:]), copy=True)
    vel = np.array(get_triangles_array(particle_velocities_extended[:,:,:]), copy=True)
    mom = np.array(get_triangles_array(particle_momentum_extended[:,:,:]), copy=True)

    ntri = pos.shape[0]
                    
    for refine in refine_list:
        if (rank==0):
            t_start = MPI.Wtime()

        sub_folder = '/' + str(step) + '/' + str(deposit_n_x*refine) + 'x' + str(deposit_n_y*refine) + '/'
        charge = (particle_charge * np.ones(ntri) / 2.0) * refine**3
        grid = (deposit_n_x*refine, deposit_n_y*refine)
        
        deposit_triangles_current(cartcomm, pos, vel, charge, grid, window, box,
                                  output_folder + sub_folder, species, t, time,
                                  axis)

        if (rank==0):
            t_end = MPI.Wtime()
            t_elapsed = t_end - t_start
            print('Time for triangle deposit:')
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

def distribution_function_2d(comm, species, t, raw_folder, output_folder,
                             sample_locations):
    # Load raw data to be deposited
    input_filename = raw_folder + "/RAW-" + species + "-" + str(t).zfill(6) + ".h5"

    f_input = h5py.File(input_filename, 'r', driver='mpio', comm=comm)

    n_proc_x = f_input.attrs['PAR_NODE_CONF'][0]
    n_proc_y = f_input.attrs['PAR_NODE_CONF'][1]
    n_cell_x = f_input.attrs['NX'][0]
    n_cell_y = f_input.attrs['NX'][1]

    l_x = f_input.attrs['XMAX'][0] - f_input.attrs['XMIN'][0]
    l_y = f_input.attrs['XMAX'][1] - f_input.attrs['XMIN'][1]

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
        if (rank==0):
            t_start = MPI.Wtime()

        sample_position = np.ones([pos.shape[0],2], dtype='float64')
        sample_position[:,1] = sample_locations[i][1]
        sample_position[:,0] = sample_locations[i][0]
        # Shift triangle vertices to account for periodic boundaries
        max_x = np.amax(pos[:,:,1], axis=1)
        max_y = np.amax(pos[:,:,0], axis=1)
        shift_x = (max_x[:,None] - pos[:,:,1]) > (l_x/2.0)
        shift_y = (max_y[:,None] - pos[:,:,0]) > (l_y/2.0)
        pos[:,:,0] = pos[:,:,0] + shift_y * l_y
        pos[:,:,1] = pos[:,:,1] + shift_x * l_x
        
        max_x = np.amax(pos[:,:,1], axis=1)
        max_y = np.amax(pos[:,:,0], axis=1)
        shift_x = (max_x - sample_position[:,1]) >= l_x
        shift_y = (max_y - sample_position[:,0]) >= l_y
        sample_position[:,1] = sample_position[:,1] + shift_x * l_x
        sample_position[:,0] = sample_position[:,0] + shift_y * l_y

        # Calculate area of triangle    
        det = (pos[:,1,0]-pos[:,2,0])*(pos[:,0,1]-pos[:,2,1])+(pos[:,2,1]-pos[:,1,1])*(pos[:,0,0]-pos[:,2,0])
        # Calculate barycentric coordinates
        l1 = ((pos[:,1,0]-pos[:,2,0])*(sample_position[:,1]-pos[:,2,1])+(pos[:,2,1]-pos[:,1,1])*(sample_position[:,0]-pos[:,2,0])) / det
        l2 = ((pos[:,2,0]-pos[:,0,0])*(sample_position[:,1]-pos[:,2,1])+(pos[:,0,1]-pos[:,2,1])*(sample_position[:,0]-pos[:,2,0])) / det
        l3 = 1.0 - l1 - l2
        # Find triangles that contain the sample location, including edges and vertices
        sample_inside = (l1<=1.0)*(l1>=0.0)*(l2<=1.0)*(l2>=0.0)*(l3<=1.0)*(l3>=0.0)
        sample_indices = np.where(sample_inside==True)[0]
        l1_sample = l1[sample_indices]
        l2_sample = l2[sample_indices]
        l3_sample = l3[sample_indices]
        momentum_sample = momentum[sample_indices]

        # Interpolate three momentum components to sample location
        px_sample = l1_sample * momentum_sample[:,0,2] + l2_sample * momentum_sample[:,1,2] + l3_sample * momentum_sample[:,2,2]
        py_sample = l1_sample * momentum_sample[:,0,1] + l2_sample * momentum_sample[:,1,1] + l3_sample * momentum_sample[:,2,1]
        pz_sample = l1_sample * momentum_sample[:,0,0] + l2_sample * momentum_sample[:,1,0] + l3_sample * momentum_sample[:,2,0]
        area_sample = 0.5 * np.abs(det[sample_indices])

        # Send this processor's number of streams to root
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
            h5f['sample_' + str(i)].attrs['x'] = sample_locations[i][1]
            h5f['sample_' + str(i)].attrs['y'] = sample_locations[i][0]
            h5f['sample_' + str(i)].attrs['time'] = time

        if (rank==0):
            t_end = MPI.Wtime()
            t_elapsed = t_end - t_start
            print('Time for distribution function sample point ' + str(i))
            print(t_elapsed)


    if (rank==0):
            h5f.close()

    return

def distribution_save_triangles(comm, species, t, raw_folder, output_folder):
    # Load raw data to be deposited
    input_filename = raw_folder + "/RAW-" + species + "-" + str(t).zfill(6) + ".h5"

    f_input = h5py.File(input_filename, 'r', driver='mpio', comm=comm)

    n_proc_x = f_input.attrs['PAR_NODE_CONF'][0]
    n_proc_y = f_input.attrs['PAR_NODE_CONF'][1]
    n_cell_x = f_input.attrs['NX'][0]
    n_cell_y = f_input.attrs['NX'][1]

    l_x = f_input.attrs['XMAX'][0] - f_input.attrs['XMIN'][0]
    l_y = f_input.attrs['XMAX'][1] - f_input.attrs['XMIN'][1]

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
    sample_position = np.ones([pos.shape[0],2], dtype='float64')
    sample_position[:,1] = sample_locations[i][1]
    sample_position[:,0] = sample_locations[i][0]
    # Shift triangle vertices to account for periodic boundaries
    max_x = np.amax(pos[:,:,1], axis=1)
    max_y = np.amax(pos[:,:,0], axis=1)
    shift_x = (max_x[:,None] - pos[:,:,1]) > (l_x/2.0)
    shift_y = (max_y[:,None] - pos[:,:,0]) > (l_y/2.0)
    pos[:,:,0] = pos[:,:,0] + shift_y * l_y
    pos[:,:,1] = pos[:,:,1] + shift_x * l_x
    
    max_x = np.amax(pos[:,:,1], axis=1)
    max_y = np.amax(pos[:,:,0], axis=1)
    shift_x = (max_x - sample_position[:,1]) >= l_x
    shift_y = (max_y - sample_position[:,0]) >= l_y
    sample_position[:,1] = sample_position[:,1] + shift_x * l_x
    sample_position[:,0] = sample_position[:,0] + shift_y * l_y

    # Calculate area of triangle    
    det = (pos[:,1,0]-pos[:,2,0])*(pos[:,0,1]-pos[:,2,1])+(pos[:,2,1]-pos[:,1,1])*(pos[:,0,0]-pos[:,2,0])
    
    # Send this processor's number of streams to root
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
    
    # Save distribution functions
    if (rank==0):
        h5f.create_dataset('sample_' + str(i) + '/px', data=px_dist_total)
        h5f.create_dataset('sample_' + str(i) + '/py', data=py_dist_total)
        h5f.create_dataset('sample_' + str(i) + '/pz', data=pz_dist_total)
        h5f.create_dataset('sample_' + str(i) + '/area', data=area_dist_total)
        h5f['sample_' + str(i)].attrs['x'] = sample_locations[i][1]
        h5f['sample_' + str(i)].attrs['y'] = sample_locations[i][0]
        h5f['sample_' + str(i)].attrs['time'] = time

    if (rank==0):
            h5f.close()

    return
