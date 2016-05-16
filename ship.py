import numpy as np
from mpi4py import MPI
import h5py

import utilities
import osiris_interface as oi

def ship_particle_data(comm, raw_h5f, dim):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Find the indices that define my section of the raw data
    n_p_total = raw_h5f['x1'].shape[0]
    n_ppp = n_p_total / size
    i_start = rank * n_ppp
    i_end = (rank + 1) * n_ppp

    # Load my chunk of the raw data
    processor_list = raw_h5f['tag'][i_start:i_end,0] - 1 # To account for OSIRIS 1 based indexing for processor number
    particle_id = raw_h5f['tag'][i_start:i_end,1]

    x1 = raw_h5f['x1'][i_start:i_end]
    x2 = raw_h5f['x2'][i_start:i_end]
    if (dim==3):
        x3 = raw_h5f['x3'][i_start:i_end]
    p1 = raw_h5f['p1'][i_start:i_end]
    p2 = raw_h5f['p2'][i_start:i_end]
    p3 = raw_h5f['p3'][i_start:i_end]

    # Sort by processor
    processor_sort_keys = np.argsort(processor_list)
    particle_id = particle_id[processor_sort_keys]
    x1 = x1[processor_sort_keys]
    x2 = x2[processor_sort_keys]
    if (dim==3):
        x3 = x3[processor_sort_keys]
    p1 = p1[processor_sort_keys]
    p2 = p2[processor_sort_keys]
    p3 = p3[processor_sort_keys]

    # Make sure all arrays are doubles and pack together
    if (dim==2):
        particle_data_send = np.zeros([n_ppp, 6], dtype='double')
        particle_data_send[:,0] = particle_id
        particle_data_send[:,1] = x2
        particle_data_send[:,2] = x1
        particle_data_send[:,3] = p3
        particle_data_send[:,4] = p2
        particle_data_send[:,5] = p1
    elif (dim==3):
        particle_data_send = np.zeros([n_ppp, 7], dtype='double')
        particle_data_send[:,0] = particle_id
        particle_data_send[:,1] = x3
        particle_data_send[:,2] = x2
        particle_data_send[:,3] = x1
        particle_data_send[:,4] = p3
        particle_data_send[:,5] = p2
        particle_data_send[:,6] = p1

    # Find the number of particles I will send to each processor
    bins = np.arange(size+1)
    particle_count_send = np.array(np.histogram(processor_list, bins)[0], dtype='i')

    # Nonblocking send to each processor the number of particles to expect from me
    for proc in range(size):
        comm.Isend([particle_count_send[proc:proc+1], MPI.INT], proc, tag=0)

    # Blocking receive from each processor the number of particles to expect
    particle_count_receive = np.zeros_like(particle_count_send)
    for proc in range(size):
        comm.Recv([particle_count_receive[proc:proc+1], MPI.INT], proc, tag=0)

    comm.Barrier()

    # Find slicing indices for sending to each processor
    send_indices = np.concatenate([[0], np.cumsum(particle_count_send)])

    # Nonblocking send my particles to their respective processors
    for proc in range(size):
        if (particle_count_send[proc]==0):
            continue
        i_start = send_indices[proc]
        i_end = send_indices[proc+1]
        comm.Isend([particle_data_send[i_start:i_end,:], MPI.DOUBLE], proc, tag=0)

    # Blocking receive my data
    particle_data_receive = np.zeros_like(particle_data_send)
    receive_indices = np.concatenate([[0], np.cumsum(particle_count_receive)])
    for proc in range(size):
        if (particle_count_receive[proc]==0):
            continue
        i_start = receive_indices[proc]
        i_end = receive_indices[proc+1]
        comm.Recv([particle_data_receive[i_start:i_end,:], MPI.DOUBLE], proc, tag=0)

    comm.Barrier()

    # Sort data by lagrangian id
    particle_tag = particle_data_receive[:,0].astype('i')

    if (dim==2):
        n_proc_x = raw_h5f.attrs['PAR_NODE_CONF'][0]
        n_proc_y = raw_h5f.attrs['PAR_NODE_CONF'][1]
        n_cell_x = raw_h5f.attrs['NX'][0]
        n_cell_y = raw_h5f.attrs['NX'][1]
        n_cell_proc_x = n_cell_x / n_proc_x
        n_cell_proc_y = n_cell_y / n_proc_y
        n_ppc = n_p_total / (n_cell_x * n_cell_y)
        n_ppc_x = utilities.int_nth_root(n_ppc, 2)
        n_ppc_y = n_ppc_x
        lagrangian_id = oi.osiris_tag_to_lagrangian(particle_tag, 
                                                    n_cell_proc_x,
                                                    n_cell_proc_y, 
                                                    n_ppc_x, 
                                                    n_ppc_y)
    elif (dim==3):
        n_proc_x = raw_h5f.attrs['PAR_NODE_CONF'][0]
        n_proc_y = raw_h5f.attrs['PAR_NODE_CONF'][1]
        n_proc_z = raw_h5f.attrs['PAR_NODE_CONF'][2]
        n_cell_x = raw_h5f.attrs['NX'][0]
        n_cell_y = raw_h5f.attrs['NX'][1]
        n_cell_z = raw_h5f.attrs['NX'][2]
        n_cell_proc_x = n_cell_x / n_proc_x
        n_cell_proc_y = n_cell_y / n_proc_y
        n_cell_proc_z = n_cell_z / n_proc_z
        n_ppc = n_p_total / (n_cell_x * n_cell_y * n_cell_z)
        n_ppc_x = utilities.int_nth_root(n_ppc, 3)
        n_ppc_y = n_ppc_x
        n_ppc_z = n_ppc_x
        lagrangian_id = oi.osiris_tag_to_lagrangian_3d(particle_tag, 
                                                       n_cell_proc_x,
                                                       n_cell_proc_y, 
                                                       n_cell_proc_z, 
                                                       n_ppc_x, 
                                                       n_ppc_y,
                                                       n_ppc_z)

    lagrangian_sorting_keys = np.argsort(lagrangian_id)
    particle_data_receive = particle_data_receive[lagrangian_sorting_keys]
    if (dim==2):
        particle_positions = np.array(particle_data_receive[:,1:3])
        particle_momentum = np.array(particle_data_receive[:,3:6])
    elif (dim==3):
        particle_positions = np.array(particle_data_receive[:,1:4])
        particle_momentum = np.array(particle_data_receive[:,4:7])

    comm.Barrier()

    return [particle_positions, particle_momentum]
