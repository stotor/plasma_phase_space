import numpy as np
from mpi4py import MPI
import h5py

def osiris_tag_to_lagrangian(osiris_id, n_cell_proc_x, n_cell_proc_y, n_ppc_x, n_ppc_y):
    """Converts the OSIRIS particle ID within a processor, tag[i,1], into the 
    Lagrangian coordinate within that processors subdomain."""

    # Convert to zero based indexing
    osiris_id = osiris_id - 1
    
    n_ppc = n_ppc_x * n_ppc_y

    cell = osiris_id / n_ppc
    cell_x = cell / (n_cell_proc_y)
    cell_y = cell - (cell_x * n_cell_proc_y)

    sub_cell = osiris_id - cell * n_ppc
    sub_cell_x = sub_cell / n_ppc_y
    sub_cell_y = sub_cell - (sub_cell_x * n_ppc_y)

    lagrangian_x = cell_x * n_ppc_x + sub_cell_x
    lagrangian_y = cell_y * n_ppc_y + sub_cell_y

    # Lagrangian coordinate, with x increasing fastest
    lagrangian = lagrangian_y * n_cell_proc_x * n_ppc_x + lagrangian_x
    
    return lagrangian

def save_raw_sorted_parallel(comm, input_filename, output_filename):
    """Loads the raw data file corresponding to input_filename, and saves
    the data sorted first by processor, then by Lagrangian coordinate within
    each processor's subdomain, to output_filename."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load raw data to be sorted
    f_input = h5py.File(input_filename, 'r', driver='mpio', comm=comm)

    n_proc_x = f_input.attrs['PAR_NODE_CONF'][0]
    n_proc_y = f_input.attrs['PAR_NODE_CONF'][1]
    n_cell_x = f_input.attrs['NX'][0]
    n_cell_y = f_input.attrs['NX'][1]
    
    n_cell_proc_x = n_cell_x / n_proc_x
    n_cell_proc_y = n_cell_y / n_proc_y

    n_p_total = f_input['x1'].shape[0]
    #n_p_total = (n_proc_x * n_proc_y) * n_ppp
    n_ppc = n_p_total / (n_cell_x * n_cell_y)
    n_ppc_x = int(np.sqrt(n_ppc))
    n_ppc_y = n_ppc_x
    
    # Number of particles per processor
    n_ppp = (n_ppc_x * n_ppc_y) * (n_cell_proc_x * n_cell_proc_y)

    # Constency checks
    #(size==(n_proc_x*n_proc_y))
    # Check that domain size is equivalent for all processors and is an integer number
    # Check that total number of particles is as expected
    # Check that number of particles per cell is a perfect square

    i_start = rank * n_ppp
    i_end = (rank + 1) * n_ppp

    x1 = f_input['x1'][i_start:i_end]
    x2 = f_input['x2'][i_start:i_end]
    p1 = f_input['p1'][i_start:i_end]
    p2 = f_input['p2'][i_start:i_end]
    p3 = f_input['p3'][i_start:i_end]
    tag = f_input['tag'][i_start:i_end] 

    keys = f_input.attrs.keys()
    values = f_input.attrs.values()

    f_input.close()

    # Get memory location for saving
    processor = tag[:,0] - 1 # Convert to zero based indexing
    lagrangian_id = osiris_tag_to_lagrangian(tag[:,1], 
                                             n_cell_proc_x, 
                                             n_cell_proc_y, 
                                             n_ppc_x, 
                                             n_ppc_y)
    memory_index = processor * n_ppp + lagrangian_id

    # Save sorted raw data
    f_output = h5py.File(output_filename, 'w', driver='mpio', comm=comm)
    
    for i in range(len(keys)):
        f_output.attrs.create(keys[i], values[i])

    f_output.create_dataset('x1', (n_p_total,), dtype='float32')
    f_output.create_dataset('x2', (n_p_total,), dtype='float32') 
    f_output.create_dataset('p1', (n_p_total,), dtype='float32') 
    f_output.create_dataset('p2', (n_p_total,), dtype='float32') 
    f_output.create_dataset('p3', (n_p_total,), dtype='float32') 
    f_output.create_dataset('processor', (n_p_total,), dtype='int32')
    f_output.create_dataset('lagrangian_id', (n_p_total,), dtype='int32')

    for i in range(len(memory_index)):
        f_output['x1'][memory_index[i]] = x1[i]
        f_output['x2'][memory_index[i]] = x2[i]
        f_output['p1'][memory_index[i]] = p1[i]
        f_output['p2'][memory_index[i]] = p2[i]
        f_output['p3'][memory_index[i]] = p3[i]
        f_output['processor'][memory_index[i]] = processor[i]
        f_output['lagrangian_id'][memory_index[i]] = lagrangian_id[i]
    f_output.close()


