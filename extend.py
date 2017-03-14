# Routines for calculating triangle quantities such as area

import numpy as np
from mpi4py import MPI

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

def extend_lagrangian_quantity_2d_serial(l_quant):
    l_quant_extended = np.zeros([l_quant.shape[0]+1,l_quant.shape[1]+1,l_quant.shape[2]], dtype=l_quant.dtype)

    l_quant_extended[:-1,:-1,:] = l_quant

    x1_face_send = np.array(l_quant_extended[:-1,0,:], copy=True)
    x1_face_recv = x1_face_send
    l_quant_extended[:-1,-1,:] = x1_face_recv

    x2_face_send = np.array(l_quant_extended[0,:,:], copy=True)
    x2_face_recv = x2_face_send
    l_quant_extended[-1,:,:] = x2_face_recv

    return l_quant_extended


def extend_lagrangian_quantity_3d(cartcomm, l_quant):
    rank = cartcomm.Get_rank()

    l_quant_extended = np.zeros([l_quant.shape[0]+1,l_quant.shape[1]+1,l_quant.shape[2]+1,l_quant.shape[3]], 
                                dtype=lquant.dtype)

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
