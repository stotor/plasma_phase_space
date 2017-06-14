import numpy as np
import matplotlib.pyplot as plt
import ctypes

lib = ctypes.cdll['./pic_deposits.so']
deposit_species = lib['deposit_species']

n_p = 1
n_x = 10
n_y = 10
order = 4
particle_positions = np.zeros([n_p, 2], dtype='double')
particle_charges = np.ones(n_p, dtype='double')
field = np.zeros([n_y, n_x], dtype='double')
cell_width = 1.0

particle_positions[0, :] = [5.1, 5.1]

particle_positions_c = particle_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
field_c = field.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
particle_charges_c = particle_charges.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
n_p_c = ctypes.c_int(n_p)
n_x_c = ctypes.c_int(n_x)
n_y_c = ctypes.c_int(n_y)
order_c = ctypes.c_int(order)
cell_width_c = ctypes.c_double(cell_width)


deposit_species(particle_positions_c, field_c, particle_charges_c, n_x_c, n_y_c,
                n_p_c, cell_width_c, order_c)

plt.imshow(field, interpolation='nearest')
plt.colorbar()
plt.show()

print(np.amax(field))
