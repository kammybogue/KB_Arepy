import numpy as np
import matplotlib.pyplot as pl
from arepy.read_write import binary_read as rsnap


### 1...
header = rsnap.return_header("template_snapshot")

print header
raw_input('Press enter to continue: ')

### 2...
data, header = rsnap.read_IC("template_IC")
print header
print data
raw_input('Press enter to continue: ')

### 3...
image = rsnap.read_image("template_image")
pl.imshow(np.log10(image))
pl.show()
raw_input('Press enter to continue: ')
pl.close()

### 4...
image = rsnap.read_vector_image("template_vel_image")
X, Y = np.meshgrid(np.linspace(0., 1., image.size[1]), np.linspace(0., 1., image.size[1]))
pl.quiver(X, Y, image[:,:,0], image[:,:,1])
raw_input('Press enter to continue: ')
pl.close()

### 5...
rsnap.io_flags['sgchem'] = False
print rsnap.io_flags
data1, header1 = rsnap.read_snapshot("template_snapshot")
print header1
print data1
raw_input('Press enter to continue: ')

### 6...
rsnap.io_flags['sgchem'] = True
print rsnap.io_flags
data2, header2 = rsnap.read_snapshot("template_snapshot_sgchem")
print header2
print data2
raw_input('Press enter to continue: ')

### 7...
rsnap.io_flags['sgchem_NL99'] = True
print rsnap.io_flags
data3, header3 = rsnap.read_snapshot("template_snapshot_sgchem_NL99")
print header3
print data3
raw_input('Press enter to continue: ')

### 8...
rsnap.io_flags['sgchem_NL99'] = False
rsnap.io_flags['variable_metallicity'] = True
print rsnap.io_flags
data4, header4 = rsnap.read_snapshot("template_snapshot_variable_Z")
print header4
print data4
raw_input('Press enter to continue: ')

### 9...
rsnap.io_flags['variable_metallicity'] = False
rsnap.io_flags['mc_tracer'] = True
print rsnap.io_flags
data4, header4 = rsnap.read_snapshot("template_snapshot_with_tracers")
print header4
print data4
raw_input('Press enter to continue: ')

