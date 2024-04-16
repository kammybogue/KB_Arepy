import os
import numpy as np
from arepy.read_write import binary_write as wsnap
from arepy.read_write import binary_read as rsnap

### 1...
rsnap.io_flags['sgchem'] = False
data1, header1 = rsnap.read_snapshot("template_snapshot")
wsnap.write_IC(data1['pos'], data1['vel'], data1['mass'], data1['u_therm'], 'tmp_IC')

ICdata, ICheader = rsnap.read_IC("tmp_IC")
print ICheader
print ICdata
os.remove('tmp_IC')
raw_input('Press enter to continue: ')

### 2...
wsnap.write_snapshot('tmp_snap', header1, data1)
snapdata, snapheader = rsnap.read_snapshot("tmp_snap")
print snapheader
print snapdata
os.remove('tmp_snap')
raw_input('Press enter to continue: ')

### 3...
rsnap.io_flags['sgchem'] = True
print rsnap.io_flags
data2, header2 = rsnap.read_snapshot("template_snapshot_sgchem")
wsnap.write_snapshot('tmp_snap', header2, data2)

data2, header2 = rsnap.read_snapshot("tmp_snap")
print header2
print data2
os.remove('tmp_snap')
raw_input('Press enter to continue: ')

### 4...
rsnap.io_flags['sgchem_NL99'] = True
print rsnap.io_flags
data3, header3 = rsnap.read_snapshot("template_snapshot_sgchem_NL99")
wsnap.write_snapshot('tmp_snap', header3, data3)

data3, header3 = rsnap.read_snapshot("tmp_snap")
print header3
print data3
os.remove('tmp_snap')
raw_input('Press enter to continue: ')

### 5...
rsnap.io_flags['sgchem_NL99'] = False
rsnap.io_flags['variable_metallicity'] = True
print rsnap.io_flags
data4, header4 = rsnap.read_snapshot("template_snapshot_variable_Z")
wsnap.write_snapshot('tmp_snap', header4, data4)

data4, header4 = rsnap.read_snapshot("tmp_snap")
print header4
print data4
os.remove('tmp_snap')
raw_input('Press enter to continue: ')

### 6...
rsnap.io_flags['variable_metallicity'] = False
rsnap.io_flags['mc_tracer'] = True
print rsnap.io_flags
data5, header5 = rsnap.read_snapshot("template_snapshot_with_tracers")
wsnap.write_snapshot('tmp_snap', header5, data5)

data5, header5 = rsnap.read_snapshot("tmp_snap")
print header5
print data5
os.remove('tmp_snap')
raw_input('Press enter to continue: ')

