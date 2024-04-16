import numpy as np
from numpy import uint32, uint64, float64, float32

header_names = ('num_particles', 'mass', 'time', 'redshift', 'flag_sfr', 'flag_feedback', 'num_particles_total', 'flag_cooling', \
                'num_files', 'boxsize', 'omega0', 'omegaLambda', 'hubble0', 'flag_stellarage', 'flag_metals', \
                'npartTotalHighWord', 'flag_entropy_instead_u', 'flag_doubleprecision', \
                'flag_lpt_ics', 'lpt_scalingfactor', 'flag_tracer_field', 'composition_vector_length', 'buffer')

header_sizes = ((uint32, 6), (float64, 6), (float64, 1), (float64, 1), (uint32, 1), (uint32, 1), \
                (uint32, 6), (uint32, 1), (uint32, 1), (float64, 1), (float64, 1), (float64, 1), \
                (float64, 1), (uint32, 1), (uint32, 1), (uint32, 6), (uint32, 1), (uint32, 1), \
                (uint32, 1), (float32, 1), (uint32, 1), (uint32, 1), (np.uint8, 40))

def read_header(f):
  """ Read the binary header file into a dictionary """
  block_size = np.fromfile(f, uint32, 1)[0]
  header = dict(((name, np.fromfile(f, dtype=size[0], count=size[1])) \
               for name, size in zip(header_names, header_sizes)))

  assert(np.fromfile(f, uint32, 1)[0] == 256)
  return header

def readu(f, dtype=None, components=1):
    """ Read a numpy array from the unformatted fortran file f """
    data_size = np.fromfile(f, uint32, 1)[0]

    count = data_size/np.dtype(dtype).itemsize
    arr = np.fromfile(f, dtype, count)

    final_block = np.fromfile(f, uint32, 1)[0]

    # check the flag at the beginning corresponds to that at the end
    assert(data_size == final_block)

    return arr

def readIDs(f, count):
    """ Read a the ID block from a binary snapshot file """
    data_size = np.fromfile(f, uint32, 1)[0]
    f.seek(-4, 1)

    count = int(count)
    if data_size / 4 == count: dtype = uint32
    elif data_size / 8 == count: dtype = uint64
    else: raise Exception('Incorrect number of IDs requested')

    print "ID type: ", dtype

    return readu(f, dtype, 1)

#constants
mp = 1.6726231e-24 
kb = 1.380658e-16

#internal arepo units in cgs
arepoLength = 3.0856e20
arepoMass = 1.991e33
arepoVel = 1.0e5

arepoTime = arepoLength/arepoVel
arepoDensity = arepoMass/arepoLength/arepoLength/arepoLength
arepoEnergy= arepoMass*arepoVel*arepoVel
arepoColumnDensity = arepoMass/arepoLength/arepoLength

""" Reads a binary arepo file """
filename = 'SNAP_AREPO_MW'
f = open(filename, mode='rb')
print "Loading file %s" % filename

data = {} # dictionary to hold data

# read the header
header = read_header(f)

nparts = header['num_particles']
masses = header['mass']

total = nparts.sum()
n_gas = nparts[0]

print 'Particles', nparts
print 'Gas particles', n_gas
print 'Time = ', header['time']

precision = float32

print 'Reading positions'
data['pos'] = readu(f, precision, 3).reshape((total, 3))
print 'Reading velocities'
data['vel'] = readu(f, precision, 3).reshape((total, 3))
print 'Reading IDs'
data['id'] = readIDs(f, total)
print 'Reading masses'
data['mass'] = readu(f, precision, 1)
print 'Reading internal energy'
data['u_therm'] = readu(f, precision, 1)
print 'Reading densities'
data['rho'] = readu(f, precision, 1)
print 'Reading chemical abundances'
data['chem'] = readu(f, precision, 3).reshape((n_gas, 3))
print 'Reading dust temperatures'
data['tdust'] = readu(f, precision, 1)

print 'Converting internal energy into temperatures'

ABHE=0.1
yn = data['rho']*arepoDensity / ((1. + 4.0 * ABHE) * mp)
energy = data['u_therm'] * data['rho'] * arepoEnergy/arepoLength/arepoLength/arepoLength
yntot = (1. + ABHE - data['chem'][:,0] + data['chem'][:,1]) * yn
data['temp'] = 2.*energy / (3.*yntot*kb)


f.close()

