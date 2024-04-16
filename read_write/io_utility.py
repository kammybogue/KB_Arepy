import numpy as np
from numpy import uint32, uint64, float64, float32

########### GENERAL STUFF  ##########

#internal arepo units in cgs
arepoLength = 3.0856e20
arepoMass = 1.991e33
arepoVel = 1.0e5

arepoTime = arepoLength/arepoVel
arepoDensity = arepoMass/arepoLength/arepoLength/arepoLength
arepoEnergy= arepoMass*arepoVel*arepoVel
arepoColumnDensity = arepoMass/arepoLength/arepoLength

io_flags = {'mc_tracer'           : False,\
            'sgchem'              : True, \
            'variable_metallicity': False,\
            'sgchem_NL99'         : False}

########### SNAPSHOT FILE ##########
header_names = ('num_particles', 'mass', 'time', 'redshift', 'flag_sfr', 'flag_feedback', \
                'num_particles_total', 'flag_cooling', 'num_files', 'boxsize', 'omega0', 'omegaLambda', \
                'hubble0', 'flag_stellarage', 'flag_metals', 'npartTotalHighWord', 'flag_entropy_instead_u', 'flag_doubleprecision', \
                'flag_lpt_ics', 'lpt_scalingfactor', 'flag_tracer_field', 'composition_vector_length', 'buffer')

header_sizes = ((uint32, 6), (float64, 6), (float64, 1), (float64, 1), (uint32, 1), (uint32, 1), \
                (uint32, 6), (uint32, 1), (uint32, 1), (float64, 1), (float64, 1), (float64, 1), \
                (float64, 1), (uint32, 1), (uint32, 1), (uint32, 6), (uint32, 1), (uint32, 1), \
                (uint32, 1), (float32, 1), (uint32, 1), (uint32, 1), (np.uint8, 40))

########### IC FILE ##########
IC_header_names = ('num_particles', 'mass', 'time', 'redshift', 'flag_sfr', 'flag_feedback',              \
                   'num_particles_total', 'flag_cooling', 'num_files', 'boxsize', 'omega0', 'omegaLambda',\
                   'hubble0', 'flag_stellarage', 'flag_metals', 'npartTotalHighWord', 'utime', 'umass',   \
                   'udist', 'flag_entropyICs', 'unused')

IC_header_sizes = ((uint32, 6), (float64, 6),(float64, 1),(float64, 1),(uint32, 1), (uint32, 1),  \
                   (uint32, 6), (uint32, 1), (uint32, 1), (float64, 1),(float64, 1),(float64, 1), \
                   (float64, 1),(uint32, 1), (uint32, 1), (uint32, 6), (float64,1), (float64,1),  \
                   (float64,1), (uint32, 1), (np.uint8, 36))

########## necessary READ functions #########
def read_header(f, h_names, h_sizes):
    """ Read the binary header file into a dictionary """
    block_size = np.fromfile(f, uint32, 1)[0]
    header = dict(((name, np.fromfile(f, dtype=size[0], count=size[1])) \
                 for name, size in zip(h_names, h_sizes)))
                 
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

def readIDs(f, count):
    """ Read the ID block from a binary snapshot file """
    data_size = np.fromfile(f, uint32, 1)[0]
    f.seek(-4, 1)

    count = int(count)
    if data_size / 4 == count: dtype = uint32
    elif data_size / 8 == count: dtype = uint64
    else: raise Exception('Incorrect number of IDs requested')

    print "ID type: ", dtype

    return readu(f, dtype, 1)

########## necessary WRITE functions #########
def writeu(f, data):
    """ Write a numpy array to the unformatted fortran file f """
    data_size = data.size * data.dtype.itemsize
    data_size = np.array(data_size, dtype=uint32)

    data_size.tofile(f)

    data.tofile(f)

    data_size.tofile(f)
