import numpy as np
from numpy import uint32, uint64, float64, float32
import h5py

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
            'time_steps'          : False,\
            'sgchem'              : False, \
            'variable_metallicity': False,\
            'sgchem_NL99'         : False,\
            'potential'           : False,\
            }

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

    count = data_size//np.dtype(dtype).itemsize
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
    if data_size // 4 == count: dtype = uint32
    elif data_size // 8 == count: dtype = uint64
    else: raise Exception('Incorrect number of IDs requested')

    print("ID type: ", dtype)

    return readu(f, dtype, 1)


########## necessary WRITE functions #########

def writeu(f, data):
    """ Write a numpy array to the unformatted fortran file f """
    data_size = data.size * data.dtype.itemsize
    data_size = np.array(data_size, dtype=uint32)

    data_size.tofile(f)

    data.tofile(f)

    data_size.tofile(f)

####### binary read functions #########

def return_header(filename):
    f = open(filename, mode='rb')
    header = read_header(f, header_names, header_sizes)
    f.close()
    return header

def read_IC(filename):
    f = open(filename, mode='rb')
    print("Loading IC file %s" % filename)

    data = {} # dictionary to hold data
  
    # read the header
    header = read_header(f,IC_header_names, IC_header_sizes)
  
    nparts = header['num_particles']
  
    npts = nparts.sum()
    precision = float32
    ranges=None
  
    data['pos'] = readu(f, precision, components=3).reshape(npts, 3)
    data['vel'] = readu(f, precision, components=3).reshape(npts, 3)
    data['id'] = readIDs(f, npts)
  
    # there are two methods of defining the masses in arepo, either by defining the mass in the mass array in the header
    # (then all particles of that type are given that constant mass) or by defining the mass of each particle 
    if (header['mass'].sum() == 0.):
      data['mass'] = readu(f, precision)
      data['u_therm'] = readu(f, precision)
    
    f.close()
  
    return data, header

def read_snapshot(filename):
    """ Reads a binary snapshot file """
    f = open(filename, mode='rb')
    print ("Loading file %s" % filename)

    data = {} # dictionary to hold data

    # read the header
    header = read_header(f, header_names, header_sizes)

    nparts = header['num_particles']
    masses = header['mass']
    print ('Particles', nparts)
    print ('Masses', masses)

    n_gas = nparts[0]
    n_DM = nparts[1]
    n_disc = nparts[2]
    n_bulge = nparts[3]
    
    n_sinks = nparts[5]
    if io_flags['mc_tracer']:
        n_tracer = nparts[2]
        if n_tracer != 0:
            print ('Tracer particles', n_tracer)
        if n_sinks != 0:
            print ('Sink particles', n_sinks)
    
    total = n_gas + n_sinks + n_DM + n_disc + n_bulge

    print ('Gas particles', n_gas)

    print ('Time = ', header['time'])

    if header['flag_doubleprecision']:
        precision = float64
        print ('Precision: Double')
    else:
        precision = float32
        print ('Precision: Float')
    
    """Now starts the reading!"""

    data['pos'] = readu(f, precision, 3).reshape((total, 3))
    data['vel'] = readu(f, precision, 3).reshape((total, 3))
    data['id'] = readIDs(f, total)
    data['mass'] = readu(f, precision)
    
    data['u_therm'] = readu(f, precision)
    data['rho'] = readu(f, precision)
    
    if io_flags['mc_tracer']:
      data['numtrace'] = readu(f, uint32)
      data['tracerid'] = readIDs(f, n_tracer)
      data['parentid'] = readIDs(f, n_tracer)

    if io_flags['potential']:
      data['potential'] = readu(f, float32)
    
    if io_flags['time_steps']:
      data['tsteps'] = readu(f, precision)
      print(data['tsteps'][0:10])
    
    if io_flags['sgchem']:
      
      if io_flags['variable_metallicity']:
        data['abundz'] = readu(f, precision, 4).reshape((n_gas, 4))
      
      if io_flags['sgchem_NL99']:
        data['chem'] = readu(f, precision, 9).reshape((n_gas, 9))
      else:
        data['chem'] = readu(f, precision, 3).reshape((n_gas, 3))
      
      data['tdust'] = readu(f, precision)
      
      if io_flags['variable_metallicity']:
        data['dtgratio'] = readu(f, precision)

    f.close()
    return data, header

def read_snapshot_B(filename):
    """ Reads a binary snapshot file """
    f = open(filename, mode='rb')
    print ("Loading file %s" % filename)

    data = {} # dictionary to hold data

    # read the header
    header = read_header(f, header_names, header_sizes)

    print (header.keys())
    
    nparts = header['num_particles']
    masses = header['mass']
    print ('Particles', nparts)
    print ('Masses', masses)

    n_gas = nparts[0]
    n_sinks = nparts[5]
    n_tracer = nparts[2]
    
    #total = n_gas + n_sinks
    total = nparts[0]+nparts[1]+nparts[2]+nparts[3]+nparts[4]+nparts[5]

    print ('Gas particles', n_gas)

    if n_tracer != 0:
        print ('Tracer particles', n_tracer)
    if n_sinks != 0:
        print ('Sink particles', n_sinks)

    print ('Time = ', header['time'])

    if header['flag_doubleprecision']:
        precision = float64
        print ('Precision: Double')
    else:
        precision = float32
        print ('Precision: Float')
    
    """Now starts the reading!"""

    data['pos'] = readu(f, precision, 3).reshape((total, 3))
    print ('data[pos] =', data['pos'])
    data['vel'] = readu(f, precision, 3).reshape((total, 3))
    data['id'] = readIDs(f, total)
    data['mass'] = readu(f, precision)
    
    data['u_therm'] = readu(f, precision)
    data['rho'] = readu(f, precision)
    print ('Rho =', np.min(data['rho']),np.max(data['rho']))
    
    if io_flags['time_steps']:
      data['tsteps'] = readu(f, precision)
      print ('Tsteps: ', data['tsteps'].min(),np.median(data['tsteps']),data['tsteps'].max())  
        
    if io_flags['potential']:
      data['potential'] = readu(f, precision)
      print ('potential: ', data['potential'].min(),np.median(data['potential']),data['potential'].max()) 
    
    if io_flags['MHD']:
      print('Reading Bfield')
      data['Bfield'] = readu(f, precision, 3).reshape((n_gas, 3))
      data['divB'] = readu(f, precision)
      print ('Bfield: ', data['Bfield'].min(),np.median(data['Bfield']),data['Bfield'].max())
      print ('divB: ', data['divB'].min(),np.median(data['divB']),data['divB'].max())
       
      try:
        data['divBAlt'] = readu(f, precision)
      except:
        print ('Not enough blocks to read in divBAlt')
    
    if io_flags['mc_tracer']:
      data['numtrace'] = readu(f, uint32)
      data['tracerid'] = readIDs(f, n_tracer)
      data['parentid'] = readIDs(f, n_tracer)    
    
    if io_flags['sgchem']:
        
      if io_flags['variable_metallicity']:
        data['abundz'] = readu(f, precision, 4).reshape((n_gas, 4))
      
      if io_flags['sgchem_NL99']:
        data['chem'] = readu(f, precision, 9).reshape((n_gas, 9))
      else:
        data['chem'] = readu(f, precision, 3).reshape((n_gas, 3))
        print ('Chem: ', data['chem'].min(),np.median(data['chem']),data['chem'].max())
      
      data['tdust'] = readu(f, precision)
      print ('Tdust: ', data['tdust'].min(),np.median(data['tdust']),data['tdust'].max())
        
      if io_flags['variable_metallicity']:
        data['dtgratio'] = readu(f, precision)

    f.close()
    return data, header


### FIXED FUNCTION SO THAT IT CAN BE USED TO READ BOTH DATA SETS WITH OR WITHOUT SINKS 

def read_snapshot_hdf5(filename):

    header ={}
    data = {}
    sink_data = {}
    f = h5py.File(filename,'r')

    ######## get header and data ########
    for item in f['Header'].attrs:
      header[item] = f['Header'].attrs[item]

    for item in f['PartType0'].keys():
      data[item] = f['PartType0'][item][:]
    
    ######## add sink data if present ########
    if "PartType5" in f.keys():
        for item in f['PartType5'].keys():
          sink_data[item] = f['PartType5'][item][:]

    output = [data, header]
    if "PartType5" in f.keys():
        output.append(sink_data)

    return output
'''



def read_snapshot_hdf5(filename):

    header ={}
    data = {}
    sink_data = {}
    f = h5py.File(filename,'r')

    ######## get header and data ########
    for item in f['Header'].attrs:
      header[item] = f['Header'].attrs[item]

    for item in f['PartType0'].keys():
      data[item] = f['PartType0'][item][:]
    
    ######## define precision ########
    #if header['flag_doubleprecision']:
    #    precision = float64
    #    print ('Precision: Double')
    #else:
    #    precision = float32
    #    print ('Precision: Float')  
        
    ######## read in flags ########
    if io_flags['time_steps']:
      data['tsteps'] = readu(f, int)
      print ('Tsteps: ', data['tsteps'].min(),np.median(data['tsteps']),data['tsteps'].max())
    
    #n_gas = nparts[0]
    #n_sinks = nparts[5]
    #n_tracer = nparts[2]
    
    #total = n_gas + n_sinks
    #total = nparts[0]+nparts[1]+nparts[2]+nparts[3]+nparts[4]+nparts[5]

    #print ('Gas particles', n_gas)

    #if n_tracer != 0:
    #    print ('Tracer particles', n_tracer)
    #if n_sinks != 0:
    #    print ('Sink particles', n_sinks)
    if "PartType5" in f.keys():
        for item in f['PartType5'].keys():
          sink_data[item] = f['PartType5'][item][:]

    #for item in f['PartType1'].keys():
    #  data[item] = f['PartType1'][item][:]

    #for item in f['PartType2'].keys():
    #  data[item] = f['PartType2'][item][:]

    #for item in f['PartType3'].keys():
    #  data[item] = f['PartType3'][item][:]
    
    #, sink_data
    output = [data, header]
    if "PartType5" in f.keys():
        output.append(sink_data)

    return output
'''


def read_snapshot_old(filename):
    """ Reads a binary snapshot file """
    f = open(filename, mode='rb')
    print ("Loading file %s" % filename)

    data = {} # dictionary to hold data

    # read the header
    header = read_header(f, header_names, header_sizes)

    nparts = header['num_particles']
    masses = header['mass']
    print ('Particles', nparts)
    print ('Masses', masses)

    n_gas = nparts[0]
    n_sinks = nparts[5]
    n_tracer = nparts[2]
    total = n_gas + n_sinks

    print ('Gas particles', n_gas)

    if n_tracer != 0:
        print ('Tracer particles', n_tracer)
    if n_sinks != 0:
        print ('Sink particles', n_sinks)

    print ('Time = ', header['time'])

    if header['flag_doubleprecision']:
        precision = float64
        print ('Precision: Double')
    else:
        precision = float32
        print ('Precision: Float')
    
    """Now starts the reading!"""

    data['pos'] = readu(f, precision, 3).reshape((total, 3))
    data['vel'] = readu(f, precision, 3).reshape((total, 3))
    data['id'] = readIDs(f, total)
    data['mass'] = readu(f, precision)
    
    data['u_therm'] = readu(f, precision)
    data['rho'] = readu(f, precision)
    
    print('read density')
    
    if io_flags['mc_tracer']:
      data['numtrace'] = readu(f, uint32)
      data['tracerid'] = readIDs(f, n_tracer)
      data['parentid'] = readIDs(f, n_tracer)
    
    print('Volume tag',io_flags['volume'])
    
    if io_flags['volume']:
      print('going to read volume')
      dummyread = readu(f, precision)
      dummyread = readu(f, precision)
      print('read volume')
    
    if io_flags['time_steps']:
      data['tsteps'] = readu(f, precision)
   
    
    if io_flags['sgchem']:
      
      if io_flags['variable_metallicity']:
        data['abundz'] = readu(f, precision, 4).reshape((n_gas, 4))
      
      if io_flags['sgchem_NL99']:
        data['chem'] = readu(f, precision, 9).reshape((n_gas, 9))
      else:
        data['chem'] = readu(f, precision, 3).reshape((n_gas, 3))
      
      data['tdust'] = readu(f, precision)
      
      if io_flags['variable_metallicity']:
        data['dtgratio'] = readu(f, precision)

    f.close()
    return data, header



def read_image(filename):
    f = open(filename, mode='rb')
    print ("Loading file %s" % filename)

    npix_x = np.fromfile(f, uint32, 1)
    npix_y = np.fromfile(f, uint32, 1)
    
    npix_x=int(npix_x)
    npix_y=int(npix_y)

    print (npix_x, npix_y)
    arepo_image = np.fromfile(f, float32, npix_x*npix_y).reshape((int(npix_x), int(npix_y))).T
#    arepo_image = np.rot90(arepo_image)
    f.close()
    return arepo_image

def read_vector_image(filename):
    
    f = open(filename, mode='rb')
    print ("Loading file %s" % filename)

    npix_x = np.fromfile(f, uint32, 1)
    npix_y = np.fromfile(f, uint32, 1)
    
    arepo_image = np.fromfile(f, float32, npix_x[0]*npix_y[0]*3).reshape(npix_x[0],npix_y[0],3)
    #arepo_image = np.rot90(arepo_image)
    
    f.close()
    return arepo_image


def read_grid(filename):
    
    f = open(filename, mode='rb')
    print ("Loading file %s" % filename)
    
    npix_x = np.fromfile(f, uint32, 1)
    npix_y = np.fromfile(f, uint32, 1)
    npix_z = np.fromfile(f, uint32, 1)
    print (npix_x, npix_y,npix_z)
    print (npix_x[0])
    
    # This assumes one value (float32) per grid point!!
    arepo_grid = np.fromfile(f, float32, npix_x[0]*npix_y[0]*npix_z[0]).reshape((npix_z[0], npix_y[0],npix_x[0]))
    
    f.close()
    
    return arepo_grid

#adding a function to read magnetic or velocity grids with 3 components
def read_vector_grid(filename):
    
    f = open(filename, mode='rb')
    print ("Loading file %s" % filename)
    
    npix_x = np.fromfile(f, uint32, 1)
    npix_y = np.fromfile(f, uint32, 1)
    npix_z = np.fromfile(f, uint32, 1)
    print (npix_x, npix_y,npix_z)
    print (npix_x[0])
    
    # Assuming three values (float32) per grid point
    arepo_grid = np.fromfile(f, dtype=np.float32, count=npix_x[0]*npix_y[0]*npix_z[0]*3).reshape((npix_z[0], npix_y[0], npix_x[0], 3))
    
    f.close()
    
    return arepo_grid
    
def read_sink_evolution_file(filename):
    f = open(filename, mode='rb')
    print ("Loading file %s" %filename)

    time = np.fromfile(f, float64, 1)

    print ("Time = ", time)

    NSinks = np.fromfile(f, uint32, 1)

    print ("Number of sink particles = ", NSinks)

    SinkP = {'Pos':[],
             'Vel':[],
             'Acc':[],
             'Mass':[],
	     #'MassOld':[],
             'AccretionRadius':[],
             'FormationMass':[],
             'FormationTime':[],
             'ID':[],
             'HomeTask':[],
             'Index':[],
             'FormationOrder':[],
             'N_sne':[],
             'StellarMass':[],
             'explosion_time':[],
             'MassStillToConvert':[],
             'AccretionTime':[],}

    for i in np.arange(NSinks):

      app = SinkP['Pos']
      app.append(np.fromfile(f, float64, 3).reshape(1,3))
      SinkP['Pos'] = app

      app = SinkP['Vel']
      app.append(np.fromfile(f, float64, 3).reshape(1,3))
      SinkP['Vel'] = app

      app = SinkP['Acc']
      app.append(np.fromfile(f, float64, 3).reshape(1,3))
      SinkP['Acc'] = app

      app = SinkP['Mass']
      app.append(np.fromfile(f, float64, 1))
      SinkP['Mass'] = app

      #app = SinkP['MassOld']
      #app.append(np.fromfile(f, float64, 1))
      #SinkP['MassOld'] = app

      app = SinkP['AccretionRadius']
      app.append(np.fromfile(f, float64, 1))
      SinkP['AccretionRadius'] = app

      app = SinkP['FormationMass']
      app.append(np.fromfile(f, float64, 1))
      SinkP['FormationMass'] = app

      app = SinkP['FormationTime']
      app.append(np.fromfile(f, float64, 1))
      SinkP['FormationTime'] = app

      app = SinkP['ID']
      app.append(np.fromfile(f, uint64, 1))
      SinkP['ID'] = app

      app = SinkP['HomeTask']
      app.append(np.fromfile(f, uint32, 1))
      SinkP['HomeTask'] = app


      app = SinkP['Index']
      app.append(np.fromfile(f, uint32, 1))
      SinkP['Index'] = app

      app = SinkP['FormationOrder']
      app.append(np.fromfile(f, uint32, 1))
      SinkP['FormationOrder'] = app

      app = SinkP['N_sne']
      app.append(np.fromfile(f, uint32, 1))
      SinkP['N_sne'] = app

      app = SinkP['StellarMass']
      app.append(np.fromfile(f, float64, 1))
      SinkP['StellarMass'] = app

      app = SinkP['explosion_time']
      app.append(np.fromfile(f, float64, 800))
      SinkP['explosion_time'] = app
    
      app = SinkP['MassStillToConvert']
      app.append(np.fromfile(f, float64, 50))
      SinkP['MassStillToConvert'] = app

      app = SinkP['AccretionTime']
      app.append(np.fromfile(f, float64, 50))
      SinkP['AccretionTime'] = app

#    return time, SinkP
    return SinkP

def return_gas_particles(data, header):
    ngas = header['num_particles'][0]
    
    data['pos'] = data['pos'][0:ngas,:]
    data['vel'] = data['vel'][0:ngas,:]
    data['id'] = data['id'][0:ngas]
    data['mass'] = data['mass'][0:ngas]

    return data


####### NEW FUNCTION (27/09/23) from Jonathan to read sink_snap files #######

def read_dump(file, spin=False, bh=False, hm=False, rcirc=False):
    f = open(file, 'rb')
    
    time = np.fromfile(f, np.float64, 1)
    NSinksAllTasks = np.fromfile(f, np.uint32, 1)
    sinks = {}
    
    fields = ['Pos', 'Vel', 'Accel', 'Mass', 'FormationMass', 'FormationTime', 
              'ID', 'HomeTask', 'Index', 'FormationOrder']
    
    if spin == True:
        fields += ['AngularMomentum']
        
    if bh == True:
        fields += ['BlackHole'] 
        if hm == True:
            fields += ['BlackHoleHotMode']
        fields += ['BlackHoleAccRadius', 'BlackHoleMass', 'BlackHoleDiskMass', 'BlackHoleReservoir', 'BlackHoleSinkAccRate', 'CellsTotalMassBuffer']
        if rcirc == True:
            fields += ['BlackHoleCircRadius']
    
    for i in fields:
        sinks[i] = []
        
    for i in range(NSinksAllTasks[0]):
        struct = pycstruct.StructDef(alignment = 8)
        struct.add('float64', 'Pos', shape=3)
        struct.add('float64', 'Vel', shape=3)
        struct.add('float64', 'Accel', shape=3)
        struct.add('float64', 'Mass')
        struct.add('float64', 'FormationMass')
        struct.add('float64', 'FormationTime')
        struct.add('uint64', 'ID')
        struct.add('uint32', 'HomeTask')
        struct.add('uint32', 'Index')
        struct.add('uint32', 'FormationOrder')
        if spin == True:
            struct.add('float64', 'AngularMomentum', shape=3)
        if bh == True:
            struct.add('uint32', 'BlackHole')
            if hm == True:
                struct.add('uint32', 'BlackHoleHotMode')
            struct.add('float64', 'BlackHoleAccRadius')
            struct.add('float64', 'BlackHoleMass')
            struct.add('float64', 'BlackHoleDiskMass')
            struct.add('float64', 'BlackHoleReservoir')
            struct.add('float64', 'BlackHoleSinkAccRate')
            struct.add('float64', 'CellsTotalMassBuffer')
            if rcirc == True:
                struct.add('float64', 'BlackHoleCircRadius')        
        
        inbytes = f.read(struct.size())
        data = struct.deserialize(inbytes)
        for field in fields:
            sinks[field] += [data[field]]
    
    for field in fields:
        sinks[field] = np.array(sinks[field])
    
    f.close()
    
    return time, NSinksAllTasks, sinks

############# FUNCTION FROM ROBIN, MAY NEED BIT PADDING #############

def read_sink_snap(filename):
    MAXSNE = 800
    MAXACCRETIONEVENTS = 50
    
    f = open(filename, mode='rb')
    Time = np.fromfile(f, np.float64, 1)
    print(Time)
    NSinksAllTasks = np.fromfile(f, np.uint32, 1)
    print(NSinksAllTasks)    
    
    fields = ['pos', 
              'vel', 
              'accel',
              'mass',
              'formationMass',
              'formationTime',
              'ID',
              'HomeTask',
              'Index',
              'FormationOrder',
              'N_SNe',
              'StellarMass',
              'ExplosionTime',
              'MassStillToConvert',
              'AccretionTime'] #'SFeff'    
    
    data = {}
    
    for field in fields:
        data[field] = []    
        
    for i in np.arange(NSinksAllTasks):
        data['pos']                += [np.fromfile(f, np.float64, 3)]
        data['vel']                += [np.fromfile(f, np.float64, 3)]
        data['accel']              += [np.fromfile(f, np.float64, 3)]
        data['mass']               += [np.fromfile(f, np.float64, 1)]
        data['formationMass']      += [np.fromfile(f, np.float64, 1)]
        data['formationTime']      += [np.fromfile(f, np.float64, 1)]
        data['ID']                 += [np.fromfile(f, np.uint64, 1)]
        data['HomeTask']           += [np.fromfile(f, np.uint32, 1)]
        data['Index']              += [np.fromfile(f, np.uint32, 1)]
        data['FormationOrder']     += [np.fromfile(f, np.uint32, 1)]
        data['N_SNe']              += [np.fromfile(f, np.uint32, 1)]
        data['StellarMass']        += [np.fromfile(f, np.float64, 1)]
        data['ExplosionTime']      += [np.fromfile(f, np.float64, MAXSNE)]
        data['MassStillToConvert'] += [np.fromfile(f, np.float64, MAXACCRETIONEVENTS)]
        data['AccretionTime']      += [np.fromfile(f, np.float64, MAXACCRETIONEVENTS)]    
        
        #data['SFeff']      += [np.fromfile(f, np.float64, ??)
        
    f.close()
    
    for field in fields:
        data[field] = np.array(data[field])
        
    return data, Time, NSinksAllTasks