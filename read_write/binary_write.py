#from io_utility import *
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

io_flags = {'mc_tracer'           : True,\
            'time_steps'          : True,\
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

########## necessary WRITE functions #########

def writeu(f, data):
    """ Write a numpy array to the unformatted fortran file f """
    data_size = data.size * data.dtype.itemsize

    data_size = np.array(data_size, dtype=uint32)

    data_size.tofile(f)

    data.tofile(f)

    data_size.tofile(f)
    
####### binary write functions #########

def write_snapshot(filename, header, data):
    """ Write a binary snapshot file (unformatted fortran) """
    # do some checks on the header
    for name, size in zip(header_names, header_sizes):
      if name not in header:
        print('Missing %s in header file' % name)
        raise Exception('Missing %s in header file' % name)

      if np.array(header[name]).size != size[1]:
        msg = 'Header %s should contain %d elements, %d found' % \
              (name, size[1], np.array(header[name]).size)
        print(msg)
        raise Exception(msg)

    nparts = header['num_particles']

    #should we write in single or double precision
    if header['flag_doubleprecision']==0:
        precision = float32
    else:
        precision = float64

    # ok so far, lets write the file
    f = open(filename, 'wb')

    # write the header
    np.array(256, uint32).tofile(f)

    for name, size in zip(header_names, header_sizes):
        np.array(header[name]).astype(size[0]).tofile(f)
    # final block
    np.array(256, uint32).tofile(f)

    # write the data
    writeu(f, data['pos'].astype(precision))
    writeu(f, data['vel'].astype(precision))
    writeu(f, data['id'])
    writeu(f, data['mass'].astype(precision))

    writeu(f,data['u_therm'].astype(precision) )
    writeu(f, data['rho'].astype(precision))
    
    if io_flags['mc_tracer']:
      writeu(f, data['numtrace'])
      writeu(f, data['tracerid'])
      writeu(f, data['parentid'])
    
    if io_flags['time_steps']:
      writeu(f, data['tsteps'].astype(precision))
    
    if io_flags['sgchem']:

      if io_flags['variable_metallicity']:
        writeu(f, data['abundz'].astype(precision))
      
      writeu(f, data['chem'].astype(precision))
      writeu(f, data['tdust'].astype(precision))
      
      if io_flags['variable_metallicity']:
        writeu(f, data['dtgratio'].astype(precision))

    # all done!
    f.close()

def write_snapshot_B(filename, header, data):
    """ Write a binary snapshot file (unformatted fortran) with magnetic fields"""
    # do some checks on the header
    for name, size in zip(header_names, header_sizes):
      if name not in header:
        print('Missing %s in header file' % name)
        raise Exception('Missing %s in header file' % name)

      if np.array(header[name]).size != size[1]:
        msg = 'Header %s should contain %d elements, %d found' % \
              (name, size[1], np.array(header[name]).size)
        print(msg)
        raise Exception(msg)

    nparts = header['num_particles']

    #should we write in single or double precision
    if header['flag_doubleprecision']==0:
        precision = float32
    else:
        precision = float64

    # ok so far, lets write the file
    f = open(filename, 'wb')

    # write the header
    np.array(256, uint32).tofile(f)

    for name, size in zip(header_names, header_sizes):
        np.array(header[name]).astype(size[0]).tofile(f)
    # final block
    np.array(256, uint32).tofile(f)

    # write the data
    writeu(f, data['pos'].astype(precision))
    writeu(f, data['vel'].astype(precision))
    writeu(f, data['id'])
    writeu(f, data['mass'].astype(precision))

    writeu(f,data['u_therm'].astype(precision) )
    writeu(f, data['rho'].astype(precision))
    
    if io_flags['time_steps']:
      writeu(f, data['tsteps'].astype(precision))
    
    if io_flags['potential']:
      writeu(f, data['potential'].astype(precision))
    
    if io_flags['MHD']:
      writeu(f, data['Bfield'].astype(precision))
      writeu(f, data['divB'].astype(precision))
      try: 
        writeu(f, data['divBAlt'].astype(precision))
      except:
        print ('No divBAlt to write, skipping this entry')
    
    if io_flags['mc_tracer']:
      writeu(f, data['numtrace'])
      writeu(f, data['tracerid'])
      writeu(f, data['parentid'])
    
    if io_flags['sgchem']:
        
      if io_flags['variable_metallicity']:
        writeu(f, data['abundz'].astype(precision))
      
      writeu(f, data['chem'].astype(precision))
      writeu(f, data['tdust'].astype(precision))

    if io_flags['variable_metallicity']:
      writeu(f, data['dtgratio'].astype(precision))
    
    # all done!
    f.close()
    
    
def write_snapshot_hdf5(filename,new_MHD): #TEMPORARY MASS CHANGE ADAPTATION

    # ok so far, lets write the file
    hf = h5py.File(filename, 'r+') #loads up file, allows to read and write

    header = {}
    data = {}

    #test the file structure
    for item in hf['Header'].attrs:
        header[item] = hf['Header'].attrs[item]

    print (header.keys())

    for item in hf['PartType0'].keys():
        data[item] = hf['PartType0'][item][:]

    print (data.keys())

    #lets add the new data - this can be changed to whatever data you want
    print (new_MHD)
    hf['PartType0']['MagneticField'].write_direct(new_MHD) #PartType0 = gas (this bit tells it to re-write the magnetic field part of the array)
    print (data['MagneticField'])
    # all done!
    hf.close()
    
'''  
def write_snapshot_hdf5(filename,new_MHD):

    # ok so far, lets write the file
    hf = h5py.File(filename, 'r+') #loads up file, allows to read and write

    header = {}
    data = {}

    #test the file structure
    for item in hf['Header'].attrs:
        header[item] = hf['Header'].attrs[item]

    print header.keys()

    for item in hf['PartType0'].keys():
        data[item] = hf['PartType0'][item][:]

    print data.keys()

    #lets add the new data - this can be changed to whatever data you want
    print new_MHD
    hf['PartType0']['MagneticField'].write_direct(new_MHD) #PartType0 = gas (this bit tells it to re-write the magnetic field part of the array)
    print data['MagneticField']
    # all done!
    hf.close()
''' 
    
def write_snapshot_hdf5_all(filename,ID,pos,rho,chem,temp,mass,vel,Bfield,divBalt,divB,Fieldpsi,timestep,potentialpeak,potential,dusttemp,Numpart_new):

    Numpart_new = Numpart_new
    print (Numpart_new)

    # ok so far, lets write the file
    hf = h5py.File(filename, 'r+')

    header = {}
    data = {}

    #test the file structure
    for item in hf['Header'].attrs:
        header[item] = hf['Header'].attrs[item]

    print (header.keys())

    for item in hf['PartType0'].keys():
        data[item] = hf['PartType0'][item][:]

    print (data.keys())

    #lets add the new data - this can be changed to whatever data you want 
    #creates new datasets in PartType0 group if changing size of arrays

    #first need to delete the old datasets
    del hf['PartType0']['ParticleIDs']
    del hf['PartType0']['Coordinates']
    del hf['PartType0']['Density']
    del hf['PartType0']['ChemicalAbundances']
    del hf['PartType0']['InternalEnergy']
    del hf['PartType0']['Masses']
    del hf['PartType0']['Velocities']
    del hf['PartType0']['MagneticField']
    del hf['PartType0']['MagneticFieldDivergenceAlternative']
    del hf['PartType0']['MagneticFieldDivergence']
    del hf['PartType0']['MagneticFieldPsi']
    del hf['PartType0']['TimeStep']    
    del hf['PartType0']['PotentialPeak']    
    del hf['PartType0']['Potential']
    del hf['PartType0']['DustTemperature']
   
    hf.create_dataset("PartType0/ParticleIDs", data=ID)
    hf.create_dataset("PartType0/Coordinates", data=pos)
    hf.create_dataset("PartType0/Density", data=rho)
    hf.create_dataset("PartType0/ChemicalAbundances", data=chem)
    hf.create_dataset("PartType0/InternalEnergy", data=temp)
    hf.create_dataset("PartType0/Masses", data=mass)
    hf.create_dataset("PartType0/Velocities", data=vel)
    hf.create_dataset("PartType0/MagneticField", data=Bfield)
    hf.create_dataset("PartType0/MagneticFieldDivergenceAlternative", data=divBalt)
    hf.create_dataset("PartType0/MagneticFieldDivergence", data=divB)
    hf.create_dataset("PartType0/MagneticFieldPsi", data=Fieldpsi)
    hf.create_dataset("PartType0/TimeStep", data=timestep)
    hf.create_dataset("PartType0/PotentialPeak", data=potentialpeak)
    hf.create_dataset("PartType0/Potential", data=potential)
    hf.create_dataset("PartType0/DustTemperature", data=dusttemp)

    #use this if just editing the data but not array size

    #hf['PartType0']['ParticleIDs'].write_direct(ID)
    #hf['PartType0']['Coordinates'].write_direct(pos)
    #hf['PartType0']['Density'].write_direct(rho)
    #hf['PartType0']['ChemicalAbundances'].write_direct(chem)
    #hf['PartType0']['InternalEnergy'].write_direct(temp)
    #hf['PartType0']['Masses'].write_direct(mass)
    #hf['PartType0']['Velocities'].write_direct(vel)
    #hf['PartType0']['MagneticField'].write_direct(Bfield)
    #hf['PartType0']['MagneticFieldDivergenceAlternative'].write_direct(divBalt)
    #hf['PartType0']['MagneticFieldDivergence'].write_direct(divB)
    #hf['PartType0']['MagneticFieldPsi'].write_direct(Fieldpsi)
    #hf['PartType0']['TimeStep'].write_direct(timestep)
    #hf['PartType0']['PotentialPeak'].write_direct(potentialpeak)
    #hf['PartType0']['Potential'].write_direct(potential)
    #hf['PartType0']['DustTemperature'].write_direct(dusttemp)


    print (data['MagneticField'])

    #now edit the header

    hf['Header'].attrs.modify('NumPart_Total', Numpart_new)
    hf['Header'].attrs.modify('NumPart_ThisFile', Numpart_new)

    # all done!
    hf.close()


    
def write_IC(pos,vel,mass,u_therm,filename):
    # write initial conditions for arepo given:
    # pos = positions
    # vel = velocities 
    # mass = masses
    # u_therm = thermal energy per unit mass
    npts = mass.size
    header = {'num_particles': np.array([npts,       0,       0,       0,       0,       0]).astype(uint32),
             'mass': np.array([ 0.,  0.,  0.,  0.,  0.,  0.]).astype(float64),
             'time': np.array([0.]).astype(float64),
             'redshift': np.array([0.]).astype(float64),
             'flag_sfr': np.array([0]).astype(uint32),
             'flag_feedback': np.array([0]).astype(uint32),
             'num_particles_total': np.array([npts,       0,       0,       0,       0,       0]).astype(uint32),
             'flag_cooling': np.array([0]).astype(uint32),
             'num_files': np.array([1]).astype(uint32),
             'boxsize': np.array([ 0.]).astype(float64),
             'omega0': np.array([ 0.]).astype(float64),
             'omegaLambda': np.array([ 0.]).astype(float64),
             'hubble0': np.array([ 1.]).astype(float64),
             'flag_stellarage': np.array([0]).astype(uint32),
             'flag_metals': np.array([0]).astype(uint32),
             'npartTotalHighWord': np.array([0, 0, 0, 0, 0, 0]).astype(uint32),
             'flag_entropy_instead_u': np.array([0]).astype(uint32),
             'flag_doubleprecision': np.array([0]).astype(uint32),
             'flag_lpt_ics': np.array([0]).astype(uint32),
             'lpt_scalingfactor': np.array([0]).astype(float32),
             'flag_tracer_field': np.array([0]).astype(uint32),
             'composition_vector_length': np.array([0]).astype(uint32),
             'buffer': np.empty([40]).astype(np.uint8)}

    ID = np.arange(npts) + 1
    f = open(filename, 'wb')
    precision = float32
    idprecision = np.int32
    idlong=np.int64
  
    np.array(256, uint32).tofile(f)
    for name, size in zip(header_names, header_sizes):
        np.array(header[name]).astype(size[0]).tofile(f)
    np.array(256, uint32).tofile(f)
    writeu(f, pos.astype(precision))
    writeu(f, vel.astype(precision))
    writeu(f, ID.astype(idprecision))
    writeu(f, mass.astype(precision))
    writeu(f, u_therm.astype(precision) )
    f.close()
    return
    
def write_IC_long(pos,vel,mass,u_therm,filename):
    # write initial conditions for arepo given:
    # pos = positions
    # vel = velocities 
    # mass = masses
    # u_therm = thermal energy per unit mass
    npts = mass.size
    header = {'num_particles': np.array([npts,       0,       0,       0,       0,       0]).astype(uint32),
             'mass': np.array([ 0.,  0.,  0.,  0.,  0.,  0.]).astype(float64),
             'time': np.array([0.]).astype(float64),
             'redshift': np.array([0.]).astype(float64),
             'flag_sfr': np.array([0]).astype(uint32),
             'flag_feedback': np.array([0]).astype(uint32),
             'num_particles_total': np.array([npts,       0,       0,       0,       0,       0]).astype(uint32),
             'flag_cooling': np.array([0]).astype(uint32),
             'num_files': np.array([1]).astype(uint32),
             'boxsize': np.array([ 0.]).astype(float64),
             'omega0': np.array([ 0.]).astype(float64),
             'omegaLambda': np.array([ 0.]).astype(float64),
             'hubble0': np.array([ 1.]).astype(float64),
             'flag_stellarage': np.array([0]).astype(uint32),
             'flag_metals': np.array([0]).astype(uint32),
             'npartTotalHighWord': np.array([0, 0, 0, 0, 0, 0]).astype(uint32),
             'flag_entropy_instead_u': np.array([0]).astype(uint32),
             'flag_doubleprecision': np.array([0]).astype(uint32),
             'flag_lpt_ics': np.array([0]).astype(uint32),
             'lpt_scalingfactor': np.array([0]).astype(float32),
             'flag_tracer_field': np.array([0]).astype(uint32),
             'composition_vector_length': np.array([0]).astype(uint32),
             'buffer': np.empty([40]).astype(np.uint8)}

    ID = np.arange(npts) + 1
    f = open(filename, 'wb')
    precision = float32
    idprecision = np.int32
    idlong=np.int64
  
    np.array(256, uint32).tofile(f)
    for name, size in zip(header_names, header_sizes):
        np.array(header[name]).astype(size[0]).tofile(f)
    np.array(256, uint32).tofile(f)
    writeu(f, pos.astype(precision))
    writeu(f, vel.astype(precision))
    writeu(f, ID.astype(idlong))
    writeu(f, mass.astype(precision))
    writeu(f, u_therm.astype(precision) )
    f.close()
    return


def write_IC_withtracers(pos,vel,ID,mass,u_therm,numtrace,tracerid,parentid,filename):
    # write initial conditions for arepo given:
    # pos = positions
    # vel = velocities 
    # mass = masses
    # u_therm = thermal energy per unit mass
    npts = mass.size
    ntrace = tracerid.size
    header = {'num_particles': np.array([npts,       0,       ntrace,       0,       0,       0]).astype(uint32),
             'mass': np.array([ 0.,  0.,  0.,  0.,  0.,  0.]).astype(float64),
             'time': np.array([0.]).astype(float64),
             'redshift': np.array([0.]).astype(float64),
             'flag_sfr': np.array([0]).astype(uint32),
             'flag_feedback': np.array([0]).astype(uint32),
             'num_particles_total': np.array([npts,       0,       ntrace,       0,       0,       0]).astype(uint32),
             'flag_cooling': np.array([0]).astype(uint32),
             'num_files': np.array([1]).astype(uint32),
             'boxsize': np.array([ 0.]).astype(float64),
             'omega0': np.array([ 0.]).astype(float64),
             'omegaLambda': np.array([ 0.]).astype(float64),
             'hubble0': np.array([ 1.]).astype(float64),
             'flag_stellarage': np.array([0]).astype(uint32),
             'flag_metals': np.array([0]).astype(uint32),
             'npartTotalHighWord': np.array([0, 0, 0, 0, 0, 0]).astype(uint32),
             'flag_entropy_instead_u': np.array([0]).astype(uint32),
             'flag_doubleprecision': np.array([0]).astype(uint32),
             'flag_lpt_ics': np.array([0]).astype(uint32),
             'lpt_scalingfactor': np.array([0]).astype(float32),
             'flag_tracer_field': np.array([1]).astype(uint32),
             'composition_vector_length': np.array([0]).astype(uint32),
             'buffer': np.empty([40]).astype(np.uint8)}

    f = open(filename, 'wb')
    precision = float32
    idprecision = np.int32
    idlong=np.int64
  
    np.array(256, uint32).tofile(f)
    for name, size in zip(header_names, header_sizes):
        np.array(header[name]).astype(size[0]).tofile(f)
    np.array(256, uint32).tofile(f)
    writeu(f, pos.astype(precision))
    writeu(f, vel.astype(precision))
    writeu(f, ID.astype(idlong))
    writeu(f, mass.astype(precision))
    writeu(f, u_therm.astype(precision) )
    writeu(f, numtrace.astype(idprecision))
    writeu(f, tracerid.astype(idlong))
    writeu(f, parentid.astype(idlong))
    
    #add the new tracers here
    
    f.close()
    return




