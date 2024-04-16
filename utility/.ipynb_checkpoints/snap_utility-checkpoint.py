from arepy.read_write import binary_read as rsnap
from arepy.utility import cgs_constants as cgs

#from ut_utility import *

import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt

def find_square_region(pos, center, delta):

    ind = np.where( (abs(pos[:,0] - center[0]) < delta[0]) & \
                    (abs(pos[:,1] - center[1]) < delta[1]) & \
                    (abs(pos[:,2] - center[2]) < delta[2]) )
    return np.array(ind).reshape(np.size(ind))

def find_region(pos, center, radius):

    ind = np.where( ((pos[:,0] - center[0])**2 + \
                     (pos[:,1] - center[1])**2 + \
                     (pos[:,2] - center[2])**2) \
                     < radius**2 )

    return ind

def read_all_proj_images(path, number):
    ABHE = 0.1
    CarbAbund = 1.4e-4

    filenum = str(number).zfill(3)
    ColumnDensity = rsnap.read_image(path + 'density_proj_' + filenum)
    xHp = rsnap.read_image(path + 'xHP_proj_' + filenum) 
    xH2 = rsnap.read_image(path + 'xH2_proj_' + filenum) 
    xCO = rsnap.read_image(path + 'xCO_proj_' + filenum) 
  
    xCII = CarbAbund - xCO

    NH = (ColumnDensity * rsnap.arepoColumnDensity) / ((1.+4.*ABHE)*cgs.mp)
    
    Ndensity = NH * (1. + ABHE - xH2 + xHp)
    NCO = NH * xCO
    NH2 = NH * xH2
    NHp = NH * xHp
    NHI = NH * (1. - xHp - 2.*xH2)
    NCII = NH * xCII
    
    return Ndensity, NCO, NH2, NHp, NHI, NCII

def find_temperature(data):
    ABHE=0.1
    yn = data['rho']*rsnap.arepoDensity / ((1. + 4.0 * ABHE) * cgs.mp)
    energy = data['u_therm'] * data['rho'] * rsnap.arepoEnergy/rsnap.arepoLength/rsnap.arepoLength/rsnap.arepoLength
  
    yntot = (1. + ABHE - data['chem'][:,0] + data['chem'][:,1]) * yn
  
    temp = 2. * energy / (3. * yntot * cgs.kb)
  
    return temp

def find_cart_cyl_coord( pos ):
    halfbox = pos[:,0].max() / 2.
    halfboy = pos[:,1].max() / 2.
    halfboz = pos[:,2].max() / 2.

    x = pos[:,0] - halfbox
    y = pos[:,1] - halfboy
    z = pos[:,2] - halfboz
    R = np.sqrt(x**2. + y**2.)
    theta = np.arctan2(y, x)

    return x, y, z, R, theta

def find_xyzRtheta( pos, halfbox ):

    x = pos[:,0] - halfbox
    y = pos[:,1] - halfbox
    z = pos[:,2] - halfbox
    R = np.sqrt(x**2. + y**2.)
    theta = np.arctan2(y, x)

    return x, y, z, R, theta

def rotate(x,y,theta):
    # counter clockwise rotation
    xprime = x*cos(theta) - y*sin(theta)
    yprime = x*sin(theta) + y*cos(theta)
    return xprime, yprime

def find_number_density(data):
    ABHE = 0.1

    xH2, xHp, xCO = data['chem'].T
    
    nHtot = data['rho']*rsnap.arepoDensity/((1. + 4.*ABHE)*cgs.mp)
    
    nHp = xHp*nHtot
    nH2 = xH2*nHtot
    nCO = xCO*nHtot
    nHI = (1.0 - xHp - 2.0*xH2)*nHtot
    nHe = ABHE*nHtot

    n = nHtot * (1. + ABHE - xH2 + xHp)
    
    return n, nHI, nH2, nHp, nCO, nHe

def find_chem_masses(data):

    vol = data['mass'] / data['rho']

    n, nHI, nH2, nHp, nCO, nHe = find_number_density(data)

    massesH2 = (nH2*(2*cgs.mp)/rsnap.arepoDensity) * vol
    massesHI = (nHI*cgs.mp/rsnap.arepoDensity)     * vol
    massesHp = (nHp*cgs.mp/rsnap.arepoDensity)     * vol
    massesCO = (nCO*28*cgs.mp/rsnap.arepoDensity)  * vol
    massesHe = (nHe*4*cgs.mp/rsnap.arepoDensity)   * vol
    
    return massesHI, massesH2, massesHp, massesCO, massesHe

def return_type(data, header, returntype):    #for hdf5 files
    return_data = {}

    nn = np.cumsum(header['NumPart_Total'])
    nn = np.insert(nn,0,0)
    n1 = nn[returntype]
    n2 = nn[returntype+1]

    if (header['MassTable'][returntype] != 0):
      return_data['MassTable'] = header['MassTable'][returntype]
    else:
      nn1 = header['NumPart_Total'][0:returntype][header['MassTable'][0:returntype] == 0].sum()
      nn2 = int(nn1 + header['NumPart_Total'][returntype])# +1)
      print (nn1, nn2)
      return_data['Masses'] = data['Masses'][nn1:nn2]
    return_data['Coordinates'] = data['Coordinates'][n1:n2,:]
    return_data['Velocities'] = data['Velocities'][n1:n2,:]
    return_data['ParticleIDs'] = data['ParticleIDs'][n1:n2]

    return return_data

  
def return_type1(data, header, returntype):      #for binary files
    return_data = {}

    nn = np.cumsum(header['num_particles'])
    nn = np.insert(nn,0,0)
    n1 = nn[returntype]
    n2 = nn[returntype+1]

    if (header['mass'][returntype] != 0):
      return_data['mass'] = header['mass'][returntype]
    else:
      nn1 = header['num_particles'][0:returntype][header['mass'][0:returntype] == 0].sum()
      nn2 = int(nn1 + header['num_particles'][returntype])# +1)
      print (nn1, nn2)
      return_data['mass'] = data['mass'][nn1:nn2]
    return_data['pos'] = data['pos'][n1:n2,:]
    return_data['vel'] = data['vel'][n1:n2,:]
    return_data['id'] = data['id'][n1:n2]

    return return_data
