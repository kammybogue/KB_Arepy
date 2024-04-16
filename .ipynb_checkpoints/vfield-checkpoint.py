import numpy as np
from scipy.interpolate import griddata, NearestNDInterpolator
import matplotlib.pyplot as pl
from numpy import uint32, uint64, float64, float32, linspace, array, logspace, sin, cos, pi, arange, sqrt, arctan2, arccos
import matplotlib.colors as mc

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

def read_snapshot_file(filename):
  """ Reads a binary arepo file """
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
  f.close()
  return data, header

def rotate(x,y,theta):
  xprime = x*cos(theta) - y*sin(theta)
  yprime = x*sin(theta) + y*cos(theta)
  return xprime, yprime
  
# XYZ are cartesian coordinates centered at the Sun position. X axis points to the GC, Z axis up to north Galactic pole.
# xyz are cartesian coordinates centered at the GC with same orientations as XYZ.
# lbr are spherical coordinates centered at the Sun position (i.e., the usual Galactic coordinates).

# the following functions convert back and forth between the above coordinates

def XYZ2lbr(X,Y,Z):
  r = sqrt(X**2+Y**2+Z**2)
  l = arctan2(Y,X)
  b = pi/2 - arccos(Z/r)
  return l,b,r

def lbr2XYZ(l,b,r):
  X = r*sin(b+pi/2)*cos(l)
  Y = r*sin(b+pi/2)*sin(l)
  Z = r*cos(b+pi/2)
  return X,Y,Z

def xyz2XYZ(x,y,z):
  R0 = 80.0
  return x+R0,y,z

def XYZ2xyz(X,Y,Z):
  R0 = 80.0
  return X-R0,Y,Z

def xyz2lbr(x,y,z):
  X,Y,Z = xyz2XYZ(x,y,z)
  l,b,r = XYZ2lbr(X,Y,Z)
  return l,b,r

def lbr2xyz(l,b,r):
  X,Y,Z = lbr2XYZ(l,b,r)
  x,y,z = XYZ2xyz(X,Y,Z)
  return x,y,z

def vxyz2vlbr(x,y,z,vx,vy,vz):
  # see wiki "vector fields in spherical coords" for formulas
  X,Y,Z = xyz2XYZ(x,y,z)
  l,b,r = XYZ2lbr(X,Y,Z)
  rhat = [sin(b+pi/2)*cos(l),sin(b+pi/2)*sin(l),cos(b+pi/2)]
  bhat = [cos(b+pi/2)*cos(l),cos(b+pi/2)*sin(l),-sin(b+pi/2)] 
  lhat = [-sin(l),cos(l),0] 
  vr = vx*rhat[0] + vy*rhat[1] + vz*rhat[2]
  vb = vx*bhat[0] + vy*bhat[1] + vz*bhat[2]
  vl = vx*lhat[0] + vy*lhat[1]
  return l,b,r,vl,vb,vr
  
def Rotate_around_u(x,y,z,u,alpha):
  # rotates x,y,z by angle alpha around axis passing through the origin defined by unit vector u =[ux,uy,uz]
  # found formulas googling...hope it works
  R_matrix = array([[cos(alpha) + u[0]**2 * (1-cos(alpha)),
                     u[0] * u[1] * (1-cos(alpha)) - u[2] * sin(alpha),
                     u[0] * u[2] * (1 - cos(alpha)) + u[1] * sin(alpha)],
                    [u[0] * u[1] * (1-cos(alpha)) + u[2] * sin(alpha),
                     cos(alpha) + u[1]**2 * (1-cos(alpha)),
                     u[1] * u[2] * (1 - cos(alpha)) - u[0] * sin(alpha)],
                    [u[0] * u[2] * (1-cos(alpha)) - u[1] * sin(alpha),
                     u[1] * u[2] * (1-cos(alpha)) + u[0] * sin(alpha),
                     cos(alpha) + u[2]**2 * (1-cos(alpha))]])
  xr,yr,zr = R_matrix.dot([x,y,z])
  return xr,yr,zr

#internal arepo units in cgs
arepoLength = 3.0856e20
arepoMass = 1.991e33
arepoVel = 1.0e5
arepoTime = arepoLength/arepoVel
arepoDensity = arepoMass/arepoLength/arepoLength/arepoLength
arepoEnergy= arepoMass*arepoVel*arepoVel
arepoColumnDensity = arepoMass/arepoLength/arepoLength

# read the data
# TODO: test rotation of vx vy is ok (plot vfields)
# TODO: lvplots
data, header = read_snapshot_file('whole_disk_212')
t = header['time']
omega = 4.0
x,y,z = data['pos'].T
x,y,z = x-120, y-120, z-10
x,y = rotate(x,y,omega*t)
vx,vy,vz = data['vel'].T
vx,vy = rotate(vx,vy,omega*t)
rho = data['rho'] * arepoDensity
energy_per_unit_mass = data['u_therm']*arepoEnergy/arepoMass
xH2, xHp, xCO = data['chem'].T
l,b,r,vl,vb,vr = vxyz2vlbr(x,y,z,vx,vy,vz)
# chemistry stuff
# nHtot = nHI + nHp + 2*nH2
# nTOT = nHI + nH2 + nHp + ne + nHe
xHe=0.1
mp = 1.6726231e-24
kb = 1.3806485e-16
xHI = 1 - xHp -2*xH2
nHtot = rho/((1. + 4.0 * xHe) * mp)
nHp = xHp*nHtot
nH2 = xH2*nHtot
nCO = xCO*nHtot
nHI = (1.0 - xHp - 2.0*xH2)*nHtot
nTOT = nHtot*(1.0 + xHp - xH2 + xHe)
mu = rho/(nTOT*mp)
T = (2.0/3.0)*energy_per_unit_mass*mu*mp/kb

# Create interpolation functions
points = np.vstack((x,y,z)).T
rhof = NearestNDInterpolator(points,rho)
vxf = NearestNDInterpolator(points,vx)
vyf = NearestNDInterpolator(points,vy)

# produce vertical slices at different y
xgrid, ygrid, zgrid = linspace(-30,30,51), linspace(-30,30,51), linspace(-1,1,9)
XGRID,YGRID,ZGRID = np.meshgrid(xgrid,ygrid,zgrid)
RHO = rhof(XGRID,YGRID,ZGRID)
VX = vxf(XGRID,YGRID,ZGRID)
VY = vyf(XGRID,YGRID,ZGRID)

# plot vfields
# plot horizontal slices
nslices = zgrid.size
fig, axarr = pl.subplots(3,3,figsize=(12,10),sharex=True,sharey=True)
axarr = axarr.flatten()
extent = (xgrid.min(),xgrid.max(),ygrid.min(),ygrid.max())
levels = logspace(-28,-20,256)
norm = mc.BoundaryNorm(levels, 256)
for i in arange(nslices):
  im = axarr[i].quiver(XGRID[:,:,i], YGRID[:,:,i], VX[:,:,i], VY[:,:,i],scale=0.8*1e4,angles='xy')
  axarr[i].annotate('z=%.2f'%(zgrid[i]),xy=(0.7,0.9),xycoords='axes fraction',fontsize=16)
axarr[0].set_aspect('equal')
axarr[7].set_xlabel(r'$x\, ({\rm 100\, pc})$',fontsize=25)
axarr[3].set_ylabel(r'$y\, ({\rm 100\, pc})$',fontsize=25)
axarr[0].set_xlim(-30,30)
axarr[1].set_ylim(-20,20)
fig.subplots_adjust(wspace=0.0,hspace=0.0)
fig.savefig('vfields.png',bbox_inches='tight')
pl.close()

# plot streamlines
nslices = zgrid.size
fig, axarr = pl.subplots(3,3,figsize=(12,10),sharex=True,sharey=True)
axarr = axarr.flatten()
extent = (xgrid.min(),xgrid.max(),ygrid.min(),ygrid.max())
levels = logspace(-28,-20,256)
norm = mc.BoundaryNorm(levels, 256)
for i in arange(nslices):
  im = axarr[i].streamplot(xgrid, ygrid, VX[:,:,i], VY[:,:,i],density=1.0)
  axarr[i].annotate('z=%.2f'%(zgrid[i]),xy=(0.7,0.9),xycoords='axes fraction',fontsize=16)
axarr[0].set_aspect('equal')
axarr[7].set_xlabel(r'$x\, ({\rm 100\, pc})$',fontsize=25)
axarr[3].set_ylabel(r'$y\, ({\rm 100\, pc})$',fontsize=25)
axarr[0].set_xlim(-30,30)
axarr[1].set_ylim(-20,20)
#fig.subplots_adjust(right=0.9)
fig.subplots_adjust(wspace=0.0,hspace=0.0)
#cbar_ax = fig.add_axes([0.92, 0.1, 0.01, 0.8])
#pl.colorbar(im,cax=cbar_ax,ticks=[1e-20,1e-21,1e-22,1e-23,1e-24,1e-25,1e-26,1e-27,1e-28],format='%.0e')
fig.savefig('streamlines.png',bbox_inches='tight')
pl.close()


