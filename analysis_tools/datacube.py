import os
import matplotlib as mpl
import matplotlib.pyplot as pl
import matplotlib.colors as mc
import matplotlib.ticker as ticker
import numpy as np
import compute_emissivities as emissivity
from numpy import uint32, uint64, float64, float32
from numpy import sqrt, log, exp, arccosh, sinh, cosh, tanh, pi, cos, sin, arctan2, arccos, array
from scipy import ndimage
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

mpl.use('Agg')
pl.rcParams['text.usetex'] = True

def fmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    if (int(a)==1):
        return r'$10^{{{}}}$'.format(b)
    else:
        return r'${} \times 10^{{{}}}$'.format(a, b)

def get_aspect(ax=None):
    if ax is None:
        ax = pl.gca()
    fig = ax.figure

    ll, ur = ax.get_position() * fig.get_size_inches()
    width, height = ur - ll
    axes_ratio = height / width
    aspect = axes_ratio / ax.get_data_ratio()

    return aspect

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
    count = int(data_size/np.dtype(dtype).itemsize)
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
    print("ID type: ", dtype)
    return readu(f, dtype, 1)

def read_snapshot_file(filename):
    """ Reads a binary arepo file """
    f = open(filename, mode='rb')
    print("Loading file %s" % filename)
    data = {} # dictionary to hold data
    data_gas = {}
    data_sink = {}
    # read the header
    header = read_header(f)
    nparts = header['num_particles']
    masses = header['mass']
    total = nparts.sum()
    n_gas = nparts[0]
    n_sink = nparts[5]
    print('Particles', nparts)
    print('Gas particles', n_gas)
    print('Sink particles', n_sink)
    print('Time = ', header['time'])
    precision = float32
    print('Reading positions')
    data['pos'] = readu(f, precision, 3).reshape((total, 3))
    print('Reading velocities')
    data['vel'] = readu(f, precision, 3).reshape((total, 3))
    print('Reading IDs')
    data['id'] = readIDs(f, total)
    print('Reading masses')
    data['mass'] = readu(f, precision, 1)
    print('Reading internal energy')
    data['u_therm'] = readu(f, precision, 1)
    print('Reading densities')
    data['rho'] = readu(f, precision, 1)
    print('Reading chemical abundances')
    data['chem'] = readu(f, precision, 3).reshape((n_gas, 3))
    print('Reading dust temperatures')
    data['tdust'] = readu(f, precision, 1)
    f.close()
    for field in data:
        data_gas[field] = data[field][0:n_gas]
    data_sink['pos'] = data['pos'][n_gas:(n_gas+n_sink)]
    data_sink['vel'] = data['vel'][n_gas:(n_gas+n_sink)]
    data_sink['id'] = data['id'][n_gas:(n_gas+n_sink)]
    data_sink['mass'] = data['mass'][n_gas:(n_gas+n_sink)]
    return data_gas, data_sink, header

def read_arepo_image(filename):
    f = open(filename, mode='rb')
    print("Loading file %s" % filename)

    npix_x = np.fromfile(f, uint32, 1)
    npix_y = np.fromfile(f, uint32, 1)

    print(npix_x, npix_y)
    arepo_image = np.fromfile(f, float32, int(npix_x*npix_y)).reshape((int(npix_x), int(npix_y)))
    arepo_image = np.rot90(arepo_image)
    f.close()
    return arepo_image

def read_projections(fname, isnap):
    Sigma = read_arepo_image(fname+'/density_proj_%03d'%isnap)
    xHp =   read_arepo_image(fname+'/xHP_proj_%03d'%isnap)
    xH2 =   read_arepo_image(fname+'/xH2_proj_%03d'%isnap)
    xCO =   read_arepo_image(fname+'/xCO_proj_%03d'%isnap)

    xHe = 0.1
    mp = 1.6726231e-24
    kb = 1.3806485e-16
    xHI = 1 - xHp -2*xH2
    NHtot = (Sigma * arepoColumnDensity) / ((1. + 4.0 * xHe) * mp)
    NHp = xHp*NHtot
    NH2 = xH2*NHtot
    NCO = xCO*NHtot
    NHI = (1.0 - xHp - 2.0*xH2)*NHtot
    NTOT = NHtot*(1.0 + xHp - xH2 + xHe)

    return NTOT, NHI, NCO, NH2, NHp

def rotate(x,y,theta):
    xprime = x*cos(theta) - y*sin(theta)
    yprime = x*sin(theta) + y*cos(theta)
    return xprime, yprime

#############################
# XYZ are cartesian coordinates centered at the Sun position. X axis points to the GC, Z axis up to north Galactic pole.
# xyz are cartesian coordinates centered at the GC with same orientations as XYZ.
# lbr are spherical coordinates centered at the Sun position (i.e., the usual Galactic coordinates).
# the following functions convert back and forth between the above coordinates
#############################

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
    return x+R0,y,z

def XYZ2xyz(X,Y,Z):
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
    vsun = 220.0
    vr = vx*rhat[0] + (vy-vsun)*rhat[1] + vz*rhat[2]
    vb = vx*bhat[0] + (vy-vsun)*bhat[1] + vz*bhat[2]
    vl = vx*lhat[0] + (vy-vsun)*lhat[1]
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

def label_line(xxx, yyy, ax, line, label, color='0.5', fs=14, halign='left'):
    xdata, ydata = line.get_data()
    x1 = xdata[0]
    x2 = xdata[-1]
    y1 = ydata[0]
    y2 = ydata[-1]
    if halign.startswith('l'):
        xx = x1
        halign = 'left'
    elif halign.startswith('r'):
        xx = x2
        halign = 'right'
    elif halign.startswith('c'):
        xx = 0.5*(x1 + x2)
        halign = 'center'
    else:
        raise ValueError("Unrecognized `halign` = '{}'.".format(halign))
    yy = np.interp(xx, xdata, ydata)
    ylim = ax.get_ylim()
    # xytext = (10, 10)
    xytext = (0, 0)
    text = ax.annotate(label, xy=(xxx, yyy), xytext=xytext, textcoords='offset points',
    size=fs, color=color, zorder=1,
    horizontalalignment=halign, verticalalignment='center_baseline')
    sp1 = ax.transData.transform_point((x1, y1))
    sp2 = ax.transData.transform_point((x2, y2))
    rise = (sp2[1] - sp1[1])
    run = (sp2[0] - sp1[0])
    slope_degrees = np.degrees(np.arctan2(rise, run))
    text.set_rotation_mode('anchor')
    text.set_rotation(slope_degrees)
    ax.set_ylim(ylim)
    return text

def read_sink_snap(filename, SNe=False, longid=True):
    f = open(filename, mode='rb')
    Time = np.fromfile(f, np.float64, 1)
    print(Time)
    NSinksAllTasks = np.fromfile(f, np.uint32, 1)
    print(NSinksAllTasks)

    data = {}

    pos = []
    vel = []
    accel = []
    mass = []
    massOld = []
    #AccretionRadius = []
    formationMass = []
    formationTime = []
    ID = []
    HomeTask = []
    Index = []
    FormationOrder = []
    N_SNe = []
    ExplosionTime = []
    MassStillToConvert = []
    AccretionTime = []

    for i in np.arange(NSinksAllTasks):
        pos += [np.fromfile(f, np.float64, 3)]
        vel += [np.fromfile(f, np.float64, 3)]
        accel += [np.fromfile(f, np.float64, 3)]
        mass += [np.fromfile(f, np.float64, 1)]
        massOld += [np.fromfile(f, np.float64, 1)]
        #AccretionRadius += [np.fromfile(f, np.float64, 1)]
        formationMass += [np.fromfile(f, np.float64, 1)]
        formationTime += [np.fromfile(f, np.float64, 1)]
        if (longid):
            ID += [np.fromfile(f, np.uint64, 1)]
        else:
            ID += [np.fromfile(f, np.uint32, 1)]
        HomeTask += [np.fromfile(f, np.uint32, 1)]
        Index += [np.fromfile(f, np.uint32, 1)]
        FormationOrder += [np.fromfile(f, np.uint32, 1)]
        if (SNe):
            N_SNe += [np.fromfile(f, np.uint32, 1)]
            if (longid == False):
                dummy = [np.fromfile(f, np.uint32, 1)]
            ExplosionTime += [np.fromfile(f, np.float64, MAXSNE)]
            MassStillToConvert += [np.fromfile(f, np.float64, MAXACCRETIONEVENTS)]
            AccretionTime += [np.fromfile(f, np.float64, MAXACCRETIONEVENTS)]
        else:
            dummy = [np.fromfile(f, np.uint32, 1)]

    f.close()

    data['pos'] = np.array(pos)
    data['vel'] = np.array(vel)
    data['accel'] = np.array(accel)
    data['mass'] = np.array(mass)
    data['formationMass'] = np.array(formationMass)
    data['formationTime'] = np.array(formationTime)
    data['ID'] = np.array(ID)
    data['HomeTask'] = np.array(HomeTask)
    data['Index'] = np.array(Index)
    data['FormationOrder'] = np.array(FormationOrder)
    if (SNe):
        data['N_SNe'] = np.array(N_SNe)
        data['ExplosionTime'] = np.array(ExplosionTime)
        data['MassStillToConvert'] = np.array(MassStillToConvert)
        data['AccretionTime'] = np.array(AccretionTime)
    #data['massOld'] = np.array(massOld)
    #data['AccretionRadius'] = np.array(AccretionRadius)

    return data, Time, NSinksAllTasks

#######################
# define class to manage arepo snapshot
#######################
class Snapshot:
    # use "chem=False" to load a snapshot without chemistry
    def __init__(self,isnap,halfbox=120,halfboy=120,halfboz=120,arepoLength=3.0856e20,arepoMass=1.911e33,arepoVel=1.0e5,chem=True):
        #
        self.isnap = isnap
        self.halfbox = halfbox
        self.halfboy = halfboy
        self.halfboz = halfboz
        self.chem = chem
        # convert from code units to cgs
        self.arepoLength = arepoLength
        self.arepoMass = arepoMass
        self.arepoVel = arepoVel
        self.arepoTime = self.arepoLength/self.arepoVel
        self.arepoDensity = self.arepoMass/self.arepoLength/self.arepoLength/self.arepoLength
        self.arepoEnergy= self.arepoMass*self.arepoVel*self.arepoVel
        self.arepoColumnDensity = self.arepoMass/self.arepoLength/self.arepoLength
        # read header
        self.path = '../OUTPUT/'
        self.fname = self.path + 'snap_%03d'%self.isnap
        # proj1 and 2 flag
        self.proj1 = False
        self.proj2 = False

    #####################
    # read full snapshot
    #####################
    def read_full(self):
        # read the full snapshot data
        self.data_gas, self.dummy, self.header = read_snapshot_file(self.fname)
        self.t = self.header['time'][0]
        self.x,self.y,self.z = self.data_gas['pos'].T
        self.x,self.y,self.z = self.x-self.halfbox, self.y-self.halfboy, self.z-self.halfboz
        self.vx,self.vy,self.vz = self.data_gas['vel'].T
        self.rho = self.data_gas['rho']
        self.masses = self.data_gas['mass']
        self.energy_per_unit_mass = self.data_gas['u_therm']
        self.volumes = self.data_gas['mass'] / self.data_gas['rho'] 
        #
        self.tMyr = self.t*self.arepoTime/(60*60*24*365*1e6)
        self.rho_cgs = self.rho*self.arepoDensity
        self.energy_per_unit_mass_cgs = self.energy_per_unit_mass*(self.arepoEnergy/self.arepoMass)
        # number of particles
        self.n_gas = self.header['num_particles'][0]
        self.n_sink = self.header['num_particles'][5]
        # chemistry stuff
        # nHtot = nHI + nHp + 2*nH2
        # nTOT = nHI + nH2 + nHp + ne + nHe
        if(self.chem):
            print('calculating chemistry stuff...')
            # chemical quantities are in cgs
            self.kpc_to_cm = 3.0856e21
            self.xHe = 0.1
            self.mp = 1.6726231e-24 # proton mass in g
            self.kb = 1.3806485e-16 # Boltzmann constant in g*cm^2/s^2/K 
            self.xH2, self.xHp, self.xCO = self.data_gas['chem'].T
            self.xHI = 1 - self.xHp -2*self.xH2
            self.nHtot = self.rho_cgs/((1. + 4.0 * self.xHe) * self.mp)
            self.nHp = self.xHp*self.nHtot
            self.nH2 = self.xH2*self.nHtot
            self.nCO = self.xCO*self.nHtot
            self.nHI = (1.0 - self.xHp - 2.0*self.xH2)*self.nHtot
            self.nHe = self.xHe*self.nHtot
            self.nTOT = self.nHtot*(1.0 + self.xHp - self.xH2 + self.xHe)
            self.mu = self.rho_cgs/(self.nTOT*self.mp)
            self.T = (2.0/3.0)*self.energy_per_unit_mass_cgs*self.mu*self.mp/self.kb
            # masses in [M]=M_sol
            self.massesH2 = (self.nH2*(2*self.mp)/self.arepoDensity)*self.volumes
            self.massesHI = (self.nHI*self.mp/self.arepoDensity)*self.volumes
            self.massesHp = (self.nHp*self.mp/self.arepoDensity)*self.volumes
            self.massesCO = (self.nCO*28*self.mp/self.arepoDensity)*self.volumes
            self.massesHe = (self.nHe*4*self.mp/self.arepoDensity)*self.volumes

    #####################
    # read sinks
    #####################
    def read_sinks(self):
        print('reading sinks...')
        self.fname_sink = self.path+'sink_snap_%03d'%isnap
        self.data_sink, self.t_sink, self.NSinksAllTasks = read_sink_snap(self.fname_sink)
        self.x_sink,self.y_sink,self.z_sink = self.data_sink['pos'].T
        self.x_sink,self.y_sink,self.z_sink = self.x_sink-self.halfbox, self.y_sink-self.halfboy, self.z_sink-self.halfboz
        self.vx_sink,self.vy_sink,self.vz_sink = self.data_sink['vel'].T
        self.masses_sink = self.data_sink['mass']
        self.R_sink = sqrt(self.x_sink**2+self.y_sink**2)
        self.age_sink = self.t_sink - self.data_sink['formationTime']
        self.age_sink_Myr = self.age_sink[:,0]*(self.arepoTime/(60*60*24*365*1e6))

    #####################
    # read projections
    #####################
    def read_proj1(self,foldername='proj1'):
        print('reading projection 1...')
        self.rho_proj1 = read_arepo_image(self.path+foldername+'/density_proj_%03d'%self.isnap)
        self.xHp_proj1 = read_arepo_image(self.path+foldername+'/xHP_proj_%03d'%isnap)
        self.xH2_proj1 = read_arepo_image(self.path+foldername+'/xH2_proj_%03d'%isnap)
        self.xCO_proj1 = read_arepo_image(self.path+foldername+'/xCO_proj_%03d'%isnap)
        self.xHI_proj1 = 1 - self.xHp_proj1 -2*self.xH2_proj1
        self.nHtot_proj1 = (self.rho_proj1 * self.arepoColumnDensity) / ((1. + 4.0 * self.xHe) * self.mp)
        self.nHp_proj1 = self.xHp_proj1*self.nHtot_proj1
        self.nH2_proj1 = self.xH2_proj1*self.nHtot_proj1
        self.nCO_proj1 = self.xCO_proj1*self.nHtot_proj1
        self.nHI_proj1 = (1.0 - self.xHp_proj1 - 2.0*self.xH2_proj1)*self.nHtot_proj1
        self.nTOT_proj1 = self.nHtot_proj1*(1.0 + self.xHp_proj1 - self.xH2_proj1 + self.xHe)
        self.proj1 = True

    def read_proj2(self,foldername='proj2'):
        print('reading projection 2...')
        self.rho_proj2 = read_arepo_image(self.path+foldername+'/density_proj_%03d'%self.isnap)
        self.xHp_proj2 = read_arepo_image(self.path+foldername+'/xHP_proj_%03d'%isnap)
        self.xH2_proj2 = read_arepo_image(self.path+foldername+'/xH2_proj_%03d'%isnap)
        self.xCO_proj2 = read_arepo_image(self.path+foldername+'/xCO_proj_%03d'%isnap)
        self.xHI_proj2 = 1 - self.xHp_proj2 -2*self.xH2_proj2
        self.nHtot_proj2 = (self.rho_proj2 * self.arepoColumnDensity) / ((1. + 4.0 * self.xHe) * self.mp)
        self.nHp_proj2 = self.xHp_proj2*self.nHtot_proj2
        self.nH2_proj2 = self.xH2_proj2*self.nHtot_proj2
        self.nCO_proj2 = self.xCO_proj2*self.nHtot_proj2
        self.nHI_proj2 = (1.0 - self.xHp_proj2 - 2.0*self.xH2_proj2)*self.nHtot_proj2
        self.nTOT_proj2 = self.nHtot_proj2*(1.0 + self.xHp_proj2 - self.xH2_proj2 + self.xHe)
        self.proj2 = True

    #####################
    # calc lbv points
    #####################
    def calc_lbv(self):
        print('calculating lbv points...')
        self.l,self.b,self.r,self.vl,self.vb,self.vr = vxyz2vlbr(self.x,self.y,self.z,self.vx,self.vy,self.vz)
        if(self.n_sink):
            self.l_sink,self.b_sink,self.r_sink,self.vl_sink,self.vb_sink,self.vr_sink = vxyz2vlbr(self.x_sink,self.y_sink,self.z_sink,self.vx_sink,self.vy_sink,self.vz_sink)

    #####################
    # calc CII emissivities
    #####################
    def calc_CII_emissivity(self):
        G0 = 1.7 # ISRF in units of Habing field
        CarbAbund=1.4e-4
        OxiAbund=3.2e-4
        abcII = CarbAbund - self.xCO
        abe = self.xHp
        abh2 = self.xH2
        abHI = (1. - self.xHp - 2.*self.xH2)
        abhp = self.xHp
        aboI = OxiAbund * np.ones(self.xH2.size)
        cII_spec_emiss, oI_spec_emiss_63, oI_spec_emiss_145 = emissivity.compute_emissivities(abcII=abcII, abe=abe, abh2=abh2, abHI=abHI, abhp=abhp, aboI=aboI, G0=G0, rho=self.rho_cgs, temp=self.T)
        cII_emiss = cII_spec_emiss * self.masses*self.arepoMass
        self.cII_emiss = cII_emiss # [cII_emiss] = erg/s

    #####################
    # rotate the snapshot by a given angle
    #####################
    def rotate_full(self,theta):
        self.x,self.y = rotate(self.x,self.y,theta)
        self.vx,self.vy = rotate(self.vx,self.vy,theta)
        if(self.n_sink):
            self.x_sink,self.y_sink = rotate(self.x_sink,self.y_sink,theta)
            self.vx_sink,self.vy_sink = rotate(self.vx_sink,self.vy_sink,theta)

    #####################
    # traslate snapshot
    #####################
    def shift_full(self,x0,y0,z0,vx0,vy0,vz0):
        self.x,self.y,self.z = self.x - x0, self.y - y0, self.z - z0
        self.vx, self.vy, self.vz = self.vx - vx0, self.vy - vy0, self.vz - vz0
        if(self.n_sink):
            self.x_sink,self.y_sink,self.z_sink= self.x_sink - x0, self.y_sink - y0, self.z_sink - z0
            self.vx_sink,self.vy_sink,self.vz_sink= self.vx_sink - vx0, self.vy_sink - vy0, self.vz_sink - vz0

    #####################
    # rotate the snapshot by a given angle around given unit vector u with arbitrary direction
    # u = unit vectors that is the axis of rotation (3 components, normalised to 1)
    # alpha = angle of rotation around u in radians
    #####################

    def rotate_3D(self,u,alpha):
        self.x,self.y,self.z = Rotate_around_u(self.x,self.y,self.z,u,alpha)
        self.vx,self.vy,self.vz = Rotate_around_u(self.vx,self.vy,self.vz,u,alpha)
        if(self.n_sink):
            self.x_sink,self.y_sink,self.z_sink = Rotate_around_u(self.x_sink,self.y_sink,self.z_sink,u,alpha)
            self.vx_sink,self.vy_sink,self.vz_sink = Rotate_around_u(self.vx_sink,self.vy_sink,self.vz_sink,u,alpha)

    #####################
    # rotate projections by given angle
    #####################
    def rotate_proj(self,theta):
        if(self.proj1):
            self.rho_proj1 = ndimage.rotate(self.rho_proj1, np.degrees(theta), reshape=False,order=1)
            self.xHp_proj1 = ndimage.rotate(self.xHp_proj1, np.degrees(theta), reshape=False,order=1)
            self.xH2_proj1 = ndimage.rotate(self.xH2_proj1, np.degrees(theta), reshape=False,order=1)
            self.xCO_proj1 = ndimage.rotate(self.xCO_proj1, np.degrees(theta), reshape=False,order=1)
            self.xHI_proj1 = ndimage.rotate(self.xHI_proj1, np.degrees(theta), reshape=False,order=1)
            self.nHtot_proj1 = ndimage.rotate(self.nHtot_proj1, np.degrees(theta), reshape=False,order=1)
            self.nHp_proj1 = ndimage.rotate(self.nHp_proj1, np.degrees(theta), reshape=False,order=1)
            self.nH2_proj1 = ndimage.rotate(self.nH2_proj1, np.degrees(theta), reshape=False,order=1)
            self.nCO_proj1 = ndimage.rotate(self.nCO_proj1, np.degrees(theta), reshape=False,order=1)
            self.nHI_proj1 = ndimage.rotate(self.nHI_proj1, np.degrees(theta), reshape=False,order=1)
            self.nTOT_proj1 = ndimage.rotate(self.nTOT_proj1, np.degrees(theta), reshape=False,order=1)
        #
        if(self.proj2):
            self.rho_proj2 = ndimage.rotate(self.rho_proj2, np.degrees(theta), reshape=False,order=1)
            self.xHp_proj2 = ndimage.rotate(self.xHp_proj2, np.degrees(theta), reshape=False,order=1)
            self.xH2_proj2 = ndimage.rotate(self.xH2_proj2, np.degrees(theta), reshape=False,order=1)
            self.xCO_proj2 = ndimage.rotate(self.xCO_proj2, np.degrees(theta), reshape=False,order=1)
            self.xHI_proj2 = ndimage.rotate(self.xHI_proj2, np.degrees(theta), reshape=False,order=1)
            self.nHtot_proj2 = ndimage.rotate(self.nHtot_proj2, np.degrees(theta), reshape=False,order=1)
            self.nHp_proj2 = ndimage.rotate(self.nHp_proj2, np.degrees(theta), reshape=False,order=1)
            self.nH2_proj2 = ndimage.rotate(self.nH2_proj2, np.degrees(theta), reshape=False,order=1)
            self.nCO_proj2 = ndimage.rotate(self.nCO_proj2, np.degrees(theta), reshape=False,order=1)
            self.nHI_proj2 = ndimage.rotate(self.nHI_proj2, np.degrees(theta), reshape=False,order=1)
            self.nTOT_proj2 = ndimage.rotate(self.nTOT_proj2, np.degrees(theta), reshape=False,order=1)

    #####################
    # create interpolating functions
    #####################
    def create_interpolating_functions(self):
        print('creating interpolating functions...')
        points = np.vstack((self.x,self.y,self.z)).T
        self.frho = NearestNDInterpolator(points,self.rho)
        self.fT = NearestNDInterpolator(points,self.T)

    #####################
    # reduce size of arrays
    #####################
    def reduce_size(self,DD=10):
        print('reducing sizes...')
        self.x,self.y,self.z = self.x[::DD], self.y[::DD], self.z[::DD]
        self.vx,self.vy,self.vz = self.vx[::DD], self.vy[::DD], self.vz[::DD]
        self.rho = self.rho[::DD]
        self.masses = self.masses[::DD]
        self.xH2, self.xHp, self.xCO = self.xH2[::DD], self.xHp[::DD], self.xCO[::DD] 
        self.energy_per_unit_mass = self.energy_per_unit_mass[::DD]
        self.volumes = self.volumes[::DD]
        self.rho_cgs = self.rho_cgs[::DD]
    
    #####################
    # cut everything that does not satisfy a condition (e.g. condition = sqrt(x**2+y**2)<R**2)
    #####################
    def impose_condition(self,condition):
        CC = condition
        self.x,self.y,self.z = self.x[CC], self.y[CC], self.z[CC]
        self.vx,self.vy,self.vz = self.vx[CC], self.vy[CC], self.vz[CC]
        self.rho = self.rho[CC]
        self.masses = self.masses[CC]
        self.xH2, self.xHp, self.xCO = self.xH2[CC], self.xHp[CC], self.xCO[CC] 
        self.energy_per_unit_mass = self.energy_per_unit_mass[CC]
        self.volumes = self.volumes[CC]
        self.rho_cgs = self.rho_cgs[CC]
        self.massesH2 = self.massesH2[CC]
        self.massesHI = self.massesHI[CC]
        self.massesHp = self.massesHp[CC]
        self.massesCO = self.massesCO[CC]
        self.massesHe = self.massesHe[CC]

####################
# start main event
####################

isnap = 1150

# define parameters of the snapshot
halfbox = 500
halfboy = 500
halfboz = 500

snapshot = Snapshot(isnap,halfbox,halfboy,halfboz)
snapshot.read_full()
snapshot.n_sink = 0

snapshot.shift_full(55,-12,105,5,31,-66)
snapshot.rotate_full(np.radians(20))

snapshot.calc_CII_emissivity()

theta = np.radians(180)
alpha = np.radians(24)
u = [cos(theta),sin(theta),0]
snapshot.rotate_3D(u,alpha)
  
####################
# bin points in (x,y,vz)
####################

print('binning x,y,vz...')
triplets = np.vstack((snapshot.x,snapshot.y,snapshot.vz)).T
xmin, xmax, dx = -150, 150, 0.5
ymin, ymax, dy = -150, 150, 0.5 
vmin, vmax, dv = -300 + 66, 300 + 66, 5.0
xs = np.arange(xmin,xmax+dx,dx)
ys = np.arange(ymin,ymax+dy,dy)
vs = np.arange(vmin,vmax+dv,dv)
xbins = np.arange(xmin-dx/2,xmax+3*dx/2,dx)
ybins = np.arange(ymin-dy/2,ymax+3*dy/2,dy)
vbins = np.arange(vmin-dv/2,vmax+3*dv/2,dv)

w = (snapshot.cII_emiss)/4*pi # 4*pi is to have units of steradians

X,Y,VZ = np.meshgrid(xs,ys,vs)

datacube, bins = np.histogramdd(triplets, bins = (xbins,ybins,vbins),weights=w)

xyplot = datacube.sum(axis=2).T / (dx*100*dy*100 * 3.086e16**2)
vzplot = np.ma.average(-VZ,weights=datacube,axis=2).T

####################
# plot xy
####################

fig, axarr = pl.subplots(1,2,figsize=(10,6))
ax1 = axarr[0]
ax2 = axarr[1]
  
# plot CII intensity
s = 10
extent = (xmin/s,xmax/s,ymin/s,ymax/s)
levels = np.logspace(-2,2,256)
norm = mc.BoundaryNorm(levels,256)
im1 = ax1.imshow(xyplot,norm=norm,extent=extent,cmap='Greys',interpolation='nearest',origin='l')

# plot vr
extent = (xmin/s,xmax/s,ymin/s,ymax/s)
levels = np.linspace(-100,100,201)
norm  = mc.BoundaryNorm(levels,256)
im2 = ax2.imshow(vzplot,norm=norm,extent=extent,cmap='bwr',interpolation='nearest',origin='l')

ax1.plot([0,1000*u[0]],[0,1000*u[1]],'--',color='k',label='axis of rotation')

# formatting
ax1.grid(ls='dashed')
ax2.grid(ls='dashed')
ax1.set_aspect('1')
ax2.set_aspect('1')
ax1.set_xlabel(r'$x\ [\rm{kpc}]$',fontsize=20)
ax1.set_ylabel(r'$y\ [\rm{kpc}]$',fontsize=20)
ax2.set_xlabel(r'$x\ [\rm{kpc}]$',fontsize=20)
ax2.set_ylabel(r'$y\ [\rm{kpc}]$',fontsize=20)
ax1.set_xlim(xmin/s,xmax/s)
ax1.set_ylim(ymin/s,ymax/s)
ax2.set_xlim(xmin/s,xmax/s)
ax2.set_ylim(ymin/s,ymax/s)
ax1.set_title(r'$t=%.1f \; \rm Myr$'%(snapshot.tMyr),fontsize=18)
ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)

cbarticks = np.logspace(0,10,11)
cbar1 = fig.colorbar(im1,ax=ax1,ticks=cbarticks, format=ticker.FuncFormatter(fmt),shrink=0.6)
cbar1.set_label(r'$[\rm erg \, s^{-1} \, sr^{-1}]$',fontsize=14,y=1.08,labelpad=-18,rotation=0)

cbarticks = np.linspace(-200,200,11)
cbar2 = fig.colorbar(im2,ax=ax2,ticks=cbarticks, format='%g',shrink=0.6)
cbar2.set_label(r'$[\rm km/s]$',fontsize=14,y=1.08,labelpad=-18,rotation=0)

ax1.legend()

# final things
pl.tight_layout()
fig.savefig('datacube_%03d.png'%snapshot.isnap,bbox_inches='tight',dpi=300)
pl.show()

###################
# Write synthetic datacube to fits file
###################

from astropy.table import Table
from astropy.io import fits

hdu = fits.PrimaryHDU(data=datacube)
header = hdu.header
u = 10
header['CTYPE1'] = 'vlos'
header['CTYPE3'] = 'x'
header['CTYPE2'] = 'y'
header['CRVAL1'] = (vs.min(), 'units: km/s')
header['CRVAL2'] = (ys.min()/u, 'units: kpc')
header['CRVAL3'] = (xs.min()/u, 'units: kpc')
header['CDELT1'] = (dv, 'dv=spacing in v direction')
header['CDELT2'] = (dy/u, 'dy=spacing in y direction')
header['CDELT3'] = (dx/u, 'dx=spacing in x direction')
hdu.writeto('datacube.fits')

