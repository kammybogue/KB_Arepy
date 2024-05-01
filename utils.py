### TODO:  - change xcent, ycent, zcent to indexes of centre (default param)
#          - seperate out below functions into seperate files (load, plot, utils) rather than them all being in utils
#          - 

# AUTHOR; KAMRAN R J BOGUE #

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc

import arepy.read_write.binary_read as rsnap
from arepy.utility import cgs_constants as cgs



########### ########## ########### ########## GLOBAL PARAMETERS / CONFIGURATION ########## ########## ########### ##########

#internal arepo units in cgs
ulength = 3.0856e20
umass = 1.991e33
uvel = 1.0e5

utime = ulength/uvel
udensity = umass/ulength/ulength/ulength
uenergy= umass*uvel*uvel
ucolumn = umass/ulength/ulength
umag = umass**0.5 / ulength**1.5 * uvel

uMyr=utime/(60.*60.*24.*365.25*1.e6)

uparsec=ulength/3.0856e18



########### ########## ########### ########## LOADING DATA FUNCTIONS ########## ########## ########### ##########

def load_data(base, filenum, ABHE=0.1, read_snap=True):
    """ Loads projection images of a given snapshot """
    
    ColumnDensity = rsnap.read_image(base + 'density_proj_' + filenum)
    xHp_proj = rsnap.read_image(base + 'xHP_proj_' + filenum)
    xH2_proj = rsnap.read_image(base + 'xH2_proj_' + filenum)

    NH = ((ColumnDensity * rsnap.arepoColumnDensity) / ((1. + 4. * ABHE) * cgs.mp))
    Ndensity = NH * (1. + ABHE - xH2_proj + xHp_proj)
    
    B_field = rsnap.read_vector_image(base + 'magnetic_proj_' + filenum)
    
    if read_snap==True:
        f3 = base + 'snap_' + filenum +'.hdf5'

        rsnap.io_flags['variable_metallicity'] = False
        rsnap.io_flags['time_steps'] = True
        rsnap.io_flags['mc_tracer'] = False
        rsnap.io_flags['sgchem'] = True
        rsnap.io_flags['MHD'] = True

        output = rsnap.read_snapshot_hdf5(f3) # ,sink_data 
        if len(output) > 2:        
            data, header, sink_data = output

            # defining sink properties
            sink_pos = sink_data['Coordinates']
            sink_mass = sink_data['Masses']
            sink_vel = sink_data['Velocities']

            SSx = sink_pos[:,0]
            SSy = sink_pos[:,1]
            SSz = sink_pos[:,2]

            SSx = (SSx - 500.) / 10.
            SSy = (SSy - 500.) / 10.

            Smass = sink_mass
        else:
            data, header = output

        time = header['Time']
        
        return Ndensity, B_field, SSx, SSy, time, xHp_proj, xH2_proj, NH
    else:
        return Ndensity, B_field, xHp_proj, xH2_proj, NH


def load_snap_data(base, filenum, ABHE=0.1, xHe=0.1, mp=1.6726231e-24, kb=1.3806485e-16, **kwargs): 
    """ 
    Loads snapshot data from a specified file, doesn't load the projections.

    Parameters:
    - base (str): The base path to the snapshot files.
    - filenum (str): The snapshot number to load.
    - ABHE (float, optional): The abundance of helium in the system. Default is 0.1.
    - xHe (float, optional): The
    - mp (float, optional): The proton mass in grams. Default is 1.6726231e-24.
    - kb (float, optional): The Boltzmann constant in erg per Kelvin. Default is 1.3806485e-16.
    - **kwargs: Additional keyword arguments for future expansion.

    Returns:
    - mass (array): Array containing masses of particles in the snapshot.
    - pos (array): Array containing positions of particles in the snapshot.
    - rho (array): Array containing densities of particles in the snapshot.
    - bfield (array): Array containing magnetic field of particles in the snapshot.
    - yn (array): Array containing number densities of particles in the snapshot.
    - T (array): Array containing temperatures of particles in the snapshot.
    - vels (array): Array containing velocities of particles in the snapshot.
    - chem (array): Array containing chemical abundances of particles in the snapshot.
    - time (float): The physical time of the snapshot.
    
    """

    # Begin the data load in
    f = base + 'snap_' + filenum +'.hdf5'
    
    print("loading data from;",f)
    
    # Print used kwarg values
    print("using ABHE = ", ABHE)
    print("using xHe = ", xHe)
    #print("using mp = ", mp) #these should not change
    #print("using kb = ", kb)

    rsnap.io_flags['variable_metallicity'] = False
    rsnap.io_flags['time_steps'] = True
    rsnap.io_flags['mc_tracer'] = False
    rsnap.io_flags['sgchem'] = True
    rsnap.io_flags['MHD'] = True

    output = rsnap.read_snapshot_hdf5(f) 
    if len(output) > 2:        
        data, header, sink_data = output

        # defining sink properties
        sink_pos = sink_data['Coordinates']
        sink_mass = sink_data['Masses']
        sink_vel = sink_data['Velocities']

        SSx = sink_pos[:,0]
        SSy = sink_pos[:,1]
        SSz = sink_pos[:,2]

        SSx = (SSx - 500.) / 10.
        SSy = (SSy - 500.) / 10.

        Smass = sink_mass
    else:
        data, header = output

    time = header['Time']
    pos= data['Coordinates']
    mass= data['Masses']
    rho = data['Density']
    rho_cgs = rho * udensity    #convert to cgs units
    
    bfield = data['MagneticField']

    #NL97
    xH2, xHp, xCO = data['ChemicalAbundances'].T    #these are abundances of molecular hydrogen, Hplus, and CO relative to the number of protons

    ###### ###### ###### ###### ###### 
    
    xHI = 1 - xHp -2*xH2    #abundance of HI from conservation lawa
    yn = rho_cgs/((1. + 4.0 * xHe) * mp)    #this is the number density ‘n’ that I need for my Crutcher plots

    nHp = xHp*yn
    nH2 = xH2*yn
    nCO = xCO*yn
    nHI = (1.0 - xHp - 2.0*xH2)*yn
    nTOT = yn*(1.0 + xHp - xH2 + xHe)

    mu = rho_cgs/(nTOT*mp)
    energy_per_unit_mass = data['InternalEnergy']*uenergy/umass
    T = (2.0/3.0)*energy_per_unit_mass*mu*mp/kb
    
    vels =  data['Velocities']
    
    chem = data['ChemicalAbundances']

    return mass, pos, rho, bfield, yn, T, vels, chem, time


    
########### ########## ########### ########## PLOTTING FUNCTIONS ########## ########## ########### ##########

def plot_surface_density(Ndensity, SSx, SSy, time, fig, ax):
    """ Plot surface density of a given snapshot - specifically, face-on """
    
    extent = (-10, 10, -10, 10)
    levels = np.logspace(20, 23, 256)
    norm = mc.BoundaryNorm(levels, 256)

    im = ax.imshow(Ndensity, cmap='Reds', norm=norm, extent=extent, origin='lower')

    ax.set_xlabel('x [kpc]', fontsize=20)
    ax.set_ylabel('y [kpc]', fontsize=20)
    ax.set_aspect(1)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    ax.tick_params(labelsize=15)  # Set the tick label size directly on the axis

    textlabel = 'T = ' + str(round(time * uMyr, 2)) + ' Myr'
    ax.text(-9, 9, textlabel, fontsize=20)

    fig.subplots_adjust(right=0.9)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    cbar = plt.colorbar(im, cax=cbar_ax, ticks=[1e+20, 1e+21, 1e+22, 1e+23], format='%.0e')
    cbar.set_label('Column Density [$cm^{-2}$]', size=16)
    cbar.ax.tick_params(labelsize=16)

    ax.scatter(SSx, SSy, s=2, alpha=0.4)

    plt.grid(b=True, which='major', axis='both')


def plot_edge_density(Ndensity_edge, time, ax):
    """Currently unused, requires testing."""
    extent = (-10, 10, -10, 10)  # Adjust the extent as needed
    levels = np.logspace(20, 23, 256)
    norm = mc.BoundaryNorm(levels, 256)

    im = ax.imshow(Ndensity_edge, cmap='Reds', norm=norm, extent=extent, origin='lower')

    ax.set_xlabel('x [kpc]', fontsize=20)
    ax.set_ylabel('y [kpc]', fontsize=20)
    ax.set_aspect(1)

    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    ax.tick_params(labelsize=15)  # Set the tick label size directly on the axis

    textlabel = 'T = ' + str(round(time * uMyr, 2)) + ' Myr'
    ax.text(-9, 4, textlabel, fontsize=20)
    
    cbar = plt.colorbar(im, ax=ax, ticks=[1e+20, 1e+21, 1e+22, 1e+23], format='%.0e')
    cbar.set_label('Column Density [$cm^{-2}$]', size=16)
    cbar.ax.tick_params(labelsize=16)

    plt.grid(b=True, which='major', axis='both')


    
########### ########## ########### ########## GENERAL UTILITY FUNCTIONS ########## ########## ########### ##########

#David's function
def find_cart_cyl_coord(pos): 
  """Converts cartesian to cylindrical coordinates."""
  # cartesian_to_cylindrical
  halfbox = pos[:,0].max() / 2.
  halfboy = pos[:,1].max() / 2.
  halfboz = pos[:,2].max() / 2.

  x = pos[:,0] - halfbox
  y = pos[:,1] - halfboy
  z = pos[:,2] - halfboz
  R = np.sqrt(x**2 + y**2)
  theta = np.arctan2(y, x)

  return x, y, z, R, theta # return r, theta


# radially bin data to study variation with galactic radii
def radially_bin_data(R_min, R_max, step, x, y, z, R_kpc, z_kpc, time, mass, rho_cgs, sink_file_output, xH2, v_r, v_theta, vz, abs_b_field, center=[500,500,500]):
    """Bins input data into galactic radii of users choice.

    Args:
        R_min (arr): lower bounds on radii
        R_max (arr): upper bounds on radii
        step (int): width of radial bin
        x (arr): all cell x positions
        y (arr): all celly positions
        z (arr): all cellz positions
        R_kpc (arr): all cellradial positions
        z_kpc (arr): all cellz positions
        time (float): current time of simulation
        mass (arr): all cell mass values
        rho_cgs (arr): all cell density values (in cgs units)
        sink_file_output (): sink_snap file data
        xH2 (arr): all cell molecular hydrogen abundances
        v_r (arr): all cell radial velocities
        v_theta (arr): all cell theta velocities
        vz (arr): all cell z velocities
        abs_b_field (arr): all cell absolute magnetic field strengths
        centre (array): 3D centre point of the simulation 

    Returns:
        tuple: Plottable data
            ...: ...
            ...: ...
    """
    # make 0 arrays
    plot_r = np.zeros(int(R_max/step))
    plot_avg_rho_cgs = np.zeros(int(R_max/step))
    plot_avg_mol_dens = np.zeros(int(R_max/step))
    plot_young_sinks_num = np.zeros(int(R_max/step))
    plot_avg_KED = np.zeros(int(R_max/step))
    plot_avg_MED = np.zeros(int(R_max/step))
    
    #creates an array binned via radial distance from centre of galaxy where each bin is 1pc wide
    for j in np.arange(R_min,R_max,step):
        k = j+step

        iextract_bins = np.where((np.abs(R_kpc) > j) & (np.abs(R_kpc) < k) & (np.abs(z_kpc) < 0.5))

        mass_bin = mass[iextract_bins] #this is the mass of gas particles within the cut region
        rho_cgs_bin = rho_cgs[iextract_bins]

        ############### molecular gas density ###############
        xH2_bin = xH2[iextract_bins]
        mol_dens_bin = xH2_bin * rho_cgs_bin 
        
        ############### young sinks ###############
        sink_out = sink_file_output[0]
        time_of_formation_myr = sink_out['formationTime'] * uMyr
        sink_age = (time*uMyr) - time_of_formation_myr
        young_sink_mask = np.where(sink_age < 4.)[0]

        sink_pos = sink_out['pos']
        young_sink_pos = sink_pos[young_sink_mask,:]
        
        #get sink pos in kpc
        ysink_x_kpc = (young_sink_pos[:,0] - center[0]) / 10
        ysink_y_kpc = (young_sink_pos[:,1] - center[1]) / 10
        ysink_z_kpc = (young_sink_pos[:,2] - center[2]) / 10

        ysink_R_kpc = (ysink_x_kpc**2 + ysink_y_kpc**2)

        iextract_ysink_bins = np.where((np.abs(ysink_R_kpc) > j) & (np.abs(ysink_R_kpc) < k) & (np.abs(ysink_z_kpc) < 0.5))

        ############### KED and MED ###############
            
        x_ex = x[iextract_bins] - center[0]
        y_ex = y[iextract_bins] - center[1]
        z_ex = z[iextract_bins] - center[2]

        R = R_kpc[iextract_bins]
        #print(j)

        v_cyl_cgs = np.sqrt(((v_r[iextract_bins]**2)* uvel) + ((v_theta[iextract_bins]**2)* uvel) + ((vz[iextract_bins]**2)* uvel)) #work out cylindrical velocities np.sqrt(v_r^2 + v_theta^2 + v_z^2)#work out cylindrical velocities np.sqrt(v_r^2 + v_theta^2 + v_z^2)

        if mass_bin.size == 0:     #for really small bins occasionally one has no data, this sets the data to zero for stacking
            sigma_v_cyl_cgs = 0
            print('no size')
        else:
            sigma_v_cyl_cgs = np.sqrt(np.cov(v_cyl_cgs, aweights=mass_bin)) #this is the mass weighted stdard dev of velocites in cylindrical coords   

        KED_cgs_cut = 0.5 * rho_cgs_bin * (sigma_v_cyl_cgs**2)

        #magnetic
        b_field_bin = abs_b_field[iextract_bins] * umag
        mass_weighted_mag_bin = (np.sum(mass_bin * b_field_bin))/(np.sum(mass_bin))
        mag_energy_dens_cut = (1/(8*np.pi)) * (mass_weighted_mag_bin**2) #CHECK THIS IS ACTUALLY HOW TO CALCULATE MAGNETIC ENERGY?

        plot_r[int(j*(1./step))] = j
        plot_avg_rho_cgs[int(j*(1./step))] = np.mean(rho_cgs_bin)
        plot_avg_mol_dens[int(j*(1./step))] = np.mean(mol_dens_bin)
        plot_young_sinks_num[int(j*(1./step))] = len(young_sink_pos[iextract_ysink_bins])
        plot_avg_KED[int(j*(1./step))] = np.mean(KED_cgs_cut)
        plot_avg_MED[int(j*(1./step))] = np.mean(mag_energy_dens_cut)
        
    return plot_r, plot_avg_rho_cgs, plot_avg_mol_dens, plot_young_sinks_num, plot_avg_KED, plot_avg_MED 
#plasma beta calculation is easier to do in radial binning function than in compute_time_evolution data?? MAybe should be a seperate function? 


def compute_time_evolution_data(base, filenum, cut_R_outer, cut_R_inner, cut_z, snapshot_start, snapshot_end, step, centre=[500,500,500]):
    """function to compute time evolution data 

    Args:
        base (_type_): _description_
        filenum (_type_): _description_
        cut_R_outer (_type_): _description_
        cut_R_inner (_type_): _description_
        cut_z (_type_): _description_
        snapshot_start (_type_): _description_
        snapshot_end (_type_): _description_
        step (_type_): _description_
        centre (list, optional): _description_. Defaults to [500,500,500].

    Returns:
        _type_: _description_
    """
    number_for_loop = (snapshot_end - snapshot_start) // step + 1 # Correct off by 1 error
    print('number for loop is', number_for_loop)
    
    plot_times = np.zeros(number_for_loop)
    
    plot_dense_gas_frac_cut = np.zeros(number_for_loop)
    plot_dense_gas_frac_no_sinks = np.zeros(number_for_loop)
    plot_mol_gas_frac_cut = np.zeros(number_for_loop)
    plot_mol_gas_frac_no_sinks = np.zeros(number_for_loop)
    #plot_plasma_beta = np.zeros(number_for_loop)
    
    j=0
    
    for i in range(snapshot_start, snapshot_end+step, step):
    
        filenum = str(int(i)).zfill(3)
        ABHE=0.1

        file = base + 'snap_' + filenum +'.hdf5'

        rsnap.io_flags['variable_metallicity']=False
        rsnap.io_flags['time_steps']=True
        rsnap.io_flags['mc_tracer']=False
        rsnap.io_flags['sgchem']=True
        rsnap.io_flags['MHD']=True

        output = rsnap.read_snapshot_hdf5(file) 

        sink_file = base + 'sink_snap_' + filenum
        sink_output = rsnap.read_sink_snap(sink_file)

        if len(output) > 2:
            data, header, sink_data = output

            # defining sink properties
            sink_pos = sink_data['Coordinates']
            sink_mass = sink_data['Masses']
            sink_vel = sink_data['Velocities']

            sink_R = np.sqrt( (sink_pos[:,0] - centre[0])**2 + (sink_pos[:,1] - centre[1])**2 )
            sink_z = sink_pos[:,2] - centre[2]

            #defining sink_extract
            sink_extract = np.where((np.abs(sink_R) > cut_R_inner) & (np.abs(sink_R) < cut_R_outer) & (np.abs(sink_z) < cut_z))

            sink_mass_cut = sink_mass[sink_extract]
            #plot_sink_mass_cut[j] = np.sum(sink_mass_cut)

        else:
            data, header = output

        time = header['Time']
        plot_times[j] = time

        #NEW! Sink data from sink_snap files
        sink_data = sink_output[0]
        sink_mass_from_sink_data = sink_data['mass']

        #find young sinks
        time_of_formation_myr = sink_data['formationTime'] * uMyr
        sink_age = (time*uMyr) - time_of_formation_myr
        young_sink_mask = np.where(sink_age < 4.)

        #defining iextract
        pos = data['Coordinates']
        x, y, z, R, theta = find_cart_cyl_coord(pos) #is this slowing down my code?

        iextract = np.where((np.abs(R) > cut_R_inner) & (np.abs(R) < cut_R_outer) & (np.abs(z) < cut_z))

        #mass
        mass = data['Masses']    
        mass_cut = mass[iextract] #this is the mass of gas particles within the cut region

        #define and cut number density to the disc
        rho = data['Density']
        rho_cgs = rho * udensity    #convert to cgs units

        #NL97
        xH2, xHp, xCO = data['ChemicalAbundances'].T    #these are abundances of molecular hydrogen, Hplus, and CO relative to the number of protons
        xHe=0.1
        mp = 1.6726231e-24
        kb = 1.3806485e-16
        xHI = 1 - xHp -2*xH2    #abundance of HI from conservation laws

        #find yn 
        yn = rho_cgs/((1. + 4.0 * xHe) * mp)    #this is the number density ‘n’
        yn_cut = yn[iextract]

        #calculate dense gas frac (without young sink mass)
        dense_gas_mass_cut = mass_cut[yn_cut > 100]
        
        #calculate the molecular gas fraction (without young sink mass)
        xH2_cut = xH2[iextract] 
        mol_gas_frac_cut = np.sum(xH2_cut * mass_cut) / np.sum(mass_cut)  #fraction of mass in the gas phase that is molecular

        #sinks
        young_sink_mass = sink_mass_from_sink_data[young_sink_mask]
        young_sink_dense_mol_gas = young_sink_mass * 0.95

        #calculate dense gas frac with and without sinks
        dense_gas_frac = (np.sum(dense_gas_mass_cut) + np.sum(young_sink_dense_mol_gas)) / (np.sum(mass_cut) + np.sum(young_sink_dense_mol_gas)) #sink_mass_cut * 0.95
        dense_gas_frac_no_sinks = (np.sum(dense_gas_mass_cut)) / (np.sum(mass_cut))
        
        #calculate mol gas frac
        mol_gas_frac = (np.sum(young_sink_dense_mol_gas)) / (np.sum(mass_cut)) + mol_gas_frac_cut # fraction of mass in the sink phase that is molecular + fraction of mass in the gas phase that is molecular
        
        ###############################
        '''
        v_cyl_cgs = np.sqrt(((v_r[iextract_bins]**2)* uvel) + ((v_theta[iextract_bins]**2)* uvel) + ((vz[iextract_bins]**2)* uvel)) #work out cylindrical velocities np.sqrt(v_r^2 + v_theta^2 + v_z^2)
    
        if mass_bin.size == 0:     #for really small bins occasionally one has no data, this sets the data to zero
            sigma_v_cyl_cgs = 0
            print('no size')
        else:
            sigma_v_cyl_cgs = np.sqrt(np.cov(v_cyl_cgs, aweights=mass_bin)) #this is the mass weighted stdard dev of velocites in cylindrical coords   

        KED_cgs_cut = 0.5 * rho_cgs_bin * (sigma_v_cyl_cgs**2)

        #magnetic
        b_field_bin = abs_b_field[iextract_bins] * umag
        mass_weighted_mag_bin = (np.sum(mass_bin * b_field_bin))/(np.sum(mass_bin))
        mag_energy_dens_cut = (1/(8*np.pi)) * (mass_weighted_mag_bin**2) #CHECK THIS IS ACTUALLY HOW TO CALCULATE MAGNETIC ENERGY?
        '''
        ###############################
        
        plot_dense_gas_frac_cut[j] = dense_gas_frac
        plot_dense_gas_frac_no_sinks[j] = dense_gas_frac_no_sinks
        plot_mol_gas_frac_cut[j] = mol_gas_frac
        plot_mol_gas_frac_no_sinks[j] = mol_gas_frac_cut
        #plot_plasma_beta[j] = plot_avg_KED / plot_avg_MED
        j = j +1

        print(i)
    
    return plot_times, plot_dense_gas_frac_cut, plot_dense_gas_frac_no_sinks, plot_mol_gas_frac_cut, plot_mol_gas_frac_no_sinks#, plot_plasma_beta


#function for calculating galaxy vertical extent in the z plane
def calculate_vertical_extent(pos, mass, chem, cut_R_inner, cut_R_outer, cut_z):
    # Extract relevant data based on cylindrical coordinates
    x, y, z, R, theta = find_cart_cyl_coord(pos)
    iextract = np.where((np.abs(R) > cut_R_inner) & (np.abs(R) < cut_R_outer) & (np.abs(z) < cut_z))

    mass_cut = mass[iextract]

    z_pos = pos[:,2]
    z_cut = z_pos[iextract]
    z_cut_kpc = (z_cut - 500) / 10.
    #print('Mean z position value; ' + str(np.mean(z_cut_kpc)) + ' kpc')

    # Calculate mean and weighted standard deviation
    mass_weighted_mean = np.average(z_cut_kpc, weights=mass_cut)
    #print('Mass weighted mean; ' + str(mass_weighted_mean) + ' kpc')
    
    squared_diff = np.sum(mass_cut * (z_cut_kpc - mass_weighted_mean)**2)
    mass_weighted_std_dev = np.sqrt(squared_diff / np.sum(mass_cut))
    #print('Mass weighted std dev; ' + str(mass_weighted_std_dev) + ' kpc')
    
    # Now do the same for HI and HII #
    xH2, xHp, xCO = chem.T
    
    xHp_cut = xHp[iextract]
    xH2_cut = xH2[iextract]
    xHI_cut = 1 - xHp_cut -2*xH2_cut

    mass_xH2_cut = mass_cut * xH2_cut
    mass_xHI_cut = mass_cut * xHI_cut
    
    ############ HI ############ 

    mass_weighted_HI_mean = np.average(z_cut_kpc, weights=mass_xHI_cut)
    #print('HI mass weighted mean; ' + str(mass_weighted_HI_mean) + ' kpc')

    squared_diff = np.sum(mass_xHI_cut * (z_cut_kpc - mass_weighted_HI_mean)**2)
    mass_weighted_HI_std_dev = np.sqrt(squared_diff / np.sum(mass_xHI_cut))
    #print('HI mass weighted std dev; ' + str(mass_weighted_HI_std_dev) + ' kpc')

    ############ H2 ############ 

    mass_weighted_H2_mean = np.average(z_cut_kpc, weights=mass_xH2_cut)
    #print('H2 mass weighted mean; ' + str(mass_weighted_H2_mean) + ' kpc')

    squared_diff = np.sum(mass_xH2_cut * (z_cut_kpc - mass_weighted_H2_mean)**2)
    mass_weighted_H2_std_dev = np.sqrt(squared_diff / np.sum(mass_xH2_cut))
    #print('H2 mass weighted std dev; ' + str(mass_weighted_H2_std_dev) + ' kpc')
    
    #return results
    return np.mean(z_cut_kpc), mass_weighted_mean, mass_weighted_std_dev, mass_weighted_HI_mean, mass_weighted_HI_std_dev, mass_weighted_H2_mean, mass_weighted_H2_std_dev


#function used to find net mass increase of sinks 
def net_increase(id0, id1, mass0, mass1):
    """Calculates the net increase in mass from sink data arrays.
    Args:
        id0 np.array: Integer ids for each sink particle at time 0.
        id1 np.array: Integer ids for each sink particle at time 1.
        mass0 np.array: Array of masses of sinks at time 0.
        mass1 np.array: Array of masses of sinks at time 1.
    """
    matching_ids = [
        np.isin(id0, id1),
        np.isin(id1, id0)
    ]
    
    matching_sorted_ids = [
        np.argsort(id0[matching_ids[0]]),
        np.argsort(id1[matching_ids[1]])
    ]
    
    changed_mass_of_t0_sinks = mass1[matching_ids[1]][matching_sorted_ids[1]] - mass0[matching_ids[0]][matching_sorted_ids[0]]
    sink_pos_mass_change = changed_mass_of_t0_sinks[changed_mass_of_t0_sinks>0] #just want the positive increase i.e star FORMAITON for SFR, irrespective of any mass loss. 
    
    new = mass1[np.invert(matching_ids[1])]
    
    total_increase = new.sum() + sink_pos_mass_change.sum()
    
    return total_increase


def net_increase_with_positions(id0, id1, mass0, mass1, pos0, pos1):
    """Calculates the net increase in sink mass and returns positions of those sinks.
    Args:
        id0 np.array: Integer ids for each sink particle at time 0.
        id1 np.array: Integer ids for each sink particle at time 1.
        mass0 np.array: Array of masses of sinks at time 0.
        mass1 np.array: Array of masses of sinks at time 1.
        pos0 np.array: Array of positions of sinks at time 0.
        pos1 np.array: Array of positions of sinks at time 1.
    Returns:
        new_sinks_mass: Mass of 'new' sinks, i.e sinks that are present at time1 that were not present at time0
        accreted_sinks_mass: Mass of 'accreted' sinks, i.e sinks there were present at time0 and are present at time1 that have had a positive mass increase.
        new_sinks_positions: Positions (x, y, z) of the newly formed sink particles.
        accreted_sinks_positions: Positions (x, y, z) of the old particles that have accreted mass.
    """
    matching_ids = [
        np.isin(id0, id1),
        np.isin(id1, id0)
    ]
    
    matching_sorted_ids = [
        np.argsort(id0[matching_ids[0]]),
        np.argsort(id1[matching_ids[1]])
    ]
    
    changed_mass_of_t0_sinks = mass1[matching_ids[1]][matching_sorted_ids[1]] - mass0[matching_ids[0]][matching_sorted_ids[0]]
    accreted_sinks_mass = changed_mass_of_t0_sinks[changed_mass_of_t0_sinks>0] #just want the positive increase i.e star FORMAITON for SFR, irrespective of any mass loss. 
    
    new_sinks_mass = mass1[np.invert(matching_ids[1])]
    new_sinks_positions = pos1[np.invert(matching_ids[1])]
    
    positive_mass_change_indices = np.where(changed_mass_of_t0_sinks > 0) #used to get positions of only positive mass increase
    #accreted_sinks_positions = pos0[matching_ids[0]][matching_sorted_ids[0]]
    accreted_sinks_positions = pos0[positive_mass_change_indices]
    
    return new_sinks_mass, accreted_sinks_mass, new_sinks_positions, accreted_sinks_positions


#function used to calculate star formation rate and produce arrays to plot it over time
def generate_SFR_data(base, snapshot_start, snapshot_end, step, SFE=0.05, center=[500,500,500]):
   
    #disc region to cut data to
    cut_R_outer = 100 #UNIT IS 100pc - so 100 = 10kpc
    cut_R_inner = 0 #
    cut_z = 5 # = 0.5kpc
    
    # Number of samples
    number_for_loop = (snapshot_end - snapshot_start) // step + 1 # Correct off by 1 error

    # Instead of using plot_times and plot_SFR, we'll create new arrays specific to each dataset
    dataset_plot_times = np.zeros(number_for_loop)
    dataset_plot_SFR = np.zeros(number_for_loop)

    j = 0

    # Include the last entry by going to end + step
    for i in range(snapshot_start, snapshot_end+step, step):
        filenum = str(int(i)).zfill(3)

        f_t0 = base + 'snap_' + filenum + '.hdf5'
        print(f_t0)

        output = rsnap.read_snapshot_hdf5(f_t0)

        if len(output) > 2:

            #data load in for time 1 
            data_t0, header_t0, sink_data_t0 = output

            sink_IDs_t0 = sink_data_t0['ParticleIDs']
            sink_mass_t0 = sink_data_t0['Masses'] * SFE
            sink_pos_t0 = sink_data_t0['Coordinates']

            sink_R_t0 = np.sqrt( (sink_pos_t0[:,0] - center[0])**2 + (sink_pos_t0[:,1] - center[1])**2 )
            sink_z_t0 = sink_pos_t0[:,2] - 500

            time_t0 = header_t0['Time'] * uMyr

            # load sink masses for time 1 and multiply by SFE. Also load in all sink IDs
            t1 = i + 1
            filenum_t1 = str(t1).zfill(3)
            f_t1 = base + 'snap_' + filenum_t1 + '.hdf5'

            output = rsnap.read_snapshot_hdf5(f_t1)
            data_t1, header_t1, sink_data_t1 = output

            sink_IDs_t1 = sink_data_t1['ParticleIDs']
            sink_mass_t1 = sink_data_t1['Masses'] * SFE
            sink_pos_t1 = sink_data_t1['Coordinates']

            sink_R_t1 = np.sqrt( (sink_pos_t1[:,0] - center[0])**2 + (sink_pos_t1[:,1] - center[1])**2 )
            sink_z_t1 = sink_pos_t1[:,2] - 500

            time_t1 = header_t1['Time'] * uMyr
            dataset_plot_times[j] = time_t1

            #defining sink_extract_t0
            sink_extract_t0 = np.where((np.abs(sink_R_t0) > cut_R_inner) & (np.abs(sink_R_t0) < cut_R_outer) & (np.abs(sink_z_t0) < cut_z))
            sink_IDs_cut_t0 = sink_IDs_t0[sink_extract_t0]
            sink_mass_cut_t0 = sink_mass_t0[sink_extract_t0]

            #defining sink_extract_t1
            sink_extract_t1 = np.where((np.abs(sink_R_t1) > cut_R_inner) & (np.abs(sink_R_t1) < cut_R_outer) & (np.abs(sink_z_t1) < cut_z))
            sink_IDs_cut_t1 = sink_IDs_t1[sink_extract_t1]
            sink_mass_cut_t1 = sink_mass_t1[sink_extract_t1]


            ###################################################################################################
            # identify new IDs in time 1 that aren't present in time 0. then find the mass for all new IDs.

            net = net_increase(sink_IDs_cut_t0, sink_IDs_cut_t1, sink_mass_cut_t0, sink_mass_cut_t1)

            time_diff = time_t1 - time_t0 # Find difference in time betwen two snapshots
            dataset_plot_SFR[j] = net / (time_diff * 1e6) # Divide the mass by this time to give SFR in solar masses per year. (1e6 to convertfrom Myr to yr)

        else:
            print('no sinks for snap', i)

        j += 1

    return dataset_plot_times, dataset_plot_SFR


def calculate_sfr_with_pos(base, number, young_sinks_only=False, sink_age_threshold=4, SFE=0.05, center=[500,500,500], cut_R_inner=0, cut_R_outer=100, cut_z=5, **kwargs):

    t0 = number
    filenum_t0 = str(t0).zfill(3)
    
    f_t0 = base + 'snap_' + str(filenum_t0) + '.hdf5'
    print(f_t0)

    output_t0 = rsnap.read_snapshot_hdf5(f_t0)
    if len(output_t0) > 2:

        #data load in for time 0
        data_t0, header_t0, sink_data_t0 = output_t0

        sink_IDs_t0 = sink_data_t0['ParticleIDs']
        sink_mass_t0 = sink_data_t0['Masses'] * SFE
        sink_pos_t0 = sink_data_t0['Coordinates']

        sink_R_t0 = np.sqrt( (sink_pos_t0[:,0] - center[0])**2 + (sink_pos_t0[:,1] - center[1])**2 ) / 10
        sink_z_t0 = sink_pos_t0[:,2] - 500

        time_t0 = header_t0['Time'] * uMyr * 1e6

        # load sink masses for time 1 and multiply by SFE. Also load in all sink IDs
        t1 = number + 1
        filenum_t1 = str(t1).zfill(3)
        f_t1 = base + 'snap_' + str(filenum_t1) + '.hdf5'

        output_t1 = rsnap.read_snapshot_hdf5(f_t1)
        data_t1, header_t1, sink_data_t1 = output_t1

        sink_IDs_t1 = sink_data_t1['ParticleIDs']
        sink_mass_t1 = sink_data_t1['Masses'] * SFE
        sink_pos_t1 = sink_data_t1['Coordinates']

        sink_R_t1 = np.sqrt( (sink_pos_t1[:,0] - center[0])**2 + (sink_pos_t1[:,1] - center[1])**2 )
        sink_z_t1 = sink_pos_t1[:,2] - 500

        time_t1 = header_t1['Time'] * uMyr * 1e6
        #plot_times[j] = time_t1

        #defining sink_extract_t0
        sink_extract_t0 = np.where((np.abs(sink_R_t0) > cut_R_inner) & (np.abs(sink_R_t0) < cut_R_outer) & (np.abs(sink_z_t0) < cut_z))
        sink_IDs_cut_t0 = sink_IDs_t0[sink_extract_t0]
        sink_mass_cut_t0 = sink_mass_t0[sink_extract_t0]
        sink_pos_cut_t0 = sink_pos_t0[sink_extract_t0]
        
        #option to only include young sinks t0
        if young_sinks_only == True: 
            sink_file_t0 = base + 'sink_snap_' + filenum_t0
            print(sink_file_t0)
            sink_output_t1 = rsnap.read_sink_snap(sink_file_t0)
            sink_out_t0 = sink_output_t0[0]
            
            time_of_formation_myr_t0 = sink_out_0['formationTime'] * uMyr
            sink_age_t0 = (time_t0*uMyr) - time_of_formation_myr_t0
            young_sink_mask_t0 = np.where(sink_age_t0 < sink_age_threshold)[0]
            
            sink_IDs_cut_t0 = sink_IDs_t0[young_sink_mask_t0]
            sink_mass_cut_t0 = sink_mass_t0[young_sink_mask_t0]
            sink_pos_cut_t0 = sink_pos_t0[young_sink_mask_t0]

        #defining sink_extract_t1
        sink_extract_t1 = np.where((np.abs(sink_R_t1) > cut_R_inner) & (np.abs(sink_R_t1) < cut_R_outer) & (np.abs(sink_z_t1) < cut_z))
        sink_IDs_cut_t1 = sink_IDs_t1[sink_extract_t1]
        sink_mass_cut_t1 = sink_mass_t1[sink_extract_t1]
        sink_pos_cut_t1 = sink_pos_t1[sink_extract_t1]
        
        #option to only include young sinks t1
        if young_sinks_only == True: 
            sink_file_t1 = base + 'sink_snap_' + filenum_t1
            print(sink_file_t1)
            sink_output_t1 = rsnap.read_sink_snap(sink_file_t1)
            sink_out_t1 = sink_output_t1[0]
            
            time_of_formation_myr_t1 = sink_out_1['formationTime'] * uMyr
            sink_age_t1 = (time_t1*uMyr) - time_of_formation_myr_t1
            young_sink_mask_t1 = np.where(sink_age_t1 < sink_age_threshold)[0]
            
            sink_IDs_cut_t1 = sink_IDs_t1[young_sink_mask_t1]
            sink_mass_cut_t1 = sink_mass_t1[young_sink_mask_t1]
            sink_pos_cut_t1 = sink_pos_t1[young_sink_mask_t1]

        #net = net_increase(sink_IDs_cut_t0, sink_IDs_cut_t1, sink_mass_cut_t0, sink_mass_cut_t1)
        new_sinks_mass, accreted_sinks_mass, new_sinks_positions, accreted_sinks_positions = net_increase_with_positions(sink_IDs_cut_t0, sink_IDs_cut_t1,
                                                                                              sink_mass_cut_t0, sink_mass_cut_t1, sink_pos_cut_t0, sink_pos_cut_t1)
        
        # Calculate SFR at each sink particle's position
        time_elapsed = time_t1 - time_t0
        SFR_new_sinks = new_sinks_mass / time_elapsed
        SFR_accreted_sinks = accreted_sinks_mass / time_elapsed

        #convert positions into kpc space
        kpc_new_sinks = (new_sinks_positions - 500) / 10.
        kpc_accreted_sinks = (accreted_sinks_positions - 500) / 10.

        # Combine positions and SFR values for new and accreted sinks into a 2D array
        SFR_pos_new = np.column_stack((SFR_new_sinks, kpc_new_sinks))
        SFR_pos_accreted = np.column_stack((SFR_accreted_sinks, kpc_accreted_sinks))

        # Combine the two sets of positions and SFR values
        SFR_pos_combined = np.vstack((SFR_pos_new, SFR_pos_accreted))
        
        return SFR_pos_combined
    
    else:
        print('No stars formed yet! Choose a later snapshot')

        
'''
#useful illustration with test data of how the stacking works

SFR_1 = 5
kpc_1 = np.array([[1, 2, 3]])

SFR_2 = 10
kpc_2 = np.array([[7, 8, 9]])

# Combine positions and SFR values for new and accreted sinks into a 2D array
SFR_pos_new = np.column_stack((SFR_1, kpc_1))
SFR_pos_accreted = np.column_stack((SFR_2, kpc_2))

# Combine the two sets of positions and SFR values
SFR_pos_combined = np.vstack((SFR_pos_new, SFR_pos_accreted))
print(SFR_pos_combined)
'''

def calculate_sfr_surface_density(SFR_with_pos, x_bins, y_bins):
    
    # Define positions and SFR values 
    x_SFR = SFR_with_pos[:,1]        #position of SFR
    y_SFR = SFR_with_pos[:,2]
    SFR = SFR_with_pos[:,0]      #SFR at given x,y
    
    # Calculate 2D histogram for SFR
    SFR_histogram, xedges, yedges = np.histogram2d(x_SFR, y_SFR, bins=[x_bins, y_bins], weights=SFR)
    
    # Calculate bin areas
    bin_areas = np.diff(xedges)[:, np.newaxis] * np.diff(yedges)[np.newaxis, :]
    
    # Calculate SFR surface density
    SFR_surface_density = SFR_histogram / bin_areas
    
    return SFR_surface_density


def calculate_gas_surface_density(pos_kpc, mass, x_bins, y_bins): #bins should also be in kpc

    # Calculate 2D histogram for gas density
    gas_histogram, xedges, yedges = np.histogram2d(pos_kpc[:, 0], pos_kpc[:, 1], bins=[x_bins, y_bins], weights=mass)

    # Calculate bin areas
    bin_areas = np.diff(xedges)[:, np.newaxis] * np.diff(yedges)[np.newaxis, :]

    # Calculate gas surface density
    gas_surface_density = gas_histogram / (bin_areas * 10**6) #get gas SD in Msolpc^-2
    
    return gas_surface_density 


def filter_zero_values(gas_surface_density, SFR_surface_density):
    
    # Print initial lengths of the arrays
    #print("Initial length of HD gas surface density array:", len(gas_surface_density.flatten()))
    #print("Initial length of HD SFR surface density array:", len(SFR_surface_density.flatten()))

    # Check if there are any zeros in the gas surface density array
    if 0 in gas_surface_density:
        print("Zeros detected in gas surface density.")
    else:
        print("No zeros detected in gas surface density.")

    # Check if there are any zeros in the SFR surface density array
    if 0 in SFR_surface_density:
        print("Zeros detected in SFR surface density.")
    else:
        print("No zeros detected in SFR surface density.")

    # If there are zeros in either array, proceed with filtering
    if 0 in gas_surface_density or 0 in SFR_surface_density:
        print("Initiating filtering.")
        # Flatten the arrays
        flat_gas_surface_density = gas_surface_density.flatten()
        flat_SFR_surface_density = SFR_surface_density.flatten()

        # Remove values from both flattened arrays if either one of them is zero
        filtered_pairs = [(gas, SFR) for gas, SFR in zip(flat_gas_surface_density, flat_SFR_surface_density) if gas != 0 and SFR != 0]

        # Check if any valid pairs were found after filtering
        if filtered_pairs:
            # Unzip the filtered pairs
            filtered_gas_surface_density, filtered_SFR_surface_density = zip(*filtered_pairs)
            print("Filtered pairs created successfully.")
        else:
            print("No valid pairs found after filtering.")
            return None, None
    else:
        print("Skipping filtering. Both HD gas surface density and HD SFR surface density arrays contain no zeros.")
        return gas_surface_density, SFR_surface_density
        
    #print("Length of HD gas surface density array after filtering:", len(filtered_gas_surface_density))
    #print("Length of HD SFR surface density array after filtering:", len(filtered_SFR_surface_density))

    return filtered_gas_surface_density, filtered_SFR_surface_density