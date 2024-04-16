import numpy as np
import matplotlib.pyplot as plt
from arepy.read_write import binary_read as rsnap
from arepy.utility import cgs_constants as cgs
import matplotlib.colors as mc
from scipy.interpolate import NearestNDInterpolator
from arepy.utility import snap_utility as snut
from arepy.utility.ut_utility import *

def plot_density_proj(number):
  filenum = str(number).zfill(3)
  ABHE = 0.1

  ColumnDensity = rsnap.read_image('density_proj_' + filenum)
  xHp = rsnap.read_image('xHP_proj_' + filenum)
  xH2 = rsnap.read_image('xH2_proj_' + filenum)
  xCO = rsnap.read_image('xCO_proj_' + filenum)

  NH = (ColumnDensity * rsnap.arepoColumnDensity) / ((1.+4.*ABHE)*cgs.mp)

  Ndensity = NH * (1. + ABHE - xH2 + xHp)

  plt.imshow(np.log10(Ndensity), origin='l')
  plt.show()

def plot_arepo_proj_binning(x, y, quantity, dynamical_range = 8):

  x = x.reshape(x.size)
  y = y.reshape(y.size)
  quantity = quantity.reshape(quantity.size)
  extent = [x.min(),x.max(),y.min(),y.max()]
  nx = 1000
  ny = 1000
  w = quantity
  datacube = np.histogram2d(x, y, weights=w, bins = (nx, ny))[0]
  levels = np.logspace(np.log10(datacube.max())-dynamical_range, np.log10(datacube.max()), 256)
  norm = mc.BoundaryNorm(levels,256)
  datacube = np.rot90(datacube)
  plt.imshow(datacube, extent=extent, norm=norm)
  plt.show()

def return_slice(x, y, z, quantity, x0, x1, y0, y1, z0, npixx = 1001, npixy = 1001):
  # Create interpolation function for quantity
  points = np.vstack((x,y,z)).T
  quantityf = NearestNDInterpolator(points,quantity)

  #produce slice
  xgrid, ygrid, zgrid = np.linspace(x0, x1, npixx), np.linspace(y0, y1, npixy), z0
  XGRID,YGRID,ZGRID = np.meshgrid(xgrid,ygrid,zgrid)
  quantity_GRID = quantityf(XGRID,YGRID,ZGRID)

  return quantity_GRID[:,:,0]

def plot_pdf(data, region=None, weighting='mass'):
  if (region == None):
    region = np.arange(data['rho'].size)

  fig, ax = plt.subplots(1,2,figsize=(10,6))
  n = find_number_density(data)[0]
  temp = find_temperature(data)
  if (weighting == 'mass'):
    weight = data['mass'][region] / data['mass'][region].sum()
  else:
    if (weighting == 'volume'):
      vol = data['mass'][region] / data['rho'][region]
      weight = vol / vol.sum()
  bins = int(np.sqrt(region.size))

  ax[0].hist(np.log10(n[region]),bins=bins, weights=weight, histtype='step', log=True, lw=1, color='k')
  ax[1].hist(np.log10(temp[region]),bins=bins, weights=weight, histtype='step', log=True, lw=1, color='k')

  ax[0].set_xlim(-4,5)
  ax[1].set_xlim(0,6)

  ax[0].set_ylim(1e-6,1e-1)
  ax[1].set_ylim(1e-6,1e-1)

  if (weighting == 'mass'):
    ax[0].set_ylabel(r'$\mathrm{d}M/M_{\rm tot}$',fontsize=12)
  else:
    if (weighting == 'volume'):
      ax[0].set_ylabel(r'$\mathrm{d}V/V_{\rm tot}$',fontsize=12)
  
  ax[0].set_xlabel(r'$\log_{10}(n) \; \rm [cm^{-3}]$',fontsize=12)
  ax[1].set_xlabel(r'$\log_{10}(T) \; \rm [K]$',fontsize=12)

  plt.show()
  
  return

def patches_select(data, region):
    x, y, rho = data['pos'][region,0], data['pos'][region,1], data['rho'][region]
    x = x.reshape(x.size)
    y = y.reshape(y.size)

    plot_arepo_proj_binning(x, y, rho)

    rr = np.array(plt.ginput(n=0,timeout=0)).T

    xpath = rr[0]
    ypath = rr[1]
    Polygon = CreatePolygon(xpath,ypath)

    points = np.array(list(zip(x,y)))
    points_inside = Polygon.contains_points(points)
    ind = np.array(np.where(points_inside))

    plot_arepo_proj_binning(x[ind], y[ind], rho[ind])

    region = np.array(region).reshape(np.size(region))
    return region[ind[0,:]]
 
