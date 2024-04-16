import matplotlib.pyplot as pl
import matplotlib.colors as mc
import matplotlib.ticker as ticker
import numpy as np
from numpy import sqrt, log, exp, arccosh, sinh, cosh, tanh, pi, cos, sin, arctan2, arccos, array
from astropy.io import fits

def fmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    if (int(a)==1):
        return r'$10^{{{}}}$'.format(b)
    else:
        return r'${} \times 10^{{{}}}$'.format(a, b)

pl.rcParams['text.usetex'] = True

#############################
# read the (x,y,v) datacube
#############################

hdu  = fits.open('datacube.fits')[0]

vmin, dv, nv = hdu.header['CRVAL1'], hdu.header['CDELT1'], hdu.header['NAXIS1']
ymin, dy, ny = hdu.header['CRVAL2'], hdu.header['CDELT2'], hdu.header['NAXIS2']
xmin, dx, nx = hdu.header['CRVAL3'], hdu.header['CDELT3'], hdu.header['NAXIS3']

vs = vmin + np.arange(nv)*dv
ys = ymin + np.arange(ny)*dy
xs = xmin + np.arange(nx)*dx

X,Y,VZ = np.meshgrid(xs,ys,vs)

#############################
# find projections
#############################

datacube = hdu.data
xy_CII = datacube.sum(axis=2).T / (dx*100*dy*100 * 3.086e16**2)
xy_v   = np.ma.average(VZ,weights=datacube,axis=2).T

###################
# plot 
###################

fig, axarr = pl.subplots(1,2,figsize=(10,6))
ax1 = axarr[0]
ax2 = axarr[1]

extent = (xs.min(),xs.max(),ys.min(),ys.max())  

# plot CII intensity
levels = np.logspace(-1,4,256)
norm = mc.BoundaryNorm(levels,256)
im1 = ax1.imshow(xy_CII,norm=norm,extent=extent,cmap='Greys',interpolation='nearest',origin='l')

# plot vr
levels = np.linspace(-100,100,201)
norm  = mc.BoundaryNorm(levels,256)
im2 = ax2.imshow(xy_v,norm=norm,extent=extent,cmap='bwr',interpolation='nearest',origin='l')

# formatting
ax1.grid(ls='dashed')
ax2.grid(ls='dashed')
ax1.set_aspect('1')
ax2.set_aspect('1')
ax1.set_xlabel(r'$x\ [\rm{kpc}]$',fontsize=20)
ax1.set_ylabel(r'$y\ [\rm{kpc}]$',fontsize=20)
ax2.set_xlabel(r'$x\ [\rm{kpc}]$',fontsize=20)
ax2.set_ylabel(r'$y\ [\rm{kpc}]$',fontsize=20)
ax1.set_xlim(xs.min(),xs.max())
ax1.set_ylim(ys.min(),ys.max())
ax2.set_xlim(xs.min(),xs.max())
ax2.set_ylim(ys.min(),ys.max())
ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)

ax1.set_title(r'$\rm CII \, emission$',fontsize=18)
ax2.set_title(r'$\rm v$',fontsize=18)

cbarticks = np.logspace(0,10,11)
cbar1 = fig.colorbar(im1,ax=ax1,ticks=cbarticks, format=ticker.FuncFormatter(fmt),shrink=0.6)
cbar1.set_label(r'$[\rm erg \, s^{-1} \, sr^{-1}]$',fontsize=14,y=1.08,labelpad=-18,rotation=0)

cbarticks = np.linspace(-200,200,11)
cbar2 = fig.colorbar(im2,ax=ax2,ticks=cbarticks, format='%g',shrink=0.6)
cbar2.set_label(r'$[\rm km/s]$',fontsize=14,y=1.08,labelpad=-18,rotation=0)

# final things
pl.tight_layout()
fig.savefig('datacube.png',bbox_inches='tight',dpi=300)
pl.close('all')


