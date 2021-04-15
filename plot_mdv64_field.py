#!/usr/local/env python

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, NullFormatter, ScalarFormatter

import metpy.calc as mpcalc
import metpy.plots as mpplt
from metpy.units import units
from metpy.plots import add_metpy_logo, SkewT

import icing_funcs as icing

import shapely.geometry as sgeom

import datetime, os, math

import pandas as pd

#f = '/d1/arugg/data/mdv/model/hrrrv4/hybrid/tile4/20190227/g_000000/f_00010800.mdv'
#f = '/var/autofs/mnt/horus_d1/fip-test/data/versionB/mdv/model-hrrr/pressure_derived/conus_TMP/20190201/g_000000/f_00010800.mdv'

fileData = xr.open_dataset('/d1/dadriaan/data/test/ral-ifi-research/mdv64_test/data/num25_hrrr/data/mdv/fip/ice_cat/conus/20190201/g_000000/20190201_g_000000_f_00010800.mdv.cf.nc')
print(fileData)

fileData = fileData.squeeze()
print(fileData)
#print(fileData.time_bounds)
#print(fileData.grid_mapping_0)
fileData = fileData.drop(['time_bounds','mdv_chunk_0000'])
print(fileData)

print(fileData.lat0.min().values)
print(fileData.lat0.max().values)
print(fileData.lon0.min().values)
print(fileData.lon0.max().values)

# Assign a new variable for remapping the categorical severity values
fileData['NEWCAT'] = xr.zeros_like(fileData.ICE_SEV_CAT)

# Masks for categories
fourmask = ((fileData.ICE_SEV_CAT>3.5) & (fileData.ICE_SEV_CAT<4.5))
onemask = ((fileData.ICE_SEV_CAT>0.5) & (fileData.ICE_SEV_CAT<1.5))
twomask = ((fileData.ICE_SEV_CAT>1.5) & (fileData.ICE_SEV_CAT<2.5))
fivemask = ((fileData.ICE_SEV_CAT>4.5))

# Assign values using the mask
fileData['NEWCAT'] = xr.where(fourmask,1.0,fileData['NEWCAT'])
fileData['NEWCAT'] = xr.where(onemask,2.0,fileData['NEWCAT'])
fileData['NEWCAT'] = xr.where(twomask,3.0,fileData['NEWCAT'])
fileData['NEWCAT'] = xr.where(fivemask,4.0,fileData['NEWCAT'])

# Set 0.0 sevcat, prob, sld to NaN for plotting
fileData['ICE_PROB'] = fileData['ICE_PROB'].where(fileData['ICE_PROB']>0.0)
fileData['SLD'] = fileData['SLD'].where(fileData['SLD']!=0.0)
fileData['NEWCAT'] = fileData['NEWCAT'].where(fileData['NEWCAT']>0.0)

sevcat_cols = [(204/255, 255/255, 255/255),
               (153/255, 204/255, 255/255),
               (102/255, 153/255, 255/255),
               (51/255, 51/255, 255/255)]
icatcols = ListedColormap(sevcat_cols,N=4)
icatnorm = BoundaryNorm([0.5,1.5,2.5,3.5,4.5],ncolors=5,clip=True)

sld_cols = [(204/255, 204/255, 204/255),
            (204/255, 255/255, 255/255),
            (153/255, 255/255, 204/255),
            (153/255, 255/255, 102/255),
            (204/255, 255/255, 102/255),
            (255/255, 255/255, 0/255),
            (255/255, 204/255, 0/255),
            (255/255, 153/255, 0/255),
            (255/255, 102/255, 0/255),
            (255/255, 51/255, 0/255),
            (204/255, 0/255, 0/255)]
sldcols = ListedColormap(sld_cols,N=11)
sldnorm = BoundaryNorm([-0.1,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],ncolors=12,clip=True)

iprob_cols = [(204/255, 255/255, 255/255),
              (153/255, 255/255, 204/255),
              (153/255, 255/255, 102/255),
              (204/255, 255/255, 102/255),
              (255/255, 255/255, 0/255),
              (255/255, 204/255, 0/255),
              (255/255, 153/255, 0/255),
              (255/255, 102/255, 0/255),
              (255/255, 51/255, 0/255)]
iprobcols = ListedColormap(iprob_cols,N=9)
iprobnorm = BoundaryNorm([0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85],ncolors=9,clip=True)

print(fileData.forecast_period)
print(fileData.forecast_reference_time)
print(fileData.time)

tstring = "INIT: "+fileData.forecast_reference_time.comment+" "+"VLD: "+fileData.time.comment
print(tstring)

# Try and plot
fig = plt.figure(1,figsize=(22,15))
ax1 = plt.subplot(111,projection=ccrs.LambertConformal(central_longitude=-97.5,central_latitude=38.5))
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
ax1.add_feature(cfeature.STATES, linewidth=0.5)
ax1.add_feature(cfeature.BORDERS, linewidth=0.5)

p1 = ax1.pcolormesh(fileData.lon0,fileData.lat0,fileData['ICE_PROB'].max(dim='z0').where(fileData['ICE_PROB'].max(dim='z0')>0.0),transform=ccrs.PlateCarree(),cmap=iprobcols,norm=iprobnorm)
#p1 = ax1.contourf(fileData.lon0,fileData.lat0,fileData['ICE_PROB'].max(dim='z0').where(fileData['ICE_PROB'].max(dim='z0')>0.0),transform=ccrs.PlateCarree(),levels=[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85],cmap=iprobcols)
cb1 = plt.colorbar(p1, ax=ax1, orientation='horizontal', shrink=0.74, pad=0.05)
ax1.set_title(tstring,fontsize=16,loc='left')
ax1.set_title('ICING PROBABILITY COMPOSITE\nNCAR_TEST_ID: '+fileData.NCAR_TEST_ID.strip(),fontsize=16,loc='right')
cb1.set_label('%', size='x-large')
fig.savefig('mdv64_ICE_PROB.png')

ax2 = plt.subplot(111,projection=ccrs.LambertConformal(central_longitude=-97.5,central_latitude=38.5),label='sld')
ax2.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
ax2.add_feature(cfeature.STATES, linewidth=0.5)
ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
p2 = ax2.pcolormesh(fileData.lon0,fileData.lat0,fileData['SLD'].max(dim='z0').where(fileData['SLD'].max(dim='z0')>-0.15),transform=ccrs.PlateCarree(),cmap=sldcols, norm=sldnorm)
#p2 = ax2.contourf(fileData.lon0,fileData.lat0,fileData['SLD'].max(dim='z0').where(fileData['SLD'].max(dim='z0')>0.0),transform=ccrs.PlateCarree(),cmap=sldcols,levels=[-0.05,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
cb2 = plt.colorbar(p2, ax=ax2, orientation='horizontal', shrink=0.74, pad=0.05)
ax2.set_title(tstring,fontsize=16,loc='left')
ax2.set_title('SLD POTENTIAL COMPOSITE\nNCAR_TEST_ID: '+fileData.NCAR_TEST_ID.strip(),fontsize=16,loc='right')
cb2.set_label('POT', size='x-large')
fig.savefig('mdv64_SLD.png')

ax3 = plt.subplot(111,projection=ccrs.LambertConformal(central_longitude=-97.5,central_latitude=38.5),label='icat')
ax3.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
ax3.add_feature(cfeature.STATES, linewidth=0.5)
ax3.add_feature(cfeature.BORDERS, linewidth=0.5)
p3 = ax3.pcolormesh(fileData.lon0,fileData.lat0,fileData['NEWCAT'].max(dim='z0').where(fileData['NEWCAT'].max(dim='z0')>0.0),transform=ccrs.PlateCarree(),cmap=icatcols,norm=icatnorm)
#p3 = ax3.contourf(fileData.lon0,fileData.lat0,fileData['NEWCAT'].max(dim='z0').where(fileData['NEWCAT'].max(dim='z0')>0.0),transform=ccrs.PlateCarree(),cmap=icatcols)
cb3 = plt.colorbar(p3, ax=ax3, orientation='horizontal',shrink=0.74, pad=0.05)
ax3.set_title(tstring,fontsize=16,loc='left')
ax3.set_title('ICING SEVERITY CATEGORY COMPOSITE\nNCAR_TEST_ID: '+fileData.NCAR_TEST_ID.strip(),fontsize=16,loc='right')
cb3.set_label('CAT', size='x-large')
fig.savefig('mdv64_ICE_SEV_CAT.png')
