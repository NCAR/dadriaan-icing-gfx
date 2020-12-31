from grib_plot_params import Params
p = Params()
p.init()

from siphon.catalog import TDSCatalog
from siphon.http_util import session_manager

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

import shapely.geometry as sgeom

import datetime, os

import pandas as pd

# What forecast lead (hours)?
flead = 3

# Point to the convair file we want
cnvFile = '/scratch/WEEKLY/dadriaan/icicle_arugg30s_03_hrrr.csv'
cnames = ['unix_time','latitude','longitude','alt','init_time','fcst_lead','minI','maxI','minJ','maxJ','homeI','homeJ','corner','npts','val']
cnv = pd.read_csv(cnvFile,names=cnames)

# Get the aircraft track for plotting
cnvTrack = sgeom.LineString(zip(cnv['longitude'],cnv['latitude']))

# Set the requested date
rd = datetime.datetime.strptime(p.opt['tstring'],'%Y-%m-%d %H:%M:%S')

# What date string
yyyymm = rd.strftime('%Y%m')
yyyymmdd = rd.strftime('%Y%m%d')
hhmmss = rd.strftime('%H%M%S')
fn = int(p.opt['fnum'])

# Set whether the grib file is native or pressure levels
if 'nat' in p.opt['gribfile']:
  nat = True
  natf = p.opt['gribfile']
  print("\nPROCESSING: %s" % (natf))

# Vars to save
natsave = ['gh','t','q','u','v','w','pres','clwmr','rwmr','snmr','grle','paramId_0']

print("\nLOADING LOCAL\n")
# DO NATIVE
if nat:
  # Get the hybrid level vars
  #ds = xr.open_dataset(natf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'hybrid'}})
  #print(ds)
  #exit()
  for v in natsave:
    if 'paramId' in v:
      tmpds = xr.open_dataset(natf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'hybrid','paramId':0},'indexpath':''})
    else:
      tmpds = xr.open_dataset(natf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'hybrid','cfVarName':v},'indexpath':''})
    if 'ds' in locals():
      ds = xr.merge([ds,tmpds])
    else:
      ds = tmpds
    del(tmpds)
  print(ds)

  # Slice out the data in the hood
  minX = 1171
  maxX = 1178
  minY = 690
  maxY = 697

  # Compute the number of x and y- it should be the same (square)
  nprofiles = maxX-minX+1

  # Define a tuple of x-y values to loop over
  coords = list((np.arange(0,nprofiles,1)))

  # Subet the data
  sub = ds.isel(x=slice(minX,maxX+1),y=slice(minY,maxY+1)) 
  #sub.to_netcdf('test.nc')

  # Perform some statistics
  mean = sub.mean(dim=['y','x'])
  med = sub.median(dim=['y','x'])
  std = sub.std(dim=['y','x'])
  var = sub.var(dim=['y','x'])
  #print(mean)
  #print(med)
  #print(std)
  #print(var)

  # Figure stuff
  fig = plt.figure(1, figsize=(22,15))
  #widths = [19,1,1,1,1,1] # skew default, 6 panels, x-y 0-1
  widths = [3.9,1,1,1,1,1] # skew -30/20, 6 panels, x-y 0-1
  #widths = [1.8,1,1,1,1,1,] # skew -20/10, 6 panels, x-y 0-1
  gs = gridspec.GridSpec(nrows=1,ncols=6,wspace=0.05,width_ratios=widths)
  #widths = [2.5,1] # skew default, 1 panel x-y 0-1
  #gs = gridspec.GridSpec(nrows=1,ncols=2,wspace=0.05,width_ratios=widths)

  # First panel, SkewT 
  aspect = '80.5'
  skew = SkewT(fig=fig,subplot=gs[0,0],rotation=45.0,aspect=aspect)
  [skew.plot(sub.pres.isel(y=y,x=x).values/100.0,sub.t.isel(y=y,x=x).values-273.15,'r',marker='.') for y,x in tuple(zip(coords,coords))]
  [skew.plot(sub.pres.isel(y=y,x=x).values/100.0,mpcalc.dewpoint_from_specific_humidity(sub.pres.isel(y=y,x=x)*units.pascals,sub.t.isel(y=y,x=x)*units.kelvin,sub.q.isel(y=y,x=x)).values,'g',marker='.') for y,x in tuple(zip(coords,coords))]
  skew.plot_dry_adiabats(t0=(233, 533, 10) * units.K,alpha=0.25, color='orangered')
  #skew.plot_dry_adiabats()
  skew.plot_moist_adiabats(t0=(233, 400, 5) * units.K,alpha=0.25, color='tab:green')
  #skew.plot_moist_adiabats()
  skew.plot_mixing_lines(pressure=(1000, 99, -20) * units.hPa,linestyle='dotted', color='tab:blue')
  #skew.plot_mixing_lines()
  skew.plot_barbs(mean.pres.where((mean.pres>10000.0) & (mean.pres<100000.0)).values/100.0,(mean.u.values*units('m/s')).to(units.knots),(mean.v.values*units('m/s')).to(units.knots),xloc=0.95)
  skew.ax.set_ylim(1000, 100)
  skew.ax.set_xlim(-30, 20)
  
  # Function to setup the axes for the bonus panels.
  # This was taken from the MetPy SkewT source code to match the y-axis behavior
  def setup_bonus_plot(ax):
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(1000,100)
    ax.set_yticklabels([])
    #ax.yaxis.set_units(units.hPa)

    ax.set_xlim(0,1.0)
    ax.grid(True)
    return(ax)

  # Convert mixing ratios from kg/kg to g/m3
  clwmr = (sub.clwmr*1000.0*sub.pres)/(287.05*sub.t)
  rwmr = (sub.rwmr*1000.0*sub.pres)/(287.05*sub.t)
  cice = (sub.paramId_0*1000.0*sub.pres)/(287.05*sub.t)
  snmr = (sub.snmr*1000.0*sub.pres)/(287.05*sub.t)
  grmr = (sub.snmr*1000.0*sub.pres)/(287.05*sub.t)

  # Plot next to skewT
  # Plot MIXRs in g/m3
  ax1 = plt.subplot(gs[0,1])
  setup_bonus_plot(ax1)
  #lp1 = ax1.plot((mean.clwmr*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='pink')
  [ax1.plot(clwmr.isel(y=y,x=x),sub.pres.isel(y=y,x=x)/100.0,marker='.',color='pink') for y,x in tuple(zip(coords,coords))]
  #lp1 = ax1.plot((mean.clwmr*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='pink')

  ax2 = plt.subplot(gs[0,2])
  setup_bonus_plot(ax2)
  #lp2 = ax2.plot((mean.snmr*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='red')
  [ax2.plot(snmr.isel(y=y,x=x),sub.pres.isel(y=y,x=x)/100.0,marker='.',color='red') for y,x in tuple(zip(coords,coords))]
  #lp2 = ax1.plot((mean.snmr*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='red')
  
  ax3 = plt.subplot(gs[0,3])
  setup_bonus_plot(ax3)
  #lp3 = ax3.plot((mean.rwmr*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='blue')
  [ax3.plot(rwmr.isel(y=y,x=x),sub.pres.isel(y=y,x=x)/100.0,marker='.',color='blue') for y,x in tuple(zip(coords,coords))]
  #lp3 = ax1.plot((mean.rwmr*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='blue')
 
  ax4 = plt.subplot(gs[0,4])
  setup_bonus_plot(ax4)
  #lp4 = ax4.plot((mean.paramId_0*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='black')
  [ax4.plot(cice.isel(y=y,x=x),sub.pres.isel(y=y,x=x)/100.0,marker='.',color='black') for y,x in tuple(zip(coords,coords))]
  #lp4 = ax1.plot((mean.paramId_0*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='black')

  ax5 = plt.subplot(gs[0,5])
  setup_bonus_plot(ax5)
  #lp5 = ax5.plot((mean.grle*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='green')
  [ax5.plot(grmr.isel(y=y,x=x),sub.pres.isel(y=y,x=x)/100.0,marker='.',color='green') for y,x in tuple(zip(coords,coords))]
  #lp5 = ax1.plot((mean.grle*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='green')

  # Plot pressure variation stuff
  #print(mean.pres/100.0)
  #print(mean.pres/100.0-2.0*(std.pres/100.0))
  #print(mean.pres/100.0+2.0*(std.pres/100.0))
  #print(var.pres/100.0)
  #print(std.pres/100.0)
  #lp6 = ax1.plot(mean.pres/100.0,mean.pres.values/100.0,marker='.',color='black',linewidth=0.5)
  #lp7 = ax1.plot(mean.pres/100.0-2.0*(std.pres/100.0),mean.pres.values/100.0,linestyle='--',linewidth=1.0,color='gray')
  #lp8 = ax1.plot(mean.pres/100.0+2.0*(std.pres/100.0),mean.pres.values/100.0,linestyle='--',linewidth=1.0,color='gray')
  #ax1.xaxis.set_major_formatter(ScalarFormatter())
  #ax1.xaxis.set_major_locator(MultipleLocator(100))
  #ax1.xaxis.set_minor_formatter(NullFormatter())
  #ax1.set_xscale('log')
  #ax1.set_xlim(1000,100)
  #ax1.set_yticklabels([])

  # Add a little inset map highlighting the geographic location of the neighborhood plotted
  lon_2d = ds.longitude
  lat_2d = ds.latitude
  crs = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5)
  ax_inset = fig.add_axes([0.125, 0.645, 0.25, 0.25],projection=crs,frameon=True)
  #ax_inset.set_extent([255.,280.,35.,50.],ccrs.PlateCarree()) # ICICLE
  ax_inset.set_extent([min(cnv['longitude'])+360.0-2.0,max(cnv['longitude'])+360.0+2.0,min(cnv['latitude'])-1.0,max(cnv['latitude'])+1.0],ccrs.PlateCarree())
  #ax_inset.set_extent([235.,290.,20.,55.],ccrs.PlateCarree()) # CONUS
  ax_inset.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
  ax_inset.add_feature(cfeature.STATES, linewidth=0.5)
  ax_inset.add_feature(cfeature.BORDERS, linewidth=0.5)
  #lobj = ax_inset.add_geometries([cnvTrack], ccrs.PlateCarree(),facecolor='white',edgecolor='white',color='white')
  lobj = ax_inset.add_geometries([cnvTrack], ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.5)
  
  # Create a 2D DataArray of zeros, then add ones where our neighborhood is
  ma = xr.DataArray(np.zeros(lon_2d.shape),dims=['y','x'],coords={'latitude':(['y','x'],lat_2d),'longitude':(['y','x'],lon_2d)},attrs={}) 
  # Turn this neighborhood to 1's with a mask
  ma[slice(minY,maxY+1),slice(minX,maxX+1)] = 1.0
  #ma[slice(846,1000+1),slice(1464,1671+1)] = 1.0
  #ma[slice(100,900+1),slice(100,1700+1)] = 1.0

  # Contour the neighborhood
  mcols = [(255/255,255/255,255/255),(255/255, 0/255, 0/255)]
  maskcols = ListedColormap(mcols)
  ic1 = ax_inset.pcolormesh(lon_2d,lat_2d,ma,transform=ccrs.PlateCarree(),cmap=maskcols,vmin=0.1)
 
  # Save sounding 
  fig.savefig('testsounding.png')
  
