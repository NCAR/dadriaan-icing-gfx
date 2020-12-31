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

import metpy.calc as mpcalc
import metpy.plots as mpplt
from metpy.units import units
from metpy.interpolate import log_interpolate_1d

import datetime, os

# What forecast lead (hours)?
flead = 3

# Set the requested date
rd = datetime.datetime.strptime(p.opt['tstring'],'%Y-%m-%d %H:%M:%S')

# What date string
yyyymm = rd.strftime('%Y%m')
yyyymmdd = rd.strftime('%Y%m%d')
hhmmss = rd.strftime('%H%M%S')
fn = int(p.opt['fnum'])

# Set whether the grib file is native or pressure levels
nat = False
prs = False
if 'nat' in p.opt['gribfile']:
  nat = True
  natf = p.opt['gribfile']
  print("\nPROCESSING: %s" % (natf))
else:
  prs = True
  prsf = p.opt['gribfile']
  print("\nPROCESSING: %s" % (prsf))

# Vars to save
natsave = ['gh','t','q','u','v','w','pres']
prssave = ['gh','t','r','q','u','v','w']

print("\nLOADING LOCAL\n")
# DO NATIVE
if nat:
  # Get the hybrid level vars
  #ds = xr.open_dataset(natf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'hybrid'}})
  for v in natsave:
    tmpds = xr.open_dataset(natf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'hybrid','cfVarName':v}})
    if 'ds' in locals():
      ds = xr.merge([ds,tmpds])
    else:
      ds = tmpds
    del(tmpds)
  
  # Get the vars at 2m
  # stepType = 'instant' or stepType = 'max'
  ds02m = xr.open_dataset(natf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'heightAboveGround','stepType':'instant','level':2,'level':2}}).rename_vars({'t2m':'VAR_2T'})

  # Get the vars at 10m
  ds10m = xr.open_dataset(natf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'heightAboveGround','stepType':'instant','level':2,'level':10}}).rename_vars({'u10':'VAR_10U','v10':'VAR_10V'})

  # Get the surface vars
  # stepType = 'instant', or stepType = 'accum'
  # instant -> sp, surface pressure MSLP
  # instant -> csnow, categorical snow
  # instant -> crain, categorical rain
  # instant -> cfrzr, categorical freezing rain
  # instant -> cicep, categorical ice pellets
  # accum -> frzr, accumulated freezing rain
  # accum -> tp, total precipiation (mm)
  # accum -> asnow, accumulated snow (mm?)
  dssfc_ins = xr.open_dataset(natf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'surface','stepType':'instant'}}).rename_vars({'sp':'MSL'})
  #dssfc_acc = xr.open_dataset(natf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'surface','stepType':'accum'}})

  # Merge together 2D fields we want
  ds2 = xr.merge([ds02m['VAR_2T'],ds10m[['VAR_10V','VAR_10U']],dssfc_ins['MSL']],compat='override')
 
  # Pressure levels for interpolation
  plevs = np.arange(100.0,1025.0,25.0) * units.hPa

  # Interpolate
  # Return nlev*ny*nx for each variable
  h,t,q,u,v,w = log_interpolate_1d(plevs,ds['pres'],ds['gh'],ds['t'],ds['q'],ds['u'],ds['v'],ds['w'],axis=0)

  hda = xr.DataArray(h,dims=['level','y','x'],coords={'level':plevs,'latitude':(['y','x'],ds.latitude.values),'longitude':(['y','x'],ds.longitude.values)},attrs={})
  tda = xr.DataArray(t*units.kelvin,dims=['level','y','x'],coords={'level':plevs,'latitude':(['y','x'],ds.latitude.values),'longitude':(['y','x'],ds.longitude.values)},attrs={})
  qda = xr.DataArray(q,dims=['level','y','x'],coords={'level':plevs,'latitude':(['y','x'],ds.latitude.values),'longitude':(['y','x'],ds.longitude.values)},attrs={})
  uda = xr.DataArray(u,dims=['level','y','x'],coords={'level':plevs,'latitude':(['y','x'],ds.latitude.values),'longitude':(['y','x'],ds.longitude.values)},attrs={})
  vda = xr.DataArray(v,dims=['level','y','x'],coords={'level':plevs,'latitude':(['y','x'],ds.latitude.values),'longitude':(['y','x'],ds.longitude.values)},attrs={})
  wda = xr.DataArray(w,dims=['level','y','x'],coords={'level':plevs,'latitude':(['y','x'],ds.latitude.values),'longitude':(['y','x'],ds.longitude.values)},attrs={})

  # Add 3D pressure
  pda = xr.DataArray(plevs,dims=['level'],coords={'level':plevs}).broadcast_like(tda)

  # Compute RH
  rda = xr.DataArray(mpcalc.relative_humidity_from_specific_humidity(pda.values*units.hPa,tda.values*units.kelvin,qda.values)*100.0,dims=['level','y','x'],coords={'level':plevs,'latitude':(['y','x'],ds.latitude.values),'longitude':(['y','x'],ds.longitude.values)},attrs={})

  # Set the dimensions for the ds
  dsdims = ['level','y','x']

  # Set the variables for the ds
  dsvars={'T':(dsdims,tda.values),'Z':(dsdims,hda.values),'Q':(dsdims,qda.values),'U':(dsdims,uda.values),'V':(dsdims,vda.values),'W':(dsdims,wda.values),'R':(dsdims,rda.values),'P':(dsdims,pda.values)}

  # Set the coordinates for the ds
  dscoords={'level':plevs,'latitude':(['y','x'],ds.latitude.values),'longitude':(['y','x'],ds.longitude.values)}

  # Set the attributes for the ds
  dsattrs = {}

  # Create the ds 
  ds3 = xr.Dataset(data_vars=dsvars,coords=dscoords,attrs=dsattrs)
  
  # Write netcdf
  f2d = '%s.2D.nc' % (os.path.basename(p.opt['gribfile']))
  f3d = '%s.3D.nc' % (os.path.basename(p.opt['gribfile']))
  ds2.to_netcdf(f2d)
  ds3.to_netcdf(f3d)

# DO PRESSURE
else:
  # Get the isobaric level vars
  #ds = xr.open_dataset(prsf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'isobaricInhPa'}})
  #print(ds)
  for v in prssave:
    tmpds = xr.open_dataset(prsf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'isobaricInhPa','cfVarName':v}})
    if 'ds' in locals():
      ds = xr.merge([ds,tmpds])
    else:
      ds = tmpds
    del(tmpds)

  # Get the vars at 2m
  # stepType = 'instant' or stepType = 'max'
  ds02m = xr.open_dataset(prsf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'heightAboveGround','stepType':'instant','level':2,'level':2,'cfVarName':'t2m'}}).rename_vars({'t2m':'VAR_2T'})

  # Get the vars at 10m
  ds10m = xr.open_dataset(prsf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'heightAboveGround','stepType':'instant','level':2,'level':10}}).rename_vars({'u10':'VAR_10U','v10':'VAR_10V'})

  # Get the surface vars
  # stepType = 'instant', or stepType = 'accum'
  # instant -> sp, surface pressure MSLP
  # instant -> csnow, categorical snow
  # instant -> crain, categorical rain
  # instant -> cfrzr, categorical freezing rain
  # instant -> cicep, categorical ice pellets
  # accum -> frzr, accumulated freezing rain
  # accum -> tp, total precipiation (mm)
  # accum -> asnow, accumulated snow (mm?)
  dssfc_ins = xr.open_dataset(prsf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'surface','stepType':'instant','cfVarName':'sp'}}).rename_vars({'sp':'MSL'})
  #dssfc_acc = xr.open_dataset(natf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'surface','stepType':'accum'}})

  # Merge together 2D fields we want
  ds2 = xr.merge([ds02m['VAR_2T'],ds10m[['VAR_10V','VAR_10U']],dssfc_ins['MSL']],compat='override')

  # NEED TO:
  # 1. Rename vars
  # 2. Create 3D pressure field?
  # 3. Write out

  # Rename isobaric vars
  ds = ds.rename_vars({'gh':'Z','t':'T','r':'R','q':'Q','u':'U','v':'V','w':'W'})

  # Add 3D pressure
  ds['P'] = xr.DataArray(ds.isobaricInhPa.values,dims=['isobaricInhPa'],coords={'isobaricInhPa':ds.isobaricInhPa.values}).broadcast_like(ds['T'])

  # Write netcdf
  f2d = '%s.2D.nc' % (os.path.basename(p.opt['gribfile']))
  f3d = '%s.3D.nc' % (os.path.basename(p.opt['gribfile']))
  ds2.to_netcdf(f2d)
  ds.to_netcdf(f3d)

# Should we try and trim down data prior to interpolating?
#min_lon = -110.0+360.0
#max_lon = -60.0+360.0
#min_lat = 20.0
#max_lat = 50.0
#mask_lat = (ds.latitude >= min_lat) & (ds.latitude <= max_lat)
#mask_lon = (ds.longitude >= min_lon) & (ds.longitude <= max_lon)
#test = ds.where(mask_lat & mask_lon,drop=True)
#ds = ds.reset_coords(['time','step','valid_time'],drop=True)
#test = ds.where(((ds.latitude >= min_lat) &
#                 (ds.latitude <= max_lat) &
#                 (ds.longitude >= min_lon) &
#                 (ds.longitude <= max_lon)),drop=False)

# We should probably subset before writing netcdf?
#mask_lat = (ds2.latitude>=min_lat) & (ds2.latitude<=max_lat)
#mask_lon = (ds2.longitude>=min_lon) & (ds2.longitude<=max_lon)
#dsout = ds2.where(mask_lat & mask_lon,drop=True)

#print(ds)
#print(ds2)
#print(ds02m)
#print(ds10m)
#print(dssfc_ins)
#print(dssfc_acc)
#print(ds3)

# ds = hybrid
# ds2 = pres
# ds3 = merged 10m/2m data
#print(ds2)
#print(dsout)
#ds2.to_netcdf('hrrr3dprs.nc')
#dsout.to_netcdf('hrrr3dprs.nc')
#ds3.to_netcdf('hrrr2dprs.nc')
