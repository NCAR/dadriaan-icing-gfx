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

DEBUG = False             # Additional info
SHORTCIRCUIT = True      # Is there an xarray netcdf file available? If so, use it for prototyping graphic layout
WRITESHORTCIRCUIT = False # Write out a netCDF file for prototyping. This needs to be True one run prior to invoking SHORTCIRCUIT
INVENTORYGRIB = False     # Just open the grib file and print all variables with vlevel hybrid

# What forecast lead (hours)?
flead = 3

# Convair data file
cnv = pd.read_csv(p.opt['matchfile'],index_col=False)

# Get the aircraft track for plotting
cnvTrack = sgeom.LineString(zip(cnv['lon'],cnv['lat']))

# Create a unique ID string of the home_i and home_j
# Create a unique ID string of the minI, maxI, minJ, and maxJ
#cnv['uid'] = cnv['home_i'].astype(str).str.zfill(4)+cnv['home_j'].astype(str).str.zfill(4)
cnv['uid'] = cnv['minI'].astype(str).str.zfill(4)+cnv['maxI'].astype(str).str.zfill(4)+cnv['minJ'].astype(str).str.zfill(4)+cnv['maxJ'].astype(str).str.zfill(4)
# Take the difference of the unique "uid" created by concatenating the home_i and home_j values. Wherever it's not 0, it's changing
sbeg = cnv[cnv['uid'].astype(float).diff()!=0.0].index
# Set the end indices, which is just one integer prior to each ending above. Strip off the first value which is NaN
send = sbeg[1:]-1
# Add the total length of the series as the last ending point
send = send.append(pd.Index([len(cnv)-1]))
# Compute the delta (length) of each section spanning sbeg:send
deltas = send-sbeg
# Create a new list of the beginning time of each sbeg:send span
begtimes = []
[begtimes.extend([x]*(num+1)) for x,num in zip(cnv['unix_time'],deltas)]
# Assign this to the dataframe
cnv['begtimes'] = begtimes
  
# We need to transform the full grid i,j indexes into subset grid space
subIwidth = max(cnv['maxI'])-min(cnv['minI'])
subJwidth = max(cnv['maxJ'])-min(cnv['minJ'])
cnv['txMinI'] = (cnv['minI']-min(cnv['minI'])) # form: hoodMinI-subMinI
cnv['txMaxI'] = subIwidth - (max(cnv['maxI'])-cnv['maxI']) # form: subIwidth - (subMaxI - hoodMaxI)
cnv['txMinJ'] = (cnv['minJ']-min(cnv['minJ'])) # form: hoodMinJ-subMinJ
cnv['txMaxJ'] = subJwidth - (max(cnv['maxJ'])-cnv['maxJ']) # form: subJwidth - (subMaxJ - hoodMaxJ)
if DEBUG:
  print("TRANSFORMED COORDS:")
  print(cnv[['minI','maxI','minJ','maxJ','txMinI','txMaxI','txMinJ','txMaxJ']])
  print(min(cnv['minI']),max(cnv['maxI']),min(cnv['minJ']),max(cnv['maxJ']))
  print(min(cnv['txMinI']),max(cnv['txMaxI']),min(cnv['txMinJ']),max(cnv['txMaxJ']))

# Group the a/c data into neighborhoods we want to use for plotting
groups = cnv.groupby(cnv.begtimes)

# Loop over the groups
#for name, group in groups:
#  #print(name)
#  print(group[[' press (hPa)','minI','maxI','minJ','maxJ','home_i','home_j']])
#exit()

# Vars to save
gribsave = ['gh','t','q','u','v','w','pres','clwmr','rwmr','snmr','grle','icmr']
  
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
  ax.set_xticks([0,0.5,1.0])
  ax.set_xticklabels(["0","0.5","1"])
  return(ax)

# Now we have the data in a dataframe, we need to loop over every group and make a plot for each
imgcnt = 0
mtimeChk = 0
for name, group in groups:

  # See what model file we need to load. Keep track of this to avoid erroneously loading the model file for every group
  # This is just indir+/YYYYMMDD_iHH_fLLL_HRRRv4_GSD_wrfnat.grb2
  # We will need to double check what lead is available, which should be either 2,3,4 hr depending on tmatch time.

  # This is the valid time, need to subtract appropriate forecast lead
  mtime = datetime.datetime.utcfromtimestamp(group['tmatch'].iloc[0])

  if DEBUG:
    print("\nGROUP MODEL VALID TIME:\n")
    print(mtime)

  # If the current mtime is not equal to the last set, then it's a new model file so get the data
  if mtime!=mtimeChk:

    # Delete the previous dataset
    if mtimeChk!=0:
      del(ds)

    # Mapping for HRRRv4 retro runs
    # Key = lead, Values = valid
    hv4map = {'2':[2,5,8,11,14,17,20,23],'3':[0,3,6,9,12,15,18,21],'4':[1,4,7,10,13,16,19,22]}

    # Get the current valid hour
    vhr = mtime.strftime('%H')

    # For the current valid hour, figure out which forecast lead and init we need
    for k,v in hv4map.items():
      if int(vhr) in v:
        lhr = k

    # Set the init time
    itime = mtime-datetime.timedelta(seconds=int(lhr)*3600)

    # Construct the filename
    natf = '%s/%s_i%s_f%03d_HRRRv4_GSD_wrfnat.grb2' % (p.opt['input_dir'],itime.strftime('%Y%m%d'),itime.strftime('%H'),int(lhr))
    if not os.path.exists(natf):
      print("\nFILE %s NOT FOUND. COPYING FROM CAMPAIGN STORE.\n")
      cmd = 'scp -r dadriaan@data-access.ucar.edu:/glade/campaign/ral/aap/icing/ICICLE/data/hrrrv4/%s/%s %s' % (itime.strftime('%Y%m%d'),os.path.basename(natf),p.opt['input_dir'])
      print(cmd)
      os.system(cmd)
    else:
      print("\nPROCESSING: %s\n" % (natf))

    # Pull out all hybrid level vars to see what's in the file
    if INVENTORYGRIB:
      ds = xr.open_dataset(natf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'hybrid'},'indexpath':''})
      print(ds)
      exit()

    # Load the testing netCDF if we request, otherwise load from GRIB
    if SHORTCIRCUIT:
      ds = xr.open_dataset('test.nc')
    else:

      # Only save the variables we want
      for v in gribsave:
        if 'paramId' in v:
          tmpds = xr.open_dataset(natf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'hybrid','paramId':0},'indexpath':''})
        else:
          tmpds = xr.open_dataset(natf,engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'hybrid','cfVarName':v},'indexpath':''})
        if 'ds' in locals():
          ds = xr.merge([ds,tmpds])
        else:
          ds = tmpds
        del(tmpds)
      if DEBUG:
        print("\nFULL DATASET:\n")
        print(ds)
    
      # Set the current model time to check next group
      mtimeChk = mtime

      # Subset the GRIB data to just this flight domain and load it to memory
      print("\nSUBSETTING AND LOADING GRID\n")
      dst = ds.isel(x=slice(min(cnv['minI']),max(cnv['maxI'])+1),y=slice(min(cnv['minJ']),max(cnv['maxJ'])+1)).load()
      del(ds)
      ds = dst
      if DEBUG:
        print("\nSUBSET DATASET:\n")
        print(ds)
      if WRITESHORTCIRCUIT:
        ds.to_netcdf('test.nc')

  # Slice out the data in the neighborhood around the current aircraft position
  if DEBUG:
    print(ds.sizes)
    print(group[['minI','maxI','minJ','maxJ','txMinI','txMaxI','txMinJ','txMaxJ']])
    print(min(cnv['minI']),max(cnv['maxI']),min(cnv['minJ']),max(cnv['maxJ']))
  minX = group['txMinI'].iloc[0]
  maxX = group['txMaxI'].iloc[0]
  minY = group['txMinJ'].iloc[0]
  maxY = group['txMaxJ'].iloc[0]

  # Compute the number of x and y- it should be the same (square)
  nprofiles = maxX-minX+1

  # Define a tuple of x-y values to loop over
  coords = list((np.arange(0,nprofiles,1)))

  # Subset the data
  sub = ds.isel(x=slice(minX,maxX+1),y=slice(minY,maxY+1))
  if DEBUG:
    print("\nLOCAL DATASET:\n")
    print(sub)

  # Perform some statistics
  mean = sub.mean(dim=['y','x'])
  #med = sub.median(dim=['y','x'])
  #std = sub.std(dim=['y','x'])
  #var = sub.var(dim=['y','x'])
  #print(mean)
  #print(med)
  #print(std)
  #print(var)

  # Figure
  fig = plt.figure(1, figsize=(22,15))

  # Widths for each panel, skewT then each x-y panel after. This will need to be adjusted
  # if any of the x-y panels are added or removed.
  widths = [4.9,1,1,1,1,1,1]

  # Define the gridspec locations for each panel
  gs = gridspec.GridSpec(nrows=1,ncols=len(widths),wspace=0.05,width_ratios=widths)

  # First panel, SkewT 
  skew = SkewT(fig=fig,subplot=gs[0,0],rotation=45.0)

  # Plot the temperature lines
  [skew.plot(sub.pres.isel(y=y,x=x)/100.0,sub.t.isel(y=y,x=x)-273.15,'r',marker='.') for y,x in tuple(zip(coords,coords))]

  # Plot the aircraft data for the current aircraft location
  skew.plot(group[' press (hPa)'],group[' temp (C)'],'cyan',marker='.',linestyle='',markersize=8)

  # Plot the Dewpoint lines
  [skew.plot(sub.pres.isel(y=y,x=x)/100.0,mpcalc.dewpoint_from_specific_humidity(sub.pres.isel(y=y,x=x)*units.pascals,sub.t.isel(y=y,x=x)*units.kelvin,sub.q.isel(y=y,x=x)),'g',marker='.') for y,x in tuple(zip(coords,coords))]

  # SkewT annotations
  skew.plot_dry_adiabats(t0=np.arange(233, 533, 10) * units.K,alpha=0.25, color='orangered')
  #skew.plot_dry_adiabats()
  skew.plot_moist_adiabats(t0=np.arange(233, 400, 5) * units.K,alpha=0.25, color='tab:green')
  #skew.plot_moist_adiabats()
  skew.plot_mixing_lines(pressure=np.arange(1000, 99, -20) * units.hPa,linestyle='dotted', color='tab:blue')
  #skew.plot_mixing_lines()

  # Add wind barbs offset from the right side a bit using xloc. Use the mean winds.
  skew.plot_barbs(mean.pres.where((mean.pres>10000.0) & (mean.pres<100000.0)).values/100.0,(mean.u.values*units('m/s')).to(units.knots),(mean.v.values*units('m/s')).to(units.knots),xloc=0.95)

  # Control skewT axis limits
  skew.ax.set_ylim(1000, 100)
  skew.ax.set_xlim(-30, 20)

  # Convert mixing ratios from kg/kg to g/m3
  clwmr = (sub.clwmr*1000.0*sub.pres)/(287.05*sub.t)
  rwmr = (sub.rwmr*1000.0*sub.pres)/(287.05*sub.t)
  cice = (sub.icmr*1000.0*sub.pres)/(287.05*sub.t)
  snmr = (sub.snmr*1000.0*sub.pres)/(287.05*sub.t)
  grmr = (sub.snmr*1000.0*sub.pres)/(287.05*sub.t)

  # Plot next to skewT
  # Plot MIXRs in g/m3
  ax1 = plt.subplot(gs[0,1])
  setup_bonus_plot(ax1)
  [ax1.plot(clwmr.isel(y=y,x=x),sub.pres.isel(y=y,x=x)/100.0,marker='.',color='pink') for y,x in tuple(zip(coords,coords))]
  #lp1 = ax1.plot((mean.clwmr*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='pink')
  ax1.set_xlabel('clwmr g/m3')

  ax2 = plt.subplot(gs[0,2])
  setup_bonus_plot(ax2)
  [ax2.plot(snmr.isel(y=y,x=x),sub.pres.isel(y=y,x=x)/100.0,marker='.',color='red') for y,x in tuple(zip(coords,coords))]
  #lp2 = ax1.plot((mean.snmr*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='red')
  ax2.set_xlabel('snmr g/m3')
  
  ax3 = plt.subplot(gs[0,3])
  setup_bonus_plot(ax3)
  [ax3.plot(rwmr.isel(y=y,x=x),sub.pres.isel(y=y,x=x)/100.0,marker='.',color='blue') for y,x in tuple(zip(coords,coords))]
  #lp3 = ax1.plot((mean.rwmr*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='blue')
  ax3.set_xlabel('rwmr g/m3')
 
  ax4 = plt.subplot(gs[0,4])
  setup_bonus_plot(ax4)
  [ax4.plot(cice.isel(y=y,x=x),sub.pres.isel(y=y,x=x)/100.0,marker='.',color='black') for y,x in tuple(zip(coords,coords))]
  #lp4 = ax1.plot((mean.paramId_0*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='black')
  ax4.set_xlabel('cice g/m3')

  ax5 = plt.subplot(gs[0,5])
  setup_bonus_plot(ax5)
  [ax5.plot(grmr.isel(y=y,x=x),sub.pres.isel(y=y,x=x)/100.0,marker='.',color='green') for y,x in tuple(zip(coords,coords))]
  #lp5 = ax1.plot((mean.grle*1000.0*mean.pres)/(287.05*mean.t),mean.pres.values/100.0,marker='.',color='green')
  ax5.set_xlabel('grmr g/m3')

  # Plot vertical wind (w)
  ax6 = plt.subplot(gs[0,6])
  setup_bonus_plot(ax6)
  ax6.set_xlim([-2.5, 2.5])
  ax6.set_xticks([-2.0,-1.0,0.0,1.0,2.0])
  ax6.set_xticklabels(['-2','-1','0','1','2'])
  [ax6.plot(sub.w.isel(y=y,x=x),sub.pres.isel(y=y,x=x)/100.0,marker='.',color='magenta') for y,x in tuple(zip(coords,coords))]
  ax6.set_xlabel('w m/s')

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
  insLeft = 0.073
  insBottom = 0.63
  insWidth = 0.25
  insHeight = 0.25
  ax_inset = fig.add_axes([insLeft, insBottom, insWidth, insHeight],projection=crs,frameon=True,label='ax%03d' % (imgcnt))
  latOff = 1
  lonOff = 0.75
  #ax_inset.set_extent([255.,280.,35.,50.],ccrs.PlateCarree()) # ICICLE
  #ax_inset.set_extent([235.,290.,20.,55.],ccrs.PlateCarree()) # CONUS
  ax_inset.set_extent([min(cnv['lon'])+360.0-lonOff,max(cnv['lon'])+360.0+lonOff,min(cnv['lat'])-latOff,max(cnv['lat'])+latOff],ccrs.PlateCarree()) # FLIGHT
  ax_inset.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
  ax_inset.add_feature(cfeature.STATES, linewidth=0.5)
  ax_inset.add_feature(cfeature.BORDERS, linewidth=0.5)

  # Add the flight track line
  lobj = ax_inset.add_geometries([cnvTrack], ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.5)
  
  # Create a 2D DataArray of zeros, then add ones where our neighborhood is
  ma = xr.DataArray(np.zeros(lon_2d.shape),dims=['y','x'],coords={'latitude':(['y','x'],lat_2d),'longitude':(['y','x'],lon_2d)},attrs={}) 
  
  # Turn this neighborhood to 1's with a mask
  ma[slice(minY,maxY+1),slice(minX,maxX+1)] = 1.0

  # Contour the neighborhood
  mcols = [(255/255,255/255,255/255),(255/255, 0/255, 0/255)]
  maskcols = ListedColormap(mcols)
  ic1 = ax_inset.pcolormesh(lon_2d,lat_2d,ma,transform=ccrs.PlateCarree(),cmap=maskcols,vmin=0.1)
  
  # String info from group
  info_mfstr = '%s_i%s_f%03d_HRRRv4_GSD_wrfnat' % (itime.strftime('%Y%m%d'),itime.strftime('%H'),int(lhr))
  info_itime = itime.strftime('%Y%m%d_%H:%M:%S')
  info_vtime = mtime.strftime('%Y%m%d_%H:%M:%S')
  info_npts = str(len(group))
  info_minactime = datetime.datetime.fromtimestamp(group['unix_time'].iloc[0]).strftime('%H:%M:%S')
  info_maxactime = datetime.datetime.fromtimestamp(group['unix_time'].iloc[len(group)-1]).strftime('%H:%M:%S')
  info_aclengths = (datetime.datetime.fromtimestamp(group['unix_time'].iloc[len(group)-1])-datetime.datetime.fromtimestamp(group['unix_time'].iloc[0])).seconds
  info_flnum = fn
  info_nice = len(group['class'].where((group['class']==9.0)).dropna())
  info_nmix = len(group['class'].where((group['class']==6.0) | (group['class']==7.0) | (group['class']==8.0)).dropna())
  info_nslw = len(group['class'].where((group['class']==1.0) | (group['class']==2.0)).dropna())
  info_nfzdz = len(group['class'].where((group['class']==3.0) | (group['class']==4.0)).dropna())
  info_nfzra = len(group['class'].where((group['class']==5.0)).dropna())
  info_nnone = len(group['class'].where((group['class']==0.0)).dropna())
  info_minseg = min(group[' segment'])
  info_maxseg = max(group[' segment'])
  info_ndesc = len(group[' type'].where((group[' type']==-1.0)).dropna())
  info_nlev = len(group[' type'].where((group[' type']==0.0)).dropna())
  info_nasc = len(group[' type'].where((group[' type']==1.0)).dropna())
  info_nporp = len(group[' type'].where((group[' type']==2.0)).dropna())
  #infostr = [info_mfstr,info_itime,info_vtime,info_npts,info_minactime,info_maxactime,info_aclengths,info_flnum,info_nice,info_nmix,info_nslw,info_nfzdz,info_nfzra,info_nnone,info_minseg,info_maxseg,info_ndesc,info_nlev,info_nasc,info_nporp]
  #print(infostr)
  #txstr1 = ('FNUM: %d' % (int(info_flnum))).ljust(20)+'\n'+('M INIT: '+info_itime).ljust(20)+'\n'+('M VALD: '+info_vtime).ljust(20)+'\n'+\
  txstr1 = ('AC BEG: '+info_minactime).ljust(20)+'\n'+('AC END: '+info_maxactime).ljust(20)+'\n'+('AC TIME: %d s' % (int(info_aclengths))).ljust(20)+'\n'+\
          ('AC PTS: %03d' % (int(info_npts))).ljust(20)+'\n'+('AC BSEG: %d' % (int(info_minseg))).ljust(20)+'\n'+('AC ESEG: %d' % (int(info_maxseg))).ljust(20)+'\n'+\
          ('AC DESC: %d / %d' % (int(info_ndesc),int(info_npts))).ljust(20)+'\n'+('AC ASC: %d / %d' % (int(info_nasc),int(info_npts))).ljust(20)+'\n'+\
          ('AC LVL: %d / %d' % (int(info_nlev),int(info_npts))).ljust(20)+'\n'+('AC PORP: %d / %d' % (int(info_nporp),int(info_npts))).ljust(20)
  #print(txstr1)
  txstr2 = ('COND ICE: %d / %d' % (int(info_nice),int(info_npts))).ljust(20)+'\n'+('COND MIX: %d / %d' % (int(info_nmix),int(info_npts))).ljust(20)+'\n'+\
           ('COND SLW: %d / %d' % (int(info_nslw),int(info_npts))).ljust(20)+'\n'+('COND FZDZ: %d / %d' % (int(info_nfzdz),int(info_npts))).ljust(20)+'\n'+\
           ('COND FZRA: %d / %d' % (int(info_nfzra),int(info_npts))).ljust(20)+'\n'+('COND NONE: %d / %d' % (int(info_nnone),int(info_npts))).ljust(20)
  #print(txstr2)
  
  # Add a text box with info about the flight
  tx1X = 0.905
  tx1Y = 0.77
  props = dict(boxstyle='square',facecolor='white',alpha=1.0)
  tb1 = fig.text(tx1X,tx1Y,txstr1,transform=fig.transFigure,fontsize=10,bbox=props,label='tb1%03d' % (imgcnt))
  tx2X = 0.905
  tx2Y = 0.69
  tb2 = fig.text(tx2X,tx2Y,txstr2,transform=fig.transFigure,fontsize=10,bbox=props,label='tb2%03d' % (imgcnt))

  # Add left/center/right titles?
  t1x = 0.13
  t1y = 0.89
  t1str = 'FNUM: %d' % (int(info_flnum))+'\nVALID: %s' % (info_vtime)+'\nFCST: %d-HR' % (int(lhr))
  tb3 = fig.text(t1x,t1y,t1str,transform=fig.transFigure,fontsize=14,label='tx1%03d' % (imgcnt))
  t2x = 0.5
  t2y = 0.9
  t2str = 'INIT: %s' % (info_itime)
  #fig.text(t2x,t2y,t2str,transform=fig.transFigure,fontsize=14)
  t3x = 0.72
  t3y = 0.9
  t3str = 'VALD: %s' % (info_vtime)
  #fig.text(t3x,t3y,t3str,transform=fig.transFigure,fontsize=14)

  # Save sounding 
  fig.savefig('testsounding_%03d.png' % imgcnt)

  # Clear the figure
  fig.clf()

  # On to the next group!
  imgcnt = imgcnt + 1
  
