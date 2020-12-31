import datetime
import os
import sys

fnum = int(sys.argv[1])
#tstart = sys.argv[2]
#prod = sys.argv[3]
#nhrs = int(sys.argv[4])
indir = '/home/dadriaan/projects/era5/era5/v2'
gfile = '/scratch/WEEKLY/dadriaan/20201223_i12_f003_HRRR-NCEP_wrfprs.grb2'

# Products
# moist, tadv, fgen, uppr, sfc, rh, isentropic
prod = ['sfc','tadv','uppr','moist','rh','fgen','isentropic']

if fnum==99:
  tstart = '2020-12-23 15:00:00'
  nhrs = 1

if fnum==2:
  tstart = '2019-01-27 13:00:00'
  nhrs = 15

if fnum==3:
  tstart = '2019-01-28 14:00:00'
  nhrs = 16

if fnum==4:
  tstart = '2019-01-29 12:00:00'
  nhrs = 16

if fnum==5:
  tstart = '2019-01-31 10:00:00'
  nhrs = 17

if fnum==6:
  tstart = '2019-02-04 07:00:00'
  nhrs = 16

if fnum==7:
  tstart = '2019-02-05 12:00:00'
  nhrs = 17

if fnum==8:
  tstart = '2019-02-06 13:00:00'
  nhrs = 16

if fnum==9:
  tstart = '2019-02-07 12:00:00'
  nhrs = 17

if fnum==10:
  tstart = '2019-02-08 09:00:00'
  nhrs = 14

if fnum==11:
  tstart = '2019-02-11 09:00:00'
  nhrs = 16

if fnum==12:
  tstart = '2019-02-12 05:00:00'
  nhrs = 17

if fnum==13:
  tstart = '2019-02-12 11:00:00'
  nhrs = 15

if fnum==14:
  tstart = '2019-02-14 11:00:00'
  nhrs = 16

if fnum==15:
  tstart = '2019-02-15 14:00:00'
  nhrs = 16

if fnum==16:
  tstart = '2019-02-15 20:00:00'
  nhrs = 16

if fnum==17:
  tstart = '2019-02-17 06:00:00'
  nhrs = 17

if fnum==18:
  tstart = '2019-02-17 11:00:00'
  nhrs = 16

if fnum==19:
  tstart = '2019-02-22 11:00:00'
  nhrs = 17

if fnum==20:
  tstart = '2019-02-23 06:00:00'
  nhrs = 17

if fnum==21:
  tstart = '2019-02-24 06:00:00'
  nhrs = 16

if fnum==22:
  tstart = '2019-02-24 12:00:00'
  nhrs = 16

if fnum==23:
  tstart = '2019-02-26 11:00:00'
  nhrs = 16

if fnum==24:
  tstart = '2019-02-26 17:00:00'
  nhrs = 16

if fnum==25:
  tstart = '2019-02-28 12:00:00'
  nhrs = 17

if fnum==26:
  tstart = '2019-03-02 05:00:00'
  nhrs = 16

if fnum==27:
  tstart = '2019-03-02 11:00:00'
  nhrs = 15

if fnum==28:
  tstart = '2019-03-05 06:00:00'
  nhrs = 17

sd = datetime.datetime.strptime(tstart,'%Y-%m-%d %H:%M:%S')

t = [sd+datetime.timedelta(seconds=x*3600) for x in range(nhrs)]

for vt in t:
  ts = vt.strftime('%Y-%m-%d %H:%M:%S')
  os.chdir('/home/dadriaan/projects/era5/era5')
  cmd = 'python read_interp_format_grib.py --tstring=\'%s\' --fnum=%d --gribfile=%s' % (ts,fnum,gfile)
  print(cmd)
  os.system(cmd)
