#!/usr/bin/env python

# Import ConfigMaster
from ConfigMaster import ConfigMaster

# Create a params class
class Params(ConfigMaster):
  defaultParams = """

#!/usr/bin/env python

# DEFAULT PARAMS

####### zoom #######
#
# NAME: zoom
# OPTIONS:
# TYPE: list
# FORMAT: [minLon, maxLon, minLat, maxLat]
# DEFAULT: conus
# DESCRIPTION: Control the zooming of the domain for plotting using latitude/longitude
# conus = [235.,290.,20.,55.]
# icicle = [255.,280.,35.,50.]
#
zoom = [255.,280.,35.,50.]

####### tstring #######
#
# NAME: tstring
# OPTIONS:
# TYPE: string
# FORMAT: YYYY-MM-DD HH:MM:SS
# DEFAULT: 2019-02-01 00:00:00
# DESCRIPTION: Set the time you wish to plot
#
tstring = '2019-02-01 00:00:00'

####### fnum #######
# NAME: fnum
# OPTIONS:
# TYPE: integer
# FORMAT: 
# DEFAULT: 5
# DESCRIPTION: Flight number to look for the flight path file
#
fnum = 5

###### input_dir #######
#
# NAME: input_dir
# OPTIONS:
# TYPE:
# FORMAT: string
# DEFAULT: ''
# DESCRIPTION: Where to look for input data and write images
#
input_dir = ''

####### model_name #######
#
# NAME: model_name
# OPTIONS: 'era5' or 'hrrr'
# TYPE:
# FORMAT: string
# DEFAULT: 'era5'
# DESCRIPTION: String to ID model
#
model_name = 'era5'

####### gribfile #######
#
# NAME: gribfile
# OPTIONS:
# TYPE:
# FORMAT: string
# DEFAULT: ''
# DESCRIPTION: Absolute path to a grib file
#
gribfile = ''

####### matchfile #######
#
# NAME: matchfile
# OPTIONS:
# TYPE:
# FORMAT: string
# DEFAULT: ''
# DESCRIPTION: CSV file containing i,j matched to aircraft location for the model_name specified
matchfile = ''
"""
