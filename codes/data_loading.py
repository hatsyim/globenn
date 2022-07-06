'''
generating a first-ever 3d reference global Earth models
@hatsyim
'''

# chain
from modules_import import *

path_file = '../../00_data/glad-m25-vp-0.0-n4.nc'

data = xr.open_dataset(path_file)

offline = False
old = False

if offline and old:
    dep_ini, dep_inc = 11, 4
    lat_ini, lat_inc = 0, 8
    lon_ini, lon_inc = 0, 8
else:
    dep_ini, dep_inc = 11, 2
    lat_ini, lat_inc = 0, 4
    lon_ini, lon_inc = 0, 4    

# variables
vpv = data.variables['vpv'].values[dep_ini::dep_inc, lat_ini::lat_inc, lon_ini::lon_inc]
vph = data.variables['vph'].values[dep_ini::dep_inc, lat_ini::lat_inc, lon_ini::lon_inc]
longitude = data.variables['longitude'].values[lon_ini::lon_inc]
latitude = data.variables['latitude'].values[lat_ini::lat_inc]
depth = data.variables['depth'].values[dep_ini::dep_inc]

dep_dim = vpv.shape[0]
lat_dim = vpv.shape[1]
lon_dim = vpv.shape[2]