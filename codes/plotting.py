import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import netCDF4
import matplotlib.pylab as pylab
import matplotlib.colors as mcolors
import h5py
import argparse
import pymap3d as pm
import os
import re
import matplotlib.patches as patches
import cartopy.feature as cfeature

# load style
plt.style.use("~/science.mplstyle")

# load US array data
USarray = pd.read_excel('../codes/_US-TA-StationList.xls')
USarrLon = USarray['LON'][::1]
USarrLat = USarray['LAT'][::1]

prediction = xr.open_dataset('../codes/prediction_all_US_array_250.nc')
model_path = "/home/taufikmh/KAUST/spring_2022/global_pinns/01_clean_implementations/models/pretrained_USarr_div3_bl21_n512"
figures_path = model_path + '/'

prediction.V_pred_var.attrs['units'] = r'km2.s-2'
prediction.V_pred_var.values = prediction.V_pred_var.values / len(USarray)

prediction = prediction.assign(V_pred_std = prediction.V_pred_var ** 0.5)
prediction.V_pred_var.attrs['long_name'] = 'Var P-wave Velocity (Recovered)'
prediction.V_pred_std.attrs = prediction.V_pred_var.attrs
prediction.V_pred_std.attrs['long_name'], prediction.V_pred_std.attrs['units'] = 'Std P-wave Velocity (Recovered)', 'km.s-1'

# mean = prediction = xr.open_dataset('../codes/prediction_mean_US_array_2233.nc')

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
depth_plot = depth.flat[np.abs(depth - 250).argmin()]

print(data.vpv[data.depth.values==depth_plot, lat_ini::lat_inc, lon_ini::lon_inc].values.min())

# Mean
fig = plt.figure(figsize=(4,4), dpi=300)
ax = plt.axes(projection=ccrs.Robinson(180))

ax.coastlines()
ax.gridlines()
prediction.V_pred_mean.plot(
    ax=ax, 
    transform=ccrs.PlateCarree(), 
    cbar_kwargs={'shrink': 0.5, 'extend':'neither', 'orientation':'horizontal','pad':0.05}, 
    cmap='jet',
    vmin=data.vpv[data.depth.values==depth_plot, lat_ini::lat_inc, lon_ini::lon_inc].values.min(),
    vmax=data.vpv[data.depth.values==depth_plot, lat_ini::lat_inc, lon_ini::lon_inc].values.max()
)
plt.savefig(figures_path + 'V_pred_map_'+str(250)+'_mean.pdf', bbox_inches="tight")

# Variance
fig = plt.figure(figsize=(4,4), dpi=300)
ax = plt.axes(projection=ccrs.Robinson(180))

ax.coastlines()
ax.gridlines()
prediction.V_pred_var.plot(
    ax=ax, 
    transform=ccrs.PlateCarree(), 
    cbar_kwargs={'shrink': 0.5, 'extend':'neither', 'orientation':'horizontal','pad':0.05}, 
    cmap='jet',
)
plt.savefig(figures_path + 'V_pred_map_'+str(250)+'_var.pdf', bbox_inches="tight")

# Variance
fig = plt.figure(figsize=(4,4), dpi=300)
ax = plt.axes(projection=ccrs.Robinson(180))

ax.coastlines()
ax.gridlines()
prediction.V_pred_std.plot(
    ax=ax, 
    transform=ccrs.PlateCarree(), 
    cbar_kwargs={'shrink': 0.5, 'extend':'neither', 'orientation':'horizontal','pad':0.05}, 
    cmap='jet',
)
plt.savefig(figures_path + 'V_pred_map_'+str(250)+'_std.pdf', bbox_inches="tight")

# Variance
fig = plt.figure(figsize=(4,4), dpi=300)
ax = plt.axes(projection=ccrs.Robinson(180))

ax.coastlines()
ax.gridlines()
prediction.V_pred_std.plot(
    ax=ax, 
    transform=ccrs.PlateCarree(), 
    cbar_kwargs={'shrink': 0.5, 'extend':'neither', 'orientation':'horizontal','pad':0.05}, 
    cmap='jet',
)
plt.savefig(figures_path + 'V_pred_map_'+str(250)+'_std.pdf', bbox_inches="tight")