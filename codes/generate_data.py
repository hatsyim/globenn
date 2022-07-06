'''
generating a first-ever 3d reference global Earth models
@hatsyim
'''

# chain
from modules_import import *

# load data
colnames=['depth', 'vp', 'vs', 'rho']
ek137 = pd.read_csv('../../00_data/ek137.tvel', skiprows=2, header=None, delim_whitespace=1, names=colnames)

dep_dim = len(ek137.depth.values[::2])
lat_dim = 46
lon_dim = 91

# variables
depth = ek137.depth.values[::2]
latitude = np.linspace(-90,90,lat_dim)
longitude = np.linspace(-180,180,lon_dim)

# generating cube
vp = np.ones((depth.shape[0], latitude.shape[0], longitude.shape[0]))
vp_gap = np.copy(vp)
vs = np.copy(vp)
rho = np.copy(vp)

for i in range(len(depth)):
    vp[i,:,:] = ek137.vp[i]
    vs[i,:,:] = ek137.vs[i]
    rho[i,:,:] = ek137.rho[i]
    if depth[i] < 3000:
        vp_gap[i,:,:] = ek137.vp[i]
    else:
        vp_gap[i,:,:] = np.NaN

# construct xarray data for plotting
data = xr.Dataset({
    'vp': xr.DataArray(
        data=vp,
        dims=["depth", "latitude", "longitude"],
        coords=dict(
            depth = (["depth"], depth),
            longitude=(["longitude"], np.linspace(-180,180,lon_dim)),
            latitude=(["latitude"], np.linspace(-90,90,lat_dim)),
        ),
        attrs=dict(
            long_name='P-Wave Radial Velocity',
            description="P-Wave Velocity",
            display_name='Vp (km/s)',
            units="km/s",
        )
    ),
    'vp_gap': xr.DataArray(
        data=vp_gap,
        dims=["depth", "latitude", "longitude"],
        coords=dict(
            depth = (["depth"], depth),
            longitude=(["longitude"], np.linspace(-180,180,lon_dim)),
            latitude=(["latitude"], np.linspace(-90,90,lat_dim)),
        ),
        attrs=dict(
            long_name='P-Wave Radial Velocity',
            description="P-Wave Velocity with Gaps",
            display_name='Vp (km/s)',
            units="km/s",
        )
    ),
    'vs': xr.DataArray(
        data=vs,
        dims=["depth", "latitude", "longitude"],
        coords=dict(
            depth = (["depth"], depth),
            longitude=(["longitude"], np.linspace(-180,180,lon_dim)),
            latitude=(["latitude"], np.linspace(-90,90,lat_dim)),
        ),
        attrs=dict(
            long_name='S-Wave Radial Velocity',
            description="S-Wave Velocity",
            display_name='Vs (km/s)',
            units="km/s",
        )
    ),
    'rho': xr.DataArray(
        data=rho,
        dims=["depth", "latitude", "longitude"],
        coords=dict(
            depth = (["depth"], depth),
            longitude=(["longitude"], np.linspace(-180,180,lon_dim)),
            latitude=(["latitude"], np.linspace(-90,90,lat_dim)),
        ),
        attrs=dict(
            long_name='Density',
            description="Density",
            display_name='Rho (Mg/m^3)',
            units="Mg/m^3",
        )
    )
})
