'''
perform geodetic to geocentric coordinate transformation
@hatsyim
'''

# chain
from data_loading import *

LAT, ALT, LON = np.meshgrid(latitude, -1e3*depth, longitude)
x, y, z = pm.geodetic2ecef(LAT, LON, ALT)
_, DEP, _ = np.meshgrid(latitude, depth, longitude)
