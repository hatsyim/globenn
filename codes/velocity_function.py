'''
create 3d radially increasing velocity for PINNs toy model
@hatsyim
'''

# chain imports
from model_parameters import *

def velocity_function(vel_sha='sphere', vel_typ='homogeneous', vel_ini=5, x=X, y=Y, z=Z):
    if vel_sha=='sphere':
        # inhomogeneous velocity without gaps
        vel_inh = vel_ini - (x**2 + z**2 + y**2)
        vel_inh[vel_inh<2.**2] = np.NaN
        if vel_typ=='inhomogeneous':  
            # velocity without gaps
            vel_all = np.copy(vel_inh)
            
            # velocity with gaps
            vel_gap = np.copy(vel_all)
            vel_gap[vel_inh>2.1**2] = np.NaN
        elif vel_typ == 'random':
            # velocity without gaps
            from scipy.ndimage import gaussian_filter
            vel_all = gaussian_filter(18*np.random.random((nx,ny,nz))**3, sigma=6)
            vel_all[np.isnan(vel_inh)] = np.NaN

            # velocity with gaps
            vel_gap = np.copy(vel_all)
            vel_gap[vel_inh>2.1**2] = np.NaN
        elif vel_typ == 'homogeneous':
            # velocity without gaps
            vel_all = vel_ini*np.ones_like(vel_inh)
            vel_all[np.isnan(vel_inh)] = np.NaN
            
            # velocity with gaps
            vel_tmp = np.copy(vel_inh)
            vel_tmp[vel_inh>2.1**2] = np.NaN
            vel_gap = np.copy(vel_all)
            vel_gap[np.isnan(vel_tmp)] = np.NaN
        elif vel_typ == 'radial':
            vel_inh = vp
            vel_all = vp
            vel_gap = vp_gap
        elif vel_typ == 'constant_radial':
            vel_inh = 5*np.ones_like(vp)
            vel_all = 5*np.ones_like(vp)
            vel_gap = np.where(DEP<3000, vel_all, np.nan)
        elif vel_typ == 'gladm25':
            vel_inh = vpv
            vel_all = vpv
            vel_gap = vpv
            
            
    return vel_inh, vel_all, vel_gap

# call function
vel_inh, vel_all, vel_gap = velocity_function(vel_typ=vel_typ)