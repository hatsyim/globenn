'''
the beginning of my journey to use argparse
@hatsyim
'''

# import
from projection import *

# laod all stations
ISCall = pd.read_csv('stations.csv')
ISCall = ISCall.rename(columns={"X":"LON", 'Y':'LAT'})

# laod only active stations (from ISC website) to 2021
ISCarray = ISCall[ISCall['description'].str.contains('to 2021')]
ISCarrLon = ISCarray['LON'][::5]
ISCarrLat = ISCarray['LAT'][::5]

# load US array data
USarray = pd.read_excel('_US-TA-StationList.xls')
USarrLon = USarray['LON'][::1]
USarrLat = USarray['LAT'][::1]

# concatenate the two receiver group
AllLon = np.hstack((USarrLon, ISCarrLon))
AllLat = np.hstack((USarrLat, ISCarrLat))

# model specifications
ear_rad = 6371

rec_typ = 'US_array'

if rec_typ == 'ISC_array':
    print(rec_typ)
    lat_sou = np.array([latitude.flat[np.abs(latitude - i).argmin()] for i in ISCarrLat])
    lon_sou = np.array([longitude.flat[np.abs(longitude - i).argmin()] for i in ISCarrLon])
elif rec_typ == 'US_array':
    print(rec_typ)
    lat_sou = np.array([latitude.flat[np.abs(latitude - i).argmin()] for i in USarrLat])
    lon_sou = np.array([longitude.flat[np.abs(longitude - i).argmin()] for i in USarrLon])
dep_sou = depth.flat[np.abs(depth - 0).argmin()]

xx = (ear_rad - DEP) * np.sin(np.radians(LAT+90)) * np.cos(np.radians(180+LON))/(1e3)
yy = (ear_rad - DEP) * np.sin(np.radians(LAT+90)) * np.sin(np.radians(180+LON))/(1e3)
zz = DEP * np.cos(np.radians(LAT+90)) / (1e3)

xx_s = (ear_rad - dep_sou) * np.sin(np.radians(lat_sou+90)) * np.cos(np.radians(180+lon_sou))/(1e3)
yy_s = (ear_rad - dep_sou) * np.sin(np.radians(lat_sou+90)) * np.sin(np.radians(180+lon_sou))/(1e3)

# coordinates setup
sx, sy, sz = pm.geodetic2ecef(lat_sou, lon_sou, -1e3*dep_sou)

# rescale
x,y,z = x/(ear_rad*1e3), y/(ear_rad*1e3), z/(ear_rad*1e3)
sx, sy, sz = sx/(ear_rad*1e3), sy/(ear_rad*1e3), sz/(ear_rad*1e3)

X,Y,Z = x,y,z

sou_idx = np.where((np.isclose(x.reshape(-1,1), sx)) & (np.isclose(y.reshape(-1,1), sy)) & (np.isclose(z.reshape(-1,1), sz)))[0]
sx, sy, sz = x.reshape(-1,1)[sou_idx], y.reshape(-1,1)[sou_idx], z.reshape(-1,1)[sou_idx]

# for plotting only
x_plot,y_plot,z_plot = x.reshape(-1,1)*ear_rad/1000, y.reshape(-1,1)*ear_rad/1000, z.reshape(-1,1)*ear_rad/1000

num_pts = x.size

try_first = False
single_source = True
permutate = True

# saving parameters
num_epo = int(2001)
num_blo = 21 #20
coo_sys = 'cartesian'
vel_sha = 'sphere'
vel_typ = 'gladm25'
num_neu = 512
lea_rat = 5e-6
act_fun = torch.nn.ELU
bat_siz = 512 #num_pts // 100
ada_wei = False
vel_sca = 1
opt_fun = torch.optim.Adam
dev_typ = "cuda"
nor_typ = "MinMax"
bac_vel = 10 #6 #10.5
hyp_par = (
    str(num_epo) + '_' +
    str(num_blo) + '_' +
    str(lea_rat) + '_' +
    str(dep_dim) + '_' +
    str(lat_dim) + '_' +
    str(lon_dim) + '_' +
    str(act_fun) + '_' +
    str(ada_wei) + '_' +
    str(num_pts) + '_' +
    str(bat_siz) + '_' +
    vel_sha + '_' +
    vel_typ + '_' +
    str(num_neu) + '_' +
    coo_sys + '_' +
    str(vel_sca) + '_' +
    str(opt_fun) + '_' +
    nor_typ + '_' +
    str(bac_vel) + '_' +
    rec_typ + '_' +
    dev_typ
)

# path
model_path = "./../models/" + hyp_par
figures_path = model_path + '/'
checkpoints_path = figures_path + 'checkpoints' + '/'
predictions_path = figures_path + 'predictions' + '/'

Path(figures_path).mkdir(parents=True, exist_ok=True)
Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
Path(predictions_path).mkdir(parents=True, exist_ok=True)

