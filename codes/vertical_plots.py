'''
analyzing each source result from the US Array via vertical plots
'''

from model import *

# load model
model_path = "/home/taufikmh/KAUST/spring_2022/global_pinns/01_clean_implementations/models/pretrained_small_USarray"
# model_path = "/home/taufikmh/KAUST/spring_2022/global_pinns/01_clean_implementations/models/2001_21_5e-06_166_91_181_<class 'torch.nn.modules.activation.ELU'>_False_2734186_512_sphere_gladm25_512_cartesian_<class 'torch.optim.adam.Adam'>_10_US_array_cuda"
figures_path = model_path + '/'
checkpoints_path = figures_path + 'checkpoints' + '/'
predictions_path = figures_path + 'predictions' + '/'

all_models = glob(model_path + '/Model_Epoch_*')

if all_models:
    latest_model = max(all_models, key=os.path.getctime)
else:
    print("It's empty")
    from train import *
    latest_model = max(glob(model_path + '/Model_Epoch_*'), key=os.path.getctime)

model = Model(model_path,VelocityClass=VelocityFunction(),device=torch.device('cpu'))
model.load(latest_model)

# load style
plt.style.use("./science.mplstyle")

# mean prediction
V_pred_lat_mean = 0
V_pred_lon_mean = 0
N = 0

# prediction
for i in range(len(lon_sou)):

    lat_i, alt_i, lon_i = np.meshgrid(latitude, -1e3*depth, longitude)
    x_i, y_i, z_i = pm.geodetic2ecef(lat_i, lon_i, alt_i)
    _, dep_i, _ = np.meshgrid(latitude, depth, longitude)

    # coordinates setup
    sx, sy, sz = pm.geodetic2ecef(lat_sou[i], lon_sou[i], -1e3*dep_sou)

    # rescale
    x,y,z = x_i/(ear_rad*1e3), y_i/(ear_rad*1e3), z_i/(ear_rad*1e3)
    sx, sy, sz = sx/(ear_rad*1e3), sy/(ear_rad*1e3), sz/(ear_rad*1e3)

    X,Y,Z = x[:,latitude.shape[0]//2,:],y[:,latitude.shape[0]//2,:],z[:,latitude.shape[0]//2,:]

    # define receiver coordinates
    xR, yR, zR = X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)

    # define source coordinates
    xS, yS, zS = sx*np.ones_like(X.reshape(-1,1)), sy*np.ones_like(X.reshape(-1,1)), sz*np.ones_like(X.reshape(-1,1))

    # define input to the neural network
    Xo = np.hstack((xS, yS, zS, xR, yR, zR))

    Xq_lat = torch.utils.data.DataLoader(
        torch.from_numpy(Xo).to(torch.float).to(torch.device('cuda')),
        batch_size=int(Xo.shape[0]//10)
    )

    t0 = time.time()
    V_pred_lat = model.velocity(Xq_lat, projection=False, normalization=True, lat_plot=1).cpu().reshape(dep_dim, -1, lon_dim)
    print('Predicted in ' + str(time.time()-t0))
    V_pred_lat = V_pred_lat*bac_vel

    X,Y,Z = x[:,:,0],y[:,:,0],z[:,:,0]

    # define receiver coordinates
    xR, yR, zR = X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)

    # define source coordinates
    xS, yS, zS = sx*np.ones_like(X.reshape(-1,1)), sy*np.ones_like(X.reshape(-1,1)), sz*np.ones_like(X.reshape(-1,1))

    # define input to the neural network
    Xo = np.hstack((xS, yS, zS, xR, yR, zR))

    Xq_lon = torch.utils.data.DataLoader(
        torch.from_numpy(Xo).to(torch.float).to(torch.device('cuda')),
        batch_size=int(Xo.shape[0]//10)
    )

    t0 = time.time()
    V_pred_lon = model.velocity(Xq_lon, projection=False, normalization=True, lat_plot=0).cpu().reshape(dep_dim, lat_dim, -1)
    print('Predicted in ' + str(time.time()-t0))
    V_pred_lon = V_pred_lon*bac_vel

    V_pred_lat_mean = V_pred_lat_mean + V_pred_lat
    V_pred_lon_mean = V_pred_lon_mean + V_pred_lon
    N = N + 1

V_pred_lat_mean /= N
V_pred_lon_mean /= N

# Set up figure and image grid
fig = plt.figure(figsize=(15, 5))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,4),
                 axes_pad=0.15,
#                  share_all=True,
                 cbar_location="bottom",
                 cbar_mode="each",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )
i = 0

D = vpv[:,latitude.shape[0]//2,:].squeeze()
P = V_pred_lat_mean[:,0,:].squeeze()

for ax in grid:
    ax.axis('off')
    if i == 0:
        ax.set_title('Data')
        im1 = ax.pcolormesh(
            xx[:,latitude.shape[0]//2,:].squeeze(),
            yy[:,latitude.shape[0]//2,:].squeeze(),
            D,
            cmap=plt.get_cmap("jet"),
            shading="gouraud",
            vmin=8,
            vmax=14
        )
        # Colorbar
        cbar = ax.cax.colorbar(im1)
        cbar.set_label(r'km.s$^{-1}$')
        ax.cax.toggle_label(True)
    elif i == 1:
        ax.set_title('Prediction')
        im2 = ax.pcolormesh(
            xx[:,latitude.shape[0]//2,:].squeeze(),
            yy[:,latitude.shape[0]//2,:].squeeze(),
            P,
            cmap=plt.get_cmap("jet"),
            shading="gouraud",
            vmin=8,
            vmax=14
        )
        cbar = ax.cax.colorbar(im2)
        cbar.set_label(r'km.s$^{-1}$')
        ax.cax.toggle_label(True)
    elif i == 2:
        ax.set_title('Residual (x100)')
        im3 = ax.pcolormesh(
            xx[:,latitude.shape[0]//2,:].squeeze(),
            yy[:,latitude.shape[0]//2,:].squeeze(),
            100*(P-D),
            cmap=plt.get_cmap("jet"),
            shading="gouraud",
            vmin=-8,
            vmax=8
        )
        # Colorbar
        cbar = ax.cax.colorbar(im3)
        cbar.set_label(r'km.s$^{-1}$')
        ax.cax.toggle_label(True)
    elif i == 3:
        ax.set_title('Relative Residual')
        im3 = ax.pcolormesh(
            xx[:,latitude.shape[0]//2,:].squeeze(),
            yy[:,latitude.shape[0]//2,:].squeeze(),
            100*(P-D)/D,
            cmap=plt.get_cmap("jet"),
            shading="gouraud",
            vmin=-1,
            vmax=1
        )
        # Colorbar
        cbar = ax.cax.colorbar(im3)
        cbar.set_label('\%')
        ax.cax.toggle_label(True)

    ax.set_ylim(0,-ear_rad/1e3)
    i += 1
    
ax0 = fig.add_axes([0.005,0.47,0.18,0.2],projection=ccrs.Orthographic(longitude[25], 0))
ax0.add_feature(cfeature.OCEAN, zorder=0)
ax0.add_feature(cfeature.LAND, zorder=0, edgecolor='white')
ax0.set_global()
ax0.gridlines()
ax0.plot(np.linspace(-180,180,100), np.ones(100)*latitude[latitude.shape[0]//2], 'r',transform=ccrs.PlateCarree(), linewidth=2)
plt.savefig(figures_path + 'vel_DS_lat1.png', bbox_inches="tight")

# Set up figure and image grid
fig = plt.figure(figsize=(15, 5))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,4),
                 axes_pad=0.15,
#                  share_all=True,
                 cbar_location="bottom",
                 cbar_mode="each",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )
i = 0

D = vpv[:,latitude.shape[0]//2,:].squeeze()
P = V_pred_lat_mean[:,0,:].squeeze()

for ax in grid:
    ax.axis('off')
    if i == 0:
        ax.set_title('Data')
        im1 = ax.pcolormesh(
            xx[:,latitude.shape[0]//2,:].squeeze(),
            yy[:,latitude.shape[0]//2,:].squeeze(),
            D,
            cmap=plt.get_cmap("jet"),
            shading="gouraud",
            vmin=8,
            vmax=14
        )
        # Colorbar
        cbar = ax.cax.colorbar(im1)
        cbar.set_label(r'km.s$^{-1}$')
        ax.cax.toggle_label(True)
    elif i == 1:
        ax.set_title('Prediction')
        im2 = ax.pcolormesh(
            xx[:,latitude.shape[0]//2,:].squeeze(),
            yy[:,latitude.shape[0]//2,:].squeeze(),
            P,
            cmap=plt.get_cmap("jet"),
            shading="gouraud",
            vmin=8,
            vmax=14
        )
        # Colorbar
        cbar = ax.cax.colorbar(im2)
        cbar.set_label(r'km.s$^{-1}$')
        ax.cax.toggle_label(True)
    elif i == 2:
        ax.set_title('Residual (x100)')
        im3 = ax.pcolormesh(
            xx[:,latitude.shape[0]//2,:].squeeze(),
            yy[:,latitude.shape[0]//2,:].squeeze(),
            100*(P-D),
            cmap=plt.get_cmap("jet"),
            shading="gouraud",
            vmin=-8,
            vmax=8
        )
        # Colorbar
        cbar = ax.cax.colorbar(im3)
        cbar.set_label(r'km.s$^{-1}$')
        ax.cax.toggle_label(True)
    elif i == 3:
        ax.set_title('Relative Residual')
        im3 = ax.pcolormesh(
            xx[:,latitude.shape[0]//2,:].squeeze(),
            yy[:,latitude.shape[0]//2,:].squeeze(),
            100*(P-D)/D,
            cmap=plt.get_cmap("jet"),
            shading="gouraud",
            vmin=-1,
            vmax=1
        )
        # Colorbar
        cbar = ax.cax.colorbar(im3)
        cbar.set_label('\%')
        ax.cax.toggle_label(True)
        
    ax.set_ylim(0,ear_rad/1e3)
    i += 1
    

ax0 = fig.add_axes([0.005,0.47,0.18,0.2],projection=ccrs.Orthographic(longitude[-25], 0))
ax0.add_feature(cfeature.OCEAN, zorder=0)
ax0.add_feature(cfeature.LAND, zorder=0, edgecolor='white')
ax0.set_global()
ax0.gridlines()
ax0.plot(np.linspace(-180,180,100), np.ones(100)*latitude[latitude.shape[0]//2], 'r',transform=ccrs.PlateCarree(), linewidth=2)
plt.savefig(figures_path + 'vel_DS_lat2.png', bbox_inches="tight")

xx = (ear_rad - DEP) * np.sin(np.radians(LAT+90)) * np.cos(np.radians(180+LON))/(1e3)
yy = (ear_rad - DEP) * np.sin(np.radians(LAT+90)) * np.sin(np.radians(180+LON))/(1e3)
zz = (ear_rad - DEP) * np.cos(np.radians(LAT+90)) / (1e3)

# Set up figure and image grid
fig = plt.figure(figsize=(15, 5))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,4),
                 axes_pad=0.15,
#                  share_all=True,
                 cbar_location="bottom",
                 cbar_mode="each",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )
i = 0

D = vpv[:,:,0].squeeze()
P = V_pred_lon_mean[:,:,0].squeeze()

for ax in grid:
    ax.axis('off')
    if i == 0:
        ax.set_title('Data')
        im1 = ax.pcolormesh(
            zz[:,:,0].squeeze(),
            xx[:,:,0].squeeze(),
            D,
            cmap=plt.get_cmap("jet"),
            shading="gouraud",
            vmin=8,
            vmax=14
        )
        # Colorbar
        cbar = ax.cax.colorbar(im1)
        cbar.set_label(r'km.s$^{-1}$')
        ax.cax.toggle_label(True)
    elif i == 1:
        ax.set_title('Prediction')
        im2 = ax.pcolormesh(
            zz[:,:,0].squeeze(),
            xx[:,:,0].squeeze(),
            P,
            cmap=plt.get_cmap("jet"),
            shading="gouraud",
            vmin=8,
            vmax=14
        )
        # Colorbar
        cbar = ax.cax.colorbar(im2)
        cbar.set_label(r'km.s$^{-1}$')
        ax.cax.toggle_label(True)
    elif i == 2:
        ax.set_title('Residual (x100)')
        im3 = ax.pcolormesh(
            zz[:,:,0].squeeze(),
            xx[:,:,0].squeeze(),
            100*(P-D),
            cmap=plt.get_cmap("jet"),
            shading="gouraud",
            vmin=-8,
            vmax=8
        )
        # Colorbar
        cbar = ax.cax.colorbar(im3)
        cbar.set_label(r'km.s$^{-1}$')
        ax.cax.toggle_label(True)
    elif i == 3:
        ax.set_title('Relative Residual')
        im3 = ax.pcolormesh(
            zz[:,:,0].squeeze(),
            xx[:,:,0].squeeze(),
            100*(P-D)/D,
            cmap=plt.get_cmap("jet"),
            shading="gouraud",
            vmin=-1,
            vmax=1
        )
        # Colorbar
        cbar = ax.cax.colorbar(im3)
        cbar.set_label('\%')
        ax.cax.toggle_label(True)
    
    ax.set_ylim(0,ear_rad/1e3)
    i += 1

ax0 = fig.add_axes([0.005,0.47,0.18,0.2],projection=ccrs.Orthographic(longitude[0], 0))
ax0.add_feature(cfeature.OCEAN, zorder=0)
ax0.add_feature(cfeature.LAND, zorder=0, edgecolor='white')
ax0.set_global()
ax0.gridlines()
ax0.plot(np.ones(100)*longitude[0], np.linspace(-90,90,100), 'r',transform=ccrs.PlateCarree(), linewidth=2)
plt.savefig(figures_path + 'vel_DS_lon1.png', bbox_inches="tight")

# # Set up figure and image grid
# fig = plt.figure(figsize=(15, 5))

# grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
#                  nrows_ncols=(1,4),
#                  axes_pad=0.15,
#                  cbar_location="bottom",
#                  cbar_mode="each",
#                  cbar_size="7%",
#                  cbar_pad=0.15,
#                  )
# i = 0

# D = vpv[:,:,90].squeeze()
# P = V_pred_lon_mean[:,:,90].squeeze()

# for ax in grid:
#     ax.axis('off')
#     if i == 0:
#         ax.set_title('Data')
#         im1 = ax.pcolormesh(
#             zz[:,:,0].squeeze(),
#             xx[:,:,0].squeeze(),
#             D,
#             cmap=plt.get_cmap("jet"),
#             shading="gouraud",
#             vmin=8,
#             vmax=14
#         )
#         # Colorbar
#         cbar = ax.cax.colorbar(im1)
#         cbar.set_label(r'km.s$^{-1}$')
#         ax.cax.toggle_label(True)
#     elif i == 1:
#         ax.set_title('Prediction')
#         im2 = ax.pcolormesh(
#             zz[:,:,0].squeeze(),
#             xx[:,:,0].squeeze(),
#             P,
#             cmap=plt.get_cmap("jet"),
#             shading="gouraud",
#             vmin=8,
#             vmax=14
#         )
#         # Colorbar
#         cbar = ax.cax.colorbar(im2)
#         cbar.set_label(r'km.s$^{-1}$')
#         ax.cax.toggle_label(True)
#     elif i == 2:
#         ax.set_title('Residual (x100)')
#         im3 = ax.pcolormesh(
#             zz[:,:,0].squeeze(),
#             xx[:,:,0].squeeze(),
#             100*(P-D),
#             cmap=plt.get_cmap("jet"),
#             shading="gouraud",
#             vmin=-8,
#             vmax=8
#         )
#         # Colorbar
#         cbar = ax.cax.colorbar(im3)
#         cbar.set_label(r'km.s$^{-1}$')
#         ax.cax.toggle_label(True)
#     elif i == 3:
#         ax.set_title('Relative Residual')
#         im3 = ax.pcolormesh(
#             zz[:,:,0].squeeze(),
#             xx[:,:,0].squeeze(),
#             100*(P-D)/D,
#             cmap=plt.get_cmap("jet"),
#             shading="gouraud",
#             vmin=-1,
#             vmax=1
#         )
#         # Colorbar
#         cbar = ax.cax.colorbar(im3)
#         cbar.set_label('\%')
#         ax.cax.toggle_label(True)
    
#     ax.set_ylim(0,ear_rad/1e3)
#     i += 1

# ax0 = fig.add_axes([0.005,0.47,0.18,0.2],projection=ccrs.Orthographic(longitude[90], 0))
# ax0.add_feature(cfeature.OCEAN, zorder=0)
# ax0.add_feature(cfeature.LAND, zorder=0, edgecolor='white')
# ax0.set_global()
# ax0.gridlines()
# ax0.plot(np.ones(100)*longitude[90], np.linspace(-90,90,100), 'r',transform=ccrs.PlateCarree(), linewidth=2)
# plt.savefig(figures_path + 'vel_DS_lon2.png', bbox_inches="tight")