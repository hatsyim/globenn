'''
neat way of plotting, whatever
@hatsyim
'''

# load predictions
from model import *

if offline:
    # model_path = "/home/taufikmh/KAUST/summer_2021/global_pinns/07_pytorch_implementation/models/bacvel_10_shallow"
    model_path = "/home/taufikmh/KAUST/spring_2022/global_pinns/01_clean_implementations/models/pretrained_USarr_div3_bl21_n512"
    figures_path = model_path + '/'
    checkpoints_path = figures_path + 'checkpoints' + '/'
    predictions_path = figures_path + 'predictions' + '/'

name = '/*.h5'
dire = predictions_path
file = [i for i in os.listdir(dire) if re.search(name, i)]
if len(file) == 0:
    print('We need to test!')
    from save_predictions import *
    
    D = [h5py.File(dire + i, 'r') for i in file]
    P = {file[i].split('/')[-1][:-3] : D[i][file[i].split('/')[-1][:-3]][()] for i in range(len(file))}
elif len(file) != 0:
    all_models = glob(model_path + '/Model_Epoch_*')
    latest_model = max(all_models, key=os.path.getctime)
    model = Model(model_path,VelocityClass=VelocityFunction(),device=torch.device('cpu'))
    model.load(latest_model)

    D = [h5py.File(dire + i, 'r') for i in file]
    P = {file[i].split('/')[-1][:-3] : D[i][file[i].split('/')[-1][:-3]][()] for i in range(len(file))}

print("Loaded")
if bac_vel:
    V_pred, T_pred = P['V_pred']*bac_vel, P['T_pred']/bac_vel
else:
    V_pred, T_pred = ['V_pred'], P['T_pred']

# dep_plt = [
#     depth.flat[np.abs(depth - id).argmin()] for id in np.arange(25,3000,225)
# ]

dep_plt = [depth.flat[np.abs(depth - 24).argmin()]]

dep_plt.append(dep_sou)

trainabel_params = sum(p.numel() for p in model.network.parameters() if p.requires_grad)

# construct xarray data
pred = xr.Dataset({
    'V_pred': xr.DataArray(
        data=V_pred,
        dims=["depth", "latitude", "longitude"],
        coords=dict(
            depth = (["depth"], depth),
            longitude=(["longitude"], longitude),
            latitude=(["latitude"], latitude),
        ),
        attrs=dict(
            long_name='P-wave Velocity Prediction',
            description="Prediction Vp.",
            display_name='Vp (km.s^-1)',
            units="km.s-1",
        )
    ),
    'V_res': xr.DataArray(
        data=(V_pred-data.vpv[dep_ini::dep_inc, lat_ini::lat_inc, lon_ini::lon_inc].values),
        dims=["depth", "latitude", "longitude"],
        coords=dict(
            depth = (["depth"], depth),
            longitude=(["longitude"], longitude),
            latitude=(["latitude"], latitude),
        ),
        attrs=dict(
            long_name='P-wave Velocity Residual',
            description="Error Vp.",
            display_name='Vp (km.s^-1)',
            units="km.s-1",
        )
    ),
    'V_rel': xr.DataArray(
        data=100*(V_pred-data.vpv[dep_ini::dep_inc, lat_ini::lat_inc, lon_ini::lon_inc].values)/data.vpv[dep_ini::dep_inc, lat_ini::lat_inc, lon_ini::lon_inc].values,
        dims=["depth", "latitude", "longitude"],
        coords=dict(
            depth = (["depth"], depth),
            longitude=(["longitude"], longitude),
            latitude=(["latitude"], latitude),
        ),
        attrs=dict(
            long_name='P-wave Velocity Residual',
            description="Relative Error Vp.",
            display_name='Vp/Vp (\%)',
            units="unitless",
        )
    ),
    'T_pred': xr.DataArray(
        data=T_pred*ear_rad,
        dims=["depth", "latitude", "longitude"],
        coords=dict(
            depth = (["depth"], depth),
            longitude=(["longitude"], longitude),
            latitude=(["latitude"], latitude),
        ),
        attrs=dict(
            long_name='First-Arrival Travel Time Prediction',
            description="Prediction T.",
            display_name='T (s)',
            units="s",
        )
    )
})

# load style
plt.style.use("./science.mplstyle")

# 2d
plt.figure()
ax = plt.gca()
im = ax.pcolormesh(
    xx[:,latitude.shape[0]//2,:],
    yy[:,latitude.shape[0]//2,:],
    T_pred[:,latitude.shape[0]//2,:],
    cmap=plt.get_cmap("jet"),
    shading="gouraud"
)
ax.set_aspect('equal')
plt.xlabel(r'Z ($x10^3$ km)')
plt.ylabel(r'X ($x10^3$ km)')
# plt.title('Travel Time from PINNs (All)')
ax.plot(xx_s,yy_s,'w*',markersize=8)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="6%", pad=0.15)
ax.xaxis.set_major_locator(plt.MultipleLocator(2))

cbar = plt.colorbar(im, cax=cax, format='%.1e')
cbar.set_label('s')
cbar.mappable.set_clim(np.nanmin(T_pred), np.nanmax(T_pred))
# cbar.ax.tick_params(labelsize=10)
plt.savefig(figures_path + 'T_PINNs_All_2d_noTit.png', bbox_inches="tight")
# plt.show()

# convergence history plot for verification
plt.figure()
ax = plt.axes()
ax.semilogy(np.arange(len(model.total_train_loss))+1,model.total_train_loss,label='Training')
ax.semilogy(np.arange(len(model.total_train_loss))+1,model.total_val_loss,label='Validation')
ax.set_xlabel('Epochs')
# plt.ylim(0.0001,0)
# plt.xticks(fontsize=10)
ax.legend()
# ax.xaxis.set_major_locator(plt.MultipleLocator(500))
# plt.title('Learning Curve')
ax.set_ylabel('Loss')
# plt.yticks(fontsize=10);
plt.grid()
plt.savefig(figures_path + 'loss_noTit.png', bbox_inches="tight")

# 3d plot

v_min, v_max = np.min(vel_all), np.max(vel_all)

fig = plt.figure(figsize=(8,8), dpi=300)
ax = fig.add_subplot(projection='3d')
ax.set_title('Velocity (All)')
xs = x_plot
ys = y_plot
zs = z_plot
im = ax.scatter(
    xs, 
    ys, 
    zs, 
    marker='o', 
    c=vel_all.reshape(-1,1), 
    cmap='jet',
    edgecolors='none'
)
cbar = plt.colorbar(
    im,
    orientation="horizontal", 
    shrink=0.5,
    pad=0.05
)
cbar.set_label(r'km.s$^{-1}$')
cbar.mappable.set_clim(v_min, v_max)

# Get rid of colored axes planes
# First remove fill
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

# Bonus: To get rid of the grid as well:
ax.grid(False)

ax.xaxis.set_major_locator(plt.MultipleLocator(3))
ax.yaxis.set_major_locator(plt.MultipleLocator(3))
ax.zaxis.set_major_locator(plt.MultipleLocator(3))

ax.set_xlabel(r'X ($x10^3$ km)')
ax.set_ylabel(r'Y ($x10^3$ km)')
ax.set_zlabel(r'Z ($x10^3$ km)')
ax.set_xlim((np.min(x_plot), np.max(x_plot)))
ax.set_ylim((np.min(y_plot), np.max(y_plot)))
ax.set_zlim((np.min(z_plot), np.max(z_plot)))


plt.savefig(figures_path + 'V_ini_all_3d_noTit.png', bbox_inches='tight')
# plt.show()

plt.figure()
ax = plt.gca()
im = ax.pcolormesh(
    xx[:,latitude.shape[0]//2,:],
    yy[:,latitude.shape[0]//2,:],
    V_pred[:,latitude.shape[0]//2,:],
    cmap=plt.get_cmap("jet"),
    shading="gouraud"
)
ax.set_aspect('equal')
plt.xlabel(r'Z ($x10^3$ km)')
plt.ylabel(r'X ($x10^3$ km)')
# plt.title('Predicted Velocity (All)')
ax.plot(xx_s,yy_s,'w*',markersize=8)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="6%", pad=0.15)
ax.xaxis.set_major_locator(plt.MultipleLocator(2))

cbar = plt.colorbar(im, cax=cax, format='%.1e')
cbar.set_label(r'km.s$^{-1}$')
cbar.mappable.set_clim(6, 14)
# cbar.ax.tick_params(labelsize=10)
plt.savefig(figures_path + 'V_pred_All_2d_noTit.png', bbox_inches="tight")
# plt.show()


plt.figure()
ax = plt.gca()
im = ax.pcolormesh(
    xx[:,latitude.shape[0]//2,:],
    yy[:,latitude.shape[0]//2,:],
    np.abs(vel_all-V_pred)[:,latitude.shape[0]//2,:],
    cmap=plt.get_cmap("jet"),
    shading="gouraud"
)
ax.set_aspect('equal')
plt.xlabel(r'Z ($x10^3$ km)')
plt.ylabel(r'X ($x10^3$ km)')
# plt.title('Absolute Velocity Residual (All)')
ax.plot(xx_s,yy_s,'w*',markersize=8)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="6%", pad=0.15)
ax.xaxis.set_major_locator(plt.MultipleLocator(2))

cbar = plt.colorbar(im, cax=cax, format='%.1e')
cbar.set_label(r'km.s$^{-1}$')
cbar.mappable.set_clim(0, 0.7)
# cbar.ax.tick_params(labelsize=10)
plt.savefig(figures_path + 'V_res_All_2d_noTit.png', bbox_inches="tight")
# plt.show()

plt.figure()
ax = plt.gca()
im = ax.pcolormesh(
    xx[:,latitude.shape[0]//2,:],
    yy[:,latitude.shape[0]//2,:],
    vel_all[:,latitude.shape[0]//2,:],
    cmap=plt.get_cmap("jet"),
    shading="gouraud"
)
ax.set_aspect('equal')
plt.xlabel(r'Z ($x10^3$ km)')
plt.ylabel(r'X ($x10^3$ km)')
# plt.title('Training Velocity')
ax.plot(xx_s,yy_s,'w*',markersize=8)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="6%", pad=0.15)
ax.xaxis.set_major_locator(plt.MultipleLocator(2))

cbar = plt.colorbar(im, cax=cax, format='%.1e')
cbar.set_label(r'km.s$^{-1}$')
cbar.mappable.set_clim(6, 14)
# cbar.mappable.set_clim(10, 14)
# cbar.ax.tick_params(labelsize=10)
plt.savefig(figures_path + 'V_train_2d_noTit.png', bbox_inches="tight")
# plt.show()

plt.figure()
ax = plt.gca()
im = ax.pcolormesh(
    xx[:,latitude.shape[0]//2,:],
    yy[:,latitude.shape[0]//2,:],
    vel_all[:,latitude.shape[0]//2,:],
    cmap=plt.get_cmap("jet"),
    shading="gouraud"
)
ax.set_aspect('equal')
plt.xlabel(r'Z ($x10^3$ km)')
plt.ylabel(r'X ($x10^3$ km)')
# plt.title('GLAD-M25 Velocity')
ax.plot(xx_s,yy_s,'w*',markersize=8)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="6%", pad=0.15)
ax.xaxis.set_major_locator(plt.MultipleLocator(2))

cbar = plt.colorbar(im, cax=cax, format='%.1e')
cbar.set_label(r'km.s$^{-1}$')
cbar.mappable.set_clim(6, 14)
# cbar.ax.tick_params(labelsize=10)
plt.savefig(figures_path + 'V_all_2d_noTit.png', bbox_inches="tight")
# plt.show()

for ip in dep_plt:

    fig = plt.figure(figsize=(4,4), dpi=300)
    ax = plt.axes(projection=ccrs.Robinson(180))
    ax.coastlines()
    ax.gridlines()
    data.vpv[dep_ini::dep_inc, lat_ini::lat_inc, lon_ini::lon_inc][depth==ip].plot(
        ax=ax, 
        transform=ccrs.PlateCarree(), 
        cbar_kwargs={'shrink': 0.5, 'extend':'neither', 'orientation':'horizontal','pad':0.05}, 
        cmap='jet',
        vmin=np.min(pred.V_pred[depth==ip]),
        vmax=data.vpv[dep_ini::dep_inc, lat_ini::lat_inc, lon_ini::lon_inc][depth==ip].values.max()
    )

    plt.plot(lon_sou, lat_sou,color='black', markersize=10, marker='*',transform=ccrs.PlateCarree(), mec='white')
    plt.savefig(figures_path + 'V_ini_map_'+str(ip)+'_noTit.png', bbox_inches="tight")
    # plt.show()

    fig = plt.figure(figsize=(4,4), dpi=300)
    ax = plt.axes(projection=ccrs.Robinson(180))

    # print(pred.T_pred[depth==ip].values.min(), dep_sou)

    ax.coastlines()
    ax.gridlines()
    pred.T_pred[depth==ip].plot(
        ax=ax, 
        transform=ccrs.PlateCarree(), 
        cbar_kwargs={'shrink': 0.5, 'extend':'neither', 'orientation':'horizontal','pad':0.05}, 
        cmap='jet',
        vmin=pred.T_pred[depth==ip].values.min(),
        vmax=pred.T_pred[depth==ip].values.max()
    )
    plt.plot(lon_sou, lat_sou,color='black', markersize=10, marker='*',transform=ccrs.PlateCarree(), mec='white')
    plt.savefig(figures_path + 'T_pred_map_'+str(ip)+'_noTit.png', bbox_inches="tight")
    # plt.show()

    fig = plt.figure(figsize=(4,4), dpi=300)
    ax = plt.axes(projection=ccrs.Robinson(180))

    ax.coastlines()
    ax.gridlines()
    pred.V_pred[depth==ip].plot(
        ax=ax, 
        transform=ccrs.PlateCarree(), 
        cbar_kwargs={'shrink': 0.5, 'extend':'neither', 'orientation':'horizontal','pad':0.05}, 
        cmap='jet',
        vmin=np.min(pred.V_pred[depth==ip]),
        vmax=data.vpv[dep_ini::dep_inc, lat_ini::lat_inc, lon_ini::lon_inc][depth==ip].values.max()
    )
    plt.plot(lon_sou, lat_sou,color='black', markersize=10, marker='*',transform=ccrs.PlateCarree(), mec='white')
    plt.savefig(figures_path + 'V_pred_map_'+str(ip)+'_noTit.png', bbox_inches="tight")
    # plt.show()
    
    fig = plt.figure(figsize=(4,4), dpi=300)
    ax = plt.axes(projection=ccrs.Robinson(180))

    ax.coastlines()
    ax.gridlines()
    pred.V_res[depth==ip].plot(
        ax=ax, 
        transform=ccrs.PlateCarree(), 
        cbar_kwargs={'shrink': 0.5, 'extend':'neither', 'orientation':'horizontal','pad':0.05}, 
        cmap='jet'
    )
    plt.plot(lon_sou, lat_sou,color='black', markersize=10, marker='*',transform=ccrs.PlateCarree(), mec='white')
    plt.savefig(figures_path + 'V_res_map_'+str(ip)+'_noTit.png', bbox_inches="tight")
    # plt.show()

    fig = plt.figure(figsize=(4,4), dpi=300)
    ax = plt.axes(projection=ccrs.Robinson(180))

    ax.coastlines()
    ax.gridlines()
    pred.V_rel[depth==ip].plot(
        ax=ax, 
        transform=ccrs.PlateCarree(), 
        cbar_kwargs={'shrink': 0.5, 'extend':'neither', 'orientation':'horizontal','pad':0.05}, 
        cmap='jet'
    )
    plt.plot(lon_sou, lat_sou,color='black', markersize=10, marker='*',transform=ccrs.PlateCarree(), mec='white')
    plt.savefig(figures_path + 'V_rel_map_'+str(ip)+'_noTit.png', bbox_inches="tight")
    # plt.show()

    fig = plt.figure(figsize=(4,4), dpi=300)
    D = pred.V_res[depth==ip].values.reshape(-1,)
    plt.hist(D, weights=np.ones(len(D)) / len(D))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.title('Relative Error at Depth '+str(ip)+' (km)')
    plt.ylabel('Percentage (\%)')
    plt.xlabel(r'Error (km.s$^{-1}$)')
    plt.savefig(figures_path + 'V_hist_'+str(ip)+'_noTit.png', bbox_inches="tight")

    fig = plt.figure(figsize=(4,4), dpi=300)
    D = pred.V_rel[depth==ip].values.reshape(-1,)
    plt.hist(D, weights=np.ones(len(D)) / len(D))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.title('Relative Error at Depth '+str(ip)+' (km)')
    plt.ylabel('Percentage (\%)')
    plt.xlabel('Error (\%)')
    plt.xlim(-15,15)
    # plt.xlim(-1.4,1)
    plt.savefig(figures_path + 'V_hist_rel_'+str(ip)+'_noTit.png', bbox_inches="tight")




    # plt.figure()
    # plt.plot(depth, V_pred[:,latitude==lat_sou,longitude==lon_sou], ':', label='Predicted')
    # plt.plot(depth, vpv[:,latitude==lat_sou,longitude==lon_sou], '--', label='Trained')
    # plt.axvline(x=ip, label='Slice')
    # # plt.title('1-D Profile Comparison')
    # plt.xlabel('Depth (km)')
    # plt.ylabel(r'Vp (km.s$^{-1}$)')
    # plt.legend()
    # plt.savefig(figures_path + 'V_comp_1d_'+str(ip)+'_noTit.png', bbox_inches="tight")

    fig = plt.figure(figsize=(4,4), dpi=300)
    P = pred.V_pred[depth==ip].values.reshape(-1,)
    D = data.vpv[dep_ini::dep_inc, lat_ini::lat_inc, lon_ini::lon_inc].values[depth==ip].reshape(-1,)
    n_P, bins_P, patches_P = plt.hist(
        P, 
        weights=np.ones(len(P)) / len(P), 
        bins=50, 
        histtype='step', 
        range=(D.min(), D.max()), 
        label='Prediction'
    )
    n_D, bins_D, patches_D = plt.hist(
        D, 
        weights=np.ones(len(P)) / len(P), 
        bins=50, 
        histtype='step', 
        range=(D.min(), D.max()), 
        label='Data'
    )
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # # plt.title('Velocity Histogram')
    # plt.title('Velocity Histogram at Depth '+str(ip)+' (km)')
    plt.ylabel('Percentage (\%)')
    plt.xlabel(r'Vp (km.s$^{-1}$)')
    plt.legend()
    plt.ylim(0,0.09)
    plt.text(
        np.min([D,P]),# +0.5*(np.max([D,P])-np.min([D,P])),
        0.07, 
        "CS="+str(round(1 - spatial.distance.cosine(n_D, n_P),3))
    )
    plt.savefig(figures_path + 'V_comp_hist_'+str(ip)+'_noTit.png', bbox_inches="tight")

fig = plt.figure(figsize=(4,4), dpi=300)
P = pred.V_pred.values.reshape(-1,)
D = data.vpv[dep_ini::dep_inc, lat_ini::lat_inc, lon_ini::lon_inc].values.reshape(-1,)
n_P, bins_P, patches_P = plt.hist(
    P, 
    weights=np.ones(len(P)) / len(P), 
    bins=50, 
    histtype='step', 
    range=(D.min(), D.max()), 
    label='Prediction'
)
n_D, bins_D, patches_D = plt.hist(
    D, 
    weights=np.ones(len(P)) / len(P), 
    bins=50, 
    histtype='step', 
    range=(D.min(), D.max()), 
    label='Data'
)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Velocity Histogram')
plt.ylabel('Percentage (\%)')
plt.xlabel(r'Vp (km.s$^{-1}$)')
plt.legend()
plt.ylim(0,0.15)
plt.text(
    np.max([D,P])-0.3*(np.max([D,P])-np.min([D,P])),
    0.12, 
    "CS="+str(round(1 - spatial.distance.cosine(n_D, n_P),3))
    )
plt.savefig(figures_path + 'V_comp_hist_all_noTit.png', bbox_inches="tight")

fig = plt.figure(figsize=(4,4), dpi=300)
D = pred.V_rel.values.reshape(-1,)
plt.hist(D, weights=np.ones(len(D)) / len(D))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title('Relative Error')
plt.ylabel('Percentage (\%)')
plt.xlabel('Error (\%)')
plt.savefig(figures_path + 'V_hist_rel_noTit.png', bbox_inches="tight")

colnames=['depth', 'vp', 'vs', 'rho']
ek137 = pd.read_csv('../../00_data/ek137.tvel', skiprows=2, header=None, delim_whitespace=1, names=colnames)
depth_ek = ek137.depth.values
vp_ek = ek137.vp.values

# 1D Validation
vp_min = [np.min(vpv[i,:,:]) for i in range(len(depth))]
vp_max = [np.max(vpv[i,:,:]) for i in range(len(depth))]
vp_mean = [np.mean(vpv[i,:,:]) for i in range(len(depth))]
plt.figure()
plt.plot(depth, V_pred[:,0,0], label='Predicted')
plt.plot(depth, vpv[:,0,0], '.', label='Trained')
plt.plot(depth, vp_mean, '.', label='Average')
plt.plot(depth, vp_max, '.', label='Max')
plt.plot(depth, vp_min, '.', label='Min')
plt.plot(depth_ek[depth_ek<=np.max(depth)], vp_ek[depth_ek<=np.max(depth)], label='1-D EK137')
# plt.title('1-D Profile Comparison')
plt.xlabel('Depth (km)')
plt.ylabel(r'Vp (km.s$^{-1}$)')
plt.legend()
plt.savefig(figures_path + 'V_comp_1d_noTit.png', bbox_inches="tight")

fig, ax = plt.subplots(1)
ax.plot(depth, [np.mean(V_pred[i,:,:]) for i in range(len(depth))], label='Mean Prediction')
ax.plot(depth, vp_mean, '--', label='Mean GLAD-M25')
ax.plot(depth, vp_max, '-.', label='Max GLAD-M25')
ax.plot(depth, vp_min, ':', label='Min GLAD-M25')
ax.plot(depth_ek[depth_ek<=np.max(depth)], vp_ek[depth_ek<=np.max(depth)], 'g', label='1-D EK137')
ax.set_title('1-D Profile Comparison')
ax.set_xlabel('Depth (km)')
ax.set_ylabel(r'Vp (km.s$^{-1}$)')
ax.legend(bbox_to_anchor=(1., 1.))

axins = zoomed_inset_axes(ax,3,loc='lower right')
axins.plot(depth, [np.mean(V_pred[i,:,:]) for i in range(len(depth))], label='Mean Prediction')
axins.plot(depth, vp_mean, '--', label='Mean GLAD-M25')
axins.plot(depth, vp_max, '-.', label='Max GLAD-M25')
axins.plot(depth, vp_min, ':', label='Min GLAD-M25')
axins.plot(depth_ek[depth_ek<=np.max(depth)], vp_ek[depth_ek<=np.max(depth)], 'g', label='1-D EK137')
x1,x2,y1,y2 = 0,400,5,9
axins.set_xlim(x1,x2)
axins.set_ylim(y1,y2)
axins.set_xticks([])
axins.set_yticks([])

mark_inset(ax,axins,loc1=2,loc2=3)

plt.savefig(figures_path + 'V_comp_1d_box1_noTit.png', bbox_inches="tight")

fig, ax = plt.subplots(1)
ax.plot(depth, [np.mean(V_pred[i,:,:]) for i in range(len(depth))], label='Mean Prediction')
ax.plot(depth, vp_mean, '--', label='Mean GLAD-M25')
ax.plot(depth, vp_max, '-.', label='Max GLAD-M25')
ax.plot(depth, vp_min, ':', label='Min GLAD-M25')
ax.plot(depth_ek[depth_ek<=np.max(depth)], vp_ek[depth_ek<=np.max(depth)], 'g', label='1-D EK137')
ax.set_title('1-D Profile Comparison')
ax.set_xlabel('Depth (km)')
ax.set_ylabel(r'Vp (km.s$^{-1}$)')
ax.legend(bbox_to_anchor=(1., 1.))

axins = zoomed_inset_axes(ax,3,loc='lower right')
axins.plot(depth, [np.mean(V_pred[i,:,:]) for i in range(len(depth))], label='Mean Prediction')
axins.plot(depth, vp_mean, '--', label='Mean GLAD-M25')
axins.plot(depth, vp_max, '-.', label='Max GLAD-M25')
axins.plot(depth, vp_min, ':', label='Min GLAD-M25')
axins.plot(depth_ek[depth_ek<=np.max(depth)], vp_ek[depth_ek<=np.max(depth)], 'g', label='1-D EK137')
x1,x2,y1,y2 = 200,600,7,11
axins.set_xlim(x1,x2)
axins.set_ylim(y1,y2)
axins.set_xticks([])
axins.set_yticks([])

mark_inset(ax,axins,loc1=2,loc2=3)

plt.savefig(figures_path + 'V_comp_1d_box2_noTit.png', bbox_inches="tight")

fig, ax = plt.subplots(1)
ax.plot(depth, [np.mean(V_pred[i,:,:]) for i in range(len(depth))], label='Mean Prediction')
ax.plot(depth, vp_mean, '--', label='Mean GLAD-M25')
ax.plot(depth, vp_max, '-.', label='Max GLAD-M25')
ax.plot(depth, vp_min, ':', label='Min GLAD-M25')
ax.plot(depth_ek[depth_ek<=np.max(depth)], vp_ek[depth_ek<=np.max(depth)], 'g', label='1-D EK137')
ax.set_title('1-D Profile Comparison')
ax.set_xlabel('Depth (km)')
ax.set_ylabel(r'Vp (km.s$^{-1}$)')
ax.legend(bbox_to_anchor=(1., 1.))

axins = zoomed_inset_axes(ax,3,loc='lower right')
axins.plot(depth, [np.mean(V_pred[i,:,:]) for i in range(len(depth))], label='Mean Prediction')
axins.plot(depth, vp_mean, '--', label='Mean GLAD-M25')
axins.plot(depth, vp_max, '-.', label='Max GLAD-M25')
axins.plot(depth, vp_min, ':', label='Min GLAD-M25')
axins.plot(depth_ek[depth_ek<=np.max(depth)], vp_ek[depth_ek<=np.max(depth)], 'g', label='1-D EK137')
x1,x2,y1,y2 = 500,900,8,12
axins.set_xlim(x1,x2)
axins.set_ylim(y1,y2)
axins.set_xticks([])
axins.set_yticks([])

mark_inset(ax,axins,loc1=2,loc2=3)

plt.savefig(figures_path + 'V_comp_1d_box3_noTit.png', bbox_inches="tight")

fig, ax = plt.subplots(1)
ax.plot(depth, [np.mean(V_pred[i,:,:]) for i in range(len(depth))], label='Mean Prediction')
ax.plot(depth, vp_mean, '--', label='Mean GLAD-M25')
ax.plot(depth, vp_max, '-.', label='Max GLAD-M25')
ax.plot(depth, vp_min, ':', label='Min GLAD-M25')
ax.plot(depth_ek[depth_ek<=np.max(depth)], vp_ek[depth_ek<=np.max(depth)], 'g', label='1-D EK137')
ax.set_title('1-D Profile Comparison')
ax.set_xlabel('Depth (km)')
ax.set_ylabel(r'Vp (km.s$^{-1}$)')
ax.legend(bbox_to_anchor=(1., 1.))

plt.savefig(figures_path + 'V_comp_1d_plain_noTit.png', bbox_inches="tight")

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
P = V_pred[:,latitude.shape[0]//2,:].squeeze()

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
plt.savefig(figures_path + 'vel_DS_lat1_noTit.png', bbox_inches="tight")

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
P = V_pred[:,latitude.shape[0]//2,:].squeeze()

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
plt.savefig(figures_path + 'vel_DS_lat2_noTit.png', bbox_inches="tight")

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
P = V_pred[:,:,0].squeeze()

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
plt.savefig(figures_path + 'vel_DS_lon1_noTit.png', bbox_inches="tight")

# Set up figure and image grid
fig = plt.figure(figsize=(15, 5))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,4),
                 axes_pad=0.15,
                 cbar_location="bottom",
                 cbar_mode="each",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )
i = 0

D = vpv[:,:,90].squeeze()
P = V_pred[:,:,90].squeeze()

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

ax0 = fig.add_axes([0.005,0.47,0.18,0.2],projection=ccrs.Orthographic(longitude[90], 0))
ax0.add_feature(cfeature.OCEAN, zorder=0)
ax0.add_feature(cfeature.LAND, zorder=0, edgecolor='white')
ax0.set_global()
ax0.gridlines()
ax0.plot(np.ones(100)*longitude[90], np.linspace(-90,90,100), 'r',transform=ccrs.PlateCarree(), linewidth=2)
plt.savefig(figures_path + 'vel_DS_lon2_noTit.png', bbox_inches="tight")