import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import datetime
import random
import xarray as xr
import cartopy.crs as ccrs
import netCDF4
import matplotlib.pylab as pylab
import matplotlib.colors as mcolors
import h5py
import argparse
import pymap3d as pm
import os
import time
import math
import torch
import copy
import re
import matplotlib.patches as patches
import cartopy.feature as cfeature

from scipy import signal
from scipy import spatial
from torch.nn import Linear
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from pyproj import Proj
from scipy.ndimage.filters import gaussian_filter
from glob import glob
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
from mpl_toolkits.axes_grid1 import ImageGrid

# for reproducibility
os.environ['PYTHONHASHSEED']= '123'
np.random.seed(123)

# use custom Matplotlib style
plt.style.use("./science.mplstyle")

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

# geodetic-geocentric projection
LAT, ALT, LON = np.meshgrid(latitude, -1e3*depth, longitude)
x, y, z = pm.geodetic2ecef(LAT, LON, ALT)
_, DEP, _ = np.meshgrid(latitude, depth, longitude)

# multi-receivers options
rec_typ = 'US_array'

# saving parameters
num_pts = x.size
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
opt_fun = torch.optim.Adam
dev_typ = "cuda"
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
    str(opt_fun) + '_' +
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

# laod all stations
ISCall = pd.read_csv('stations.csv')
ISCall = ISCall.rename(columns={"X":"LON", 'Y':'LAT'})

# laod only active stations (from ISC website) to 2021
ISCarray = ISCall[ISCall['description'].str.contains('to 2021')]
ISCarrLon = ISCarray['LON']
ISCarrLat = ISCarray['LAT']

# load US array data
USarray = pd.read_excel('_US-TA-StationList.xls')
USarrLon = USarray['LON']
USarrLat = USarray['LAT']

# concatenate the two receiver group
AllLon = np.hstack((USarrLon, ISCarrLon))
AllLat = np.hstack((USarrLat, ISCarrLat))

# model specifications
ear_rad = 6371

# source vectors
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
X,Y,Z = x/(ear_rad*1e3), y/(ear_rad*1e3), z/(ear_rad*1e3)
sx, sy, sz = sx/(ear_rad*1e3), sy/(ear_rad*1e3), sz/(ear_rad*1e3)

# finding source indices
sids = np.where((np.isclose(X.reshape(-1,1), sx, atol=1e-16)) & (np.isclose(Y.reshape(-1,1), sy, atol=1e-16)) & (np.isclose(Z.reshape(-1,1), sz, atol=1e-16)))[0]
sx, sy, sz = X.reshape(-1,1)[sids], Y.reshape(-1,1)[sids], Z.reshape(-1,1)[sids]
sids = sids[sids.astype(bool)].astype(int)

# for plotting only
x_plot,y_plot,z_plot = x,y,z

# define receiver coordinates
xR, yR, zR = X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)

# define source coordinates
xS, yS, zS = np.tile(sx,(X.size//sx.shape[0]+1, 1))[:X.size].reshape(-1,1), np.tile(sy,(Y.size//sy.shape[0]+1,1))[:Y.size].reshape(-1,1), np.tile(sz,(Z.size//sz.shape[0]+1,1))[:Z.size].reshape(-1,1)

# define all inputs and output
Xa = np.hstack((xS, yS, zS, xR, yR, zR))
ya = vpv.reshape(-1,1)

# permutate
perm_idx = torch.randperm(X.size).numpy()
Xa[:, :3] = Xa[perm_idx, :3]

# input for database
Xb = np.copy(Xa)
yb = np.copy(ya)

class to_torch(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        # Creating identical pairs
        self.data    = Variable(Tensor(data))
        self.target  = Variable(Tensor(target))

    def send_device(self,device):
        self.data    = self.data.to(device)
        self.target  = self.target.to(device)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y, index
    def __len__(self):
        return self.data.shape[0]

database = to_torch(Xb[np.logical_not(np.isnan(yb))[:,0]], yb[np.logical_not(np.isnan(yb))[:,0]])

if offline and old:
    nFeatures = 32
else:
    nFeatures = 6

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

class NN(torch.nn.Module):
    def __init__(self, nl=10, activation=act_fun):
            super(NN, self).__init__()
            self.act = activation()

            # Normalization Layer
            self.bn0 = torch.nn.BatchNorm1d(num_features=nFeatures, affine=False)

            # Input Structure
            self.fc0  = Linear(2*3,nFeatures)
            self.fc1  = Linear(nFeatures,512)

            # Resnet Block
            self.rn_fc1 = torch.nn.ModuleList([Linear(512, 512) for i in range(nl)])
            self.rn_fc2 = torch.nn.ModuleList([Linear(512, 512) for i in range(nl)])
            self.rn_fc3 = torch.nn.ModuleList([Linear(512, 512) for i in range(nl)])

            # Output structure
            self.fc8  = Linear(512,nFeatures)
            self.fc9  = Linear(nFeatures,1)

    def forward(self,x):
        if offline and old:
            x   = self.act(self.bn0(self.fc0(x)))
        else:
            x   = self.act(self.fc0(x))
        x   = self.act(self.fc1(x))
        for ii in range(len(self.rn_fc1)):
            x0 = x
            x  = self.act(self.rn_fc1[ii](x))
            x  = self.act(self.rn_fc3[ii](x)+self.rn_fc2[ii](x0))

        x     = self.act(self.fc8(x))
        tau   = abs(self.fc9(x))
        return tau

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

class Model():
    def __init__(self, ModelPath, device='cuda'):
        
        self.Params                                          = {}
        self.Params['ModelPath']                             = model_path
        self.Params['Device']                                = device
        self.Params['Pytorch Amp (bool)']                    = False
        self.Params['Network']                               = {}
        self.Params['Network']['Number of Residual Blocks']  = num_blo
        self.Params['Network']['Layer activation']           = act_fun
        self.Params['Training']                              = {}
        self.Params['Training']['Number of sample points']   = 1e4
        self.Params['Training']['Batch Size']                = bat_siz
        self.Params['Training']['Validation Percentage']     = 10
        self.Params['Training']['Number of Epochs']          = num_epo
        self.Params['Training']['Resampling Bounds']         = [0.1,0.9]
        self.Params['Training']['Print Every * Epoch']       = 1
        self.Params['Training']['Save Every * Epoch']        = 10
        self.Params['Training']['Learning Rate']             = lea_rat
        self.Params['Training']['Use Scheduler (bool)']      = True

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss   = []

    def EikonalLoss(self,Yobs,Xp,tau,device):
        dtau  = torch.autograd.grad(
            outputs=tau, 
            inputs=Xp, 
            grad_outputs=torch.ones(tau.size()).to(device), 
            only_inputs=True,
            create_graph=True,
            retain_graph=True
        )[0]
        T0    = torch.sqrt(((Xp[:,3]-Xp[:,0])**2 + (Xp[:,4]-Xp[:,1])**2 + (Xp[:,5]-Xp[:,2])**2)) 
        T1    = (T0**2)*(dtau[:,3]**2 + dtau[:,4]**2 + dtau[:,5]**2)
        T2    = 2*tau[:,0]*(dtau[:,3]*(Xp[:,3]-Xp[:,0]) + dtau[:,4]*(Xp[:,4]-Xp[:,1]) + dtau[:,5]*(Xp[:,5]-Xp[:,2]))
        T3    = tau[:,0]**2
        if bac_vel:
            S2 = (T1+T2+T3)/(bac_vel**2)
        else:
            S2 = (T1+T2+T3)
        if (S2==0).any():
            print("Whoops!")
        Ypred = torch.sqrt(1/S2)
        diff  = abs(Yobs[:,0]-Ypred)/Yobs[:,0]

        src_loc = torch.from_numpy(np.array((sx, sy, sz, sx, sy, sz)).reshape(-1,6)).to(torch.device(dev_typ)).float()
        tau_src = self.network(src_loc)
        vel_src = torch.from_numpy(1/yb[sids]).to(torch.device(dev_typ)).float()
        loss  = torch.mean(abs((Yobs[:,0]-Ypred)/Yobs[:,0])) + torch.mean(torch.abs(tau_src - vel_src)/vel_src)
        
        return loss, diff

    def init(self):
        self.network = NN(nl=self.Params['Network']['Number of Residual Blocks'],activation=self.Params['Network']['Layer activation'])
        self.network.apply(init_weights)
        self.network.float()
        self.network.to(torch.device(self.Params['Device']))

    def normalization(self,Xp=None,Yp=None):

        xmin_UTM = np.array(copy.copy([np.nanmin(Xb[:,3]),np.nanmin(Xb[:,4]),np.nanmin(Xb[:,5])]))
        xmax_UTM = np.array(copy.copy([np.nanmax(Xb[:,3]),np.nanmax(Xb[:,4]),np.nanmax(Xb[:,5])]))

        indx = np.argmax(xmax_UTM-xmin_UTM)
        self.nf_max    = xmax_UTM[indx]
        self.nf_min    = xmin_UTM[indx]
        self.sf        = (self.nf_max-self.nf_min)

        if (type(Xp)!=type(None)) and (type(Yp)==type(None)):
            Xp  = Xp/self.sf
            return Xp
        if (type(Xp)==type(None)) and (type(Yp)!=type(None)):
            Yp  = Yp*self.sf
            return Yp
        else:
            Xp = Xp/self.sf
            Yp = Yp/self.sf
            return Xp,Yp

    def train(self):

        # initialize horovod
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(123)

        # initialize the MLP
        self.init()

        # limit number of CPU threads to be used per worker.
        torch.set_num_threads(1)

        kwargs = {'num_workers': 1, 'pin_memory': True}

        # Defining the optimization scheme
        self.optimizer  = opt_fun(self.network.parameters(),lr=self.Params['Training']['Learning Rate'])
        if self.Params['Training']['Use Scheduler (bool)'] == True:
            self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        # Creating a sampling dataset
        self.dataset = database
        self.dataset.send_device(torch.device(self.Params['Device']))
        
        self.dataset.data,self.dataset.target = self.normalization(Xp=self.dataset.data,Yp=self.dataset.target)

        len_dataset         = len(self.dataset)
        n_batches           = int(len(self.dataset)/int(self.Params['Training']['Batch Size']) + 1)
        training_start_time = time.time()

        # Splitting the dataset into training and validation
        indices            = list(range(int(len_dataset)))
        validation_idx     = np.random.choice(
            indices, 
            size=int(len_dataset*(self.Params['Training']['Validation Percentage']/100)), 
            replace=False
        )
        train_idx          = list(set(indices) - set(validation_idx))
        validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_idx, num_replicas=hvd.size(), rank=hvd.rank())
        train_sampler      = torch.utils.data.distributed.DistributedSampler(train_idx, num_replicas=hvd.size(), rank=hvd.rank())

        train_loader       = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.Params['Training']['Batch Size'] ),
            sampler=train_sampler,
            )    
        validation_loader  = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.Params['Training']['Batch Size'] ),
            sampler=validation_sampler,
        )    

        # defining the initial weights to sample by
        weights = Tensor(torch.ones(len(self.dataset))).to(torch.device(self.Params['Device']))
        weights[validation_idx] = 0.0

        # horovod, broadcast parameters & optimizer state.
        hvd.broadcast_parameters(self.network.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        # redefine horovod optimizer
        self.optimizer = hvd.DistributedOptimizer(
            self.optimizer,
            named_parameters=self.network.named_parameters(),
            op=hvd.Adasum if args.use_adasum else hvd.Average,
            gradient_predivide_factor=args.gradient_predivide_factor
        )

        for epoch in range(1,self.Params['Training']['Number of Epochs']+1):
            print_every           = 1
            start_time            = time.time()
            running_sample_count  = 0
            total_train_loss      = 0
            total_val_loss        = 0

            # defining the weighting of the samples
            weights                 = torch.clamp(
                weights/weights.max(),
                self.Params['Training']['Resampling Bounds'][0],
                self.Params['Training']['Resampling Bounds'][1]
            )
            weights[validation_idx] = 0.0
            train_sampler_wei       = WeightedRandomSampler(weights, len(weights), replacement=True)
            train_loader_wei        = torch.utils.data.DataLoader(
                                        self.dataset,
                                        batch_size=int(self.Params['Training']['Batch Size'] ),
                                        sampler=train_sampler_wei,
                                      )
            weights                 = Tensor(torch.zeros(len(self.dataset))).to(torch.device(self.Params['Device']))

            for i, data in enumerate(train_loader_wei, 0):
                
                # get inputs/outputs and wrap in variable object
                inputs, labels, indexbatch = data
                inputs = inputs.float()
                labels = labels.float()

                inputs.requires_grad_()


                if self.Params['Pytorch Amp (bool)']:
                    with autocast():
                        outputs = self.network(inputs)
                        loss_value, wv  = self.EikonalLoss(labels,inputs,outputs,torch.device(self.Params['Device']))
                else:
                    outputs = self.network(inputs)
                    loss_value, wv  = self.EikonalLoss(labels,inputs,outputs,torch.device(self.Params['Device']))


                loss_value.backward()

                # update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()

                # updating the weights
                weights[indexbatch] = wv

                total_train_loss += loss_value.item()
                del inputs, labels, indexbatch, outputs, loss_value, wv


            # determining the Training Loss
            for i, data_val in enumerate(validation_loader, 0):
                inputs_val, labels_val, indexbatch_val = data_val
                inputs_val = inputs_val.float()
                labels_val = labels_val.float()
                inputs_val.requires_grad_()

                if self.Params['Pytorch Amp (bool)']:
                    with autocast():
                        outputs_val = self.network(inputs_val)
                        val_loss,wv = self.EikonalLoss(labels_val,inputs_val,outputs_val,torch.device(self.Params['Device']))
                else:
                    outputs_val  = self.network(inputs_val)
                    val_loss,wv  = self.EikonalLoss(labels_val,inputs_val,outputs_val,torch.device(self.Params['Device']))

                total_val_loss             += val_loss.item()
                del inputs_val, labels_val, indexbatch_val, outputs_val, val_loss, wv


            # creating a running loss for both training and validation data
            total_val_loss   /= len(validation_loader)
            total_train_loss /= len(train_loader)

            # applied average accross workers horovod
            test_val_loss = metric_average(test_val_loss, 'test_val_loss')
            total_train_loss = metric_average(total_train_loss, 'train_loss')

            self.total_train_loss.append(total_train_loss)
            self.total_val_loss.append(total_val_loss)

            if self.Params['Training']['Use Scheduler (bool)'] == True:
                self.scheduler.step(total_val_loss)

            del train_loader_wei,train_sampler_wei

            if epoch % self.Params['Training']['Print Every * Epoch'] == 0:
                with torch.no_grad():
                    print("Epoch = {} -- Training loss = {:.4e} -- Validation loss = {:.4e}".format(epoch,total_train_loss,total_val_loss))

            if (epoch % self.Params['Training']['Save Every * Epoch'] == 0) or (epoch == self.Params['Training']['Number of Epochs'] ) or (epoch == 1):
                with torch.no_grad():
                    self.save(epoch=epoch,val_loss=total_val_loss)

    def save(self,epoch='',val_loss=''):
        torch.save(
            {
                'epoch'                 : epoch,
                'model_state_dict'      : self.network.state_dict(),
                'optimizer_state_dict'  : self.optimizer.state_dict(),
                'train_loss'            : self.total_train_loss,
                'val_loss'              : self.total_val_loss
            }, 
            '{}/Model_Epoch_{}_ValLoss_{}.pt'.format(self.Params['ModelPath'],str(epoch).zfill(5),val_loss)
        )

    def load(self,filepath):
        
        # loading model information
        self.init()
        checkpoint            = torch.load(filepath,map_location=torch.device(self.Params['Device']))
        self.total_train_loss = checkpoint['train_loss']
        self.total_val_loss   = checkpoint['val_loss']
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.to(torch.device(self.Params['Device']))

    def traveltimes(self,Xt,projection=True,normalization=True):
        

        TT_res = torch.empty(Xb.shape[0], device=torch.device(self.Params['Device']))
        with torch.no_grad():
            
            for i, Xp in enumerate(Xt, 0):
                if i==0:
                    eva_bat=Xp.shape[0]

                # apply projection from LatLong to UTM
                Xp  = Xp.to(torch.device(self.Params['Device']))
                if projection:
                    Xp  = self.projection(Xp)
                if normalization:
                    Xp  = self.normalization(Xp=Xp,Yp=None)
                tau = self.network(Xp)
                T0  = torch.sqrt(((Xp[:,3]-Xp[:,0])**2 + (Xp[:,4]-Xp[:,1])**2 + (Xp[:,5]-Xp[:,2])**2))
                TT  = tau[:,0]*T0

                TT_res[i*eva_bat:(i+1)*eva_bat] = TT

                del i,Xp,tau,T0,TT
        return TT_res

    def velocity(self,Xt,projection=True,normalization=True):

        V_res = torch.empty(Xb.shape[0], device=torch.device('cpu'))
        for i, Xp in enumerate(Xt, 0):
            if i==0:
                eva_bat=Xp.shape[0]
            Xp    = Xp.to(torch.device(self.Params['Device']))
            if projection:
                Xp  = self.projection(Xp)
            if normalization:
                Xp  = self.normalization(Xp=Xp,Yp=None)
            Xp.requires_grad_()
            tau   = self.network(Xp)
            dtau  = torch.autograd.grad(
                outputs=tau, 
                inputs=Xp, 
                grad_outputs=torch.ones(tau.size()).to(torch.device(self.Params['Device'])),
                only_inputs=True,
                create_graph=True,
                retain_graph=True
            )[0] 
            T0    = torch.sqrt(((Xp[:,3]-Xp[:,0])**2 + (Xp[:,4]-Xp[:,1])**2 + (Xp[:,5]-Xp[:,2])**2))  
            T1    = (T0**2)*(dtau[:,3]**2 + dtau[:,4]**2 + dtau[:,5]**2)
            T2    = 2*tau[:,0]*(dtau[:,3]*(Xp[:,3]-Xp[:,0]) + dtau[:,4]*(Xp[:,4]-Xp[:,1]) + dtau[:,5]*(Xp[:,5]-Xp[:,2]))
            T3    = tau[:,0]**2
            Ypred = torch.sqrt(1/(T1+T2+T3)).detach()
            if normalization:
                Ypred = self.normalization(Yp=Ypred)

            V_res[i*eva_bat:(i+1)*eva_bat] = Ypred
            del Xp,tau,dtau,T0,T1,T2,T3,Ypred
            
        return V_res

# train
model = Model(model_path, device="cuda:0")
model.train()