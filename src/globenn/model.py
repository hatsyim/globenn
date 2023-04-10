'''
Modified from the original work of the Eikonet paper (https://arxiv.org/abs/2004.00361).
'''

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
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from pyproj import Proj
from scipy.ndimage.filters import gaussian_filter
from glob import glob
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
from mpl_toolkits.axes_grid1 import ImageGrid

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

# database = to_torch(Xb[np.logical_not(np.isnan(yb))[:,0]], yb[np.logical_not(np.isnan(yb))[:,0]])

def velocity_function(vel_sha='sphere', vel_typ='homogeneous', vel_ini=5, x=None, y=None, z=None, vp=None):
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
            vel_gap = vp
        elif vel_typ == 'constant_radial':
            vel_inh = 5*np.ones_like(vp)
            vel_all = 5*np.ones_like(vp)
            vel_gap = np.where(DEP<3000, vel_all, np.nan)
        elif vel_typ == 'gladm25':
            vel_inh = vp
            vel_all = vp
            vel_gap = vp   
            
    return vel_inh, vel_all, vel_gap


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

class NN(torch.nn.Module):
    def __init__(self, nl=10, activation=torch.nn.ELU):
            super(NN, self).__init__()
            self.act = activation()

            # Normalization Layer
            self.bn0 = torch.nn.BatchNorm1d(num_features=6, affine=False)

            # Input Structure
            self.fc0  = Linear(2*3,6)
            self.fc1  = Linear(6,512)

            # Resnet Block
            self.rn_fc1 = torch.nn.ModuleList([Linear(512, 512) for i in range(nl)])
            self.rn_fc2 = torch.nn.ModuleList([Linear(512, 512) for i in range(nl)])
            self.rn_fc3 = torch.nn.ModuleList([Linear(512, 512) for i in range(nl)])

            # Output structure
            self.fc8  = Linear(512,6)
            self.fc9  = Linear(6,1)

    def forward(self,x):
        x   = self.act(self.fc0(x))
        x   = self.act(self.fc1(x))
        for ii in range(len(self.rn_fc1)):
            x0 = x
            x  = self.act(self.rn_fc1[ii](x))
            x  = self.act(self.rn_fc3[ii](x)+self.rn_fc2[ii](x0))

        x     = self.act(self.fc8(x))
        tau   = abs(self.fc9(x))
        return tau

class Model():
    def __init__(self, params):

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss   = []
        self.params = params

    def eikonal_loss(self,Yobs,Xp,tau,device):
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
        vel_src = torch.from_numpy(1/np.array(yb[sids].mean()).reshape(-1)).to(torch.device(dev_typ)).float()
        # vel_src = torch.from_numpy(1/yb[sids]).to(torch.device(dev_typ)).float()
        loss  = torch.mean(abs((Yobs[:,0]-Ypred)/Yobs[:,0])) + torch.sum(torch.abs(tau_src - vel_src)/vel_src)
        
        return loss, diff

    def init(self):
        self.network = NN(nl=self.params['num_blocks'],activation=self.params['act_function'])
        self.network.apply(init_weights)
        self.network.float()
        self.network.to(torch.device(self.params['device']))

    def normalization(self,Xp=None,Yp=None):

        xmin_UTM = self.params['xmin']
        xmax_UTM = self.params['xmax']

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

        # Initialising the network
        self.init()

        # Defining the optimization scheme
        self.optimizer  = opt_fun(self.network.parameters(),lr=self.params['learning_rate'])
        if self.params['use_scheduler'] == True:
            self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        # Creating a sampling dataset
        self.dataset = database
        self.dataset.send_device(torch.device(self.params['device']))
        
        self.dataset.data,self.dataset.target = self.normalization(Xp=self.dataset.data,Yp=self.dataset.target)

        len_dataset         = len(self.dataset)
        n_batches           = int(len(self.dataset)/int(self.params['batch_size']) + 1)
        training_start_time = time.time()

        # Splitting the dataset into training and validation
        indices            = list(range(int(len_dataset)))
        validation_idx     = np.random.choice(
            indices, 
            size=int(len_dataset*(self.params['val_percentage']/100)), 
            replace=False
        )
        train_idx          = list(set(indices) - set(validation_idx))
        validation_sampler = SubsetRandomSampler(validation_idx)
        train_sampler      = SubsetRandomSampler(train_idx)

        train_loader       = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.params['batch_size'] ),
            sampler=train_sampler,
            )    
        validation_loader  = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.params['batch_size'] ),
            sampler=validation_sampler,
        )    

        # defining the initial weights to sample by
        weights = Tensor(torch.ones(len(self.dataset))).to(torch.device(self.params['device']))
        weights[validation_idx] = 0.0
        print(weights.device)

        for epoch in range(1,self.params['num_epochs']+1):
            print_every           = 1
            start_time            = time.time()
            running_sample_count  = 0
            total_train_loss      = 0
            total_val_loss        = 0

            # defining the weighting of the samples
            weights                 = torch.clamp(
                weights/weights.max(),
                self.params['sampling_bounds'][0],
                self.params['sampling_bounds'][1]
            )
            weights[validation_idx] = 0.0
            train_sampler_wei       = WeightedRandomSampler(weights, len(weights), replacement=True)
            train_loader_wei        = torch.utils.data.DataLoader(
                                        self.dataset,
                                        batch_size=int(self.params['batch_size'] ),
                                        sampler=train_sampler_wei,
                                      )
            weights                 = Tensor(torch.zeros(len(self.dataset))).to(torch.device(self.params['device']))

            for i, data in enumerate(train_loader_wei, 0):
                
                # Get inputs/outputs and wrap in variable object
                inputs, labels, indexbatch = data
                inputs = inputs.float()
                labels = labels.float()

                if epoch%10==0:
                    perm_idx = torch.randperm(inputs.shape[0])
                    inputs[:, :3] = inputs[perm_idx, :3]
                    # labels = labels[perm_idx]

                inputs.requires_grad_()


                if self.params['mixed_precision']:
                    with autocast():
                        outputs = self.network(inputs)
                        loss_value, wv  = self.eikonal_loss(labels,inputs,outputs,torch.device(self.params['device']))
                else:
                    outputs = self.network(inputs)
                    loss_value, wv  = self.eikonal_loss(labels,inputs,outputs,torch.device(self.params['device']))


                loss_value.backward()

                # Update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Updating the weights
                weights[indexbatch] = wv

                total_train_loss += loss_value.item()
                del inputs, labels, indexbatch, outputs, loss_value, wv


            # Determining the Training Loss
            for i, data_val in enumerate(validation_loader, 0):
                inputs_val, labels_val, indexbatch_val = data_val
                inputs_val = inputs_val.float()
                labels_val = labels_val.float()
                inputs_val.requires_grad_()

                if self.params['mixed_precision']:
                    with autocast():
                        outputs_val = self.network(inputs_val)
                        val_loss,wv = self.eikonal_loss(labels_val,inputs_val,outputs_val,torch.device(self.params['device']))
                else:
                    outputs_val  = self.network(inputs_val)
                    val_loss,wv  = self.eikonal_loss(labels_val,inputs_val,outputs_val,torch.device(self.params['device']))

                total_val_loss             += val_loss.item()
                del inputs_val, labels_val, indexbatch_val, outputs_val, val_loss, wv


            # Creating a running loss for both training and validation data
            total_val_loss   /= len(validation_loader)
            total_train_loss /= len(train_loader)
            self.total_train_loss.append(total_train_loss)
            self.total_val_loss.append(total_val_loss)

            if self.params['use_scheduler'] == True:
                self.scheduler.step(total_val_loss)

            del train_loader_wei,train_sampler_wei

            if epoch % self.params['log_frequency'] == 0:
                with torch.no_grad():
                    print("Epoch = {} -- Training loss = {:.4e} -- Validation loss = {:.4e}".format(epoch,total_train_loss,total_val_loss))

            if (epoch % self.params['save_frequency'] == 0) or (epoch == self.params['num_epochs'] ) or (epoch == 1):
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
            '{}/Model_Epoch_{}_ValLoss_{}.pt'.format(self.params['model_path'],str(epoch).zfill(5),val_loss)
        )

    def load(self,filepath):
        # Loading model information
        self.init()
        checkpoint            = torch.load(filepath,map_location=torch.device(self.params['device']))
        self.total_train_loss = checkpoint['train_loss']
        self.total_val_loss   = checkpoint['val_loss']
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.to(torch.device(self.params['device']))

    def traveltimes(self,Xt,projection=True,normalization=True):
        

        TT_res = torch.empty(Xt.dataset.shape[0], device=torch.device(self.params['device']))
        with torch.no_grad():
            
            for i, Xp in enumerate(Xt, 0):
                if i==0:
                    eva_bat=Xp.shape[0]
                # Apply projection from LatLong to UTM
                Xp  = Xp.to(torch.device(self.params['device']))
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

        V_res = torch.empty(Xt.dataset.shape[0], device=torch.device('cpu'))
        for i, Xp in enumerate(Xt, 0):
            if i==0:
                eva_bat=Xp.shape[0]
            Xp    = Xp.to(torch.device(self.params['device']))
            if projection:
                Xp  = self.projection(Xp)
            if normalization:
                Xp  = self.normalization(Xp=Xp,Yp=None)
            Xp.requires_grad_()
            tau   = self.network(Xp)
            dtau  = torch.autograd.grad(
                outputs=tau, 
                inputs=Xp, 
                grad_outputs=torch.ones(tau.size()).to(torch.device(self.params['device'])),
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
