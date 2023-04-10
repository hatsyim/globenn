"""
database for PyTorch input
"""

from torch.autograd import Variable, grad
from torch import Tensor

def database(PATH,VelocityFunction,create=False,Numsamples=5000,randomDist=False,SurfaceRecievers=False):
    if create == True:
        xmin = copy.copy(VelocityFunction.xmin)
        xmax = copy.copy(VelocityFunction.xmax)

        # Projecting from LatLong to UTM
        if type(VelocityFunction.projection) == str:
            proj = Proj(VelocityFunction.projection)
            xmin[0],xmin[1] = proj(xmin[0],xmin[1])
            xmax[0],xmax[1] = proj(xmax[0],xmax[1])

        Xp   = _randPoints(numsamples=Numsamples,Xmin=xmin,Xmax=xmax,randomDist=randomDist)
        Yp   = VelocityFunction.eval(Xp)

        # Handling NaNs values
        while len(np.where(np.isnan(Yp[:,1]))[0]) > 0:
            indx     = np.where(np.isnan(Yp[:,1]))[0]
            print('Recomputing for {} points with nans'.format(len(indx)))
            Xpi      = _randPoints(numsamples=len(indx),Xmin=xmin,Xmax=xmax,randomDist=randomDist)
            Yp[indx,:] = VelocityFunction.eval(Xpi)
            Xp[indx,:] = Xpi

        # Saving the training dataset
        np.save('{}/Xp'.format(PATH),Xp)
        np.save('{}/Yp'.format(PATH),Yp)
    else:
        try:
            Xp = Xb
            print(np.min(np.sqrt(((Xb[:,3]-Xb[:,0])**2 + (Xb[:,4]-Xb[:,1])**2 + (Xb[:,5]-Xb[:,2])**2)) ))
            Yp = VelocityFunction.eval(Xb)
        except ValueError:
            print('Please specify a correct source path, or create a dataset')

    database = _numpy2dataset(Xp[np.logical_not(np.isnan(Yp))[:,0]],Yp[np.logical_not(np.isnan(Yp))[:,0]])

    return database

class _numpy2dataset(torch.utils.data.Dataset):
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

def _randPoints(numsamples=10000,randomDist=False,Xmin=[0,0,0],Xmax=[2,2,2]):
    numsamples = int(numsamples)
    Xmin = np.append(Xmin,Xmin)
    Xmax = np.append(Xmax,Xmax)
    if randomDist:
        X  = np.zeros((numsamples,6))
        PointsOutside = np.arange(numsamples)
        while len(PointsOutside) > 0:
            P  = np.random.rand(len(PointsOutside),3)*(Xmax[:3]-Xmin[:3])[None,None,:] + Xmin[:3][None,None,:]
            dP = np.random.rand(len(PointsOutside),3)-0.5
            rL = (np.random.rand(len(PointsOutside),1))*np.sqrt(np.sum((Xmax-Xmin)**2))
            nP = P + (dP/np.sqrt(np.sum(dP**2,axis=1))[:,np.newaxis])*rL

            X[PointsOutside,:3] = P
            X[PointsOutside,3:] = nP

            maxs          = np.any((X[:,3:] > Xmax[:3][None,:]),axis=1)
            mins          = np.any((X[:,3:] < Xmin[:3][None,:]),axis=1)
            OutOfDomain   = np.any(np.concatenate((maxs[:,None],mins[:,None]),axis=1),axis=1)
            PointsOutside = np.where(OutOfDomain)[0]
    else:
        X  = (np.random.rand(numsamples,6)*(Xmax-Xmin)[None,None,:] + Xmin[None,None,:])[0,:,:]
    return X

class VelocityFunction:
    def __init__(self):
        self.xmin     = [np.nanmin(Xp[:,3]),np.nanmin(Xp[:,4]),np.nanmin(Xp[:,5])]
        self.xmax     = [np.nanmax(Xp[:,3]),np.nanmax(Xp[:,4]),np.nanmax(Xp[:,5])]

        # projection 
        self.projection = None

        # Velocity values
        self.velocity_mean     = np.nanmean(yp)
        self.velocity_phase    = 0.5
        self.velocity_offset   = -2.5 
        self.velcoity_amp      = 1.0

    def eval(self,Xp):
        Yb = np.ones((Xp.shape[0],2))
        Yb[:,0] = yb.reshape(Xp.shape[0],)
        Yb[:,1] = yb.reshape(Xp.shape[0],)

        return Yb