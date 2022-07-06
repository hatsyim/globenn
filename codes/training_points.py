'''
neat way of train-test-val splitting
@hatsyim
'''

# model parameters
from velocity_function import *

# define receiver coordinates
xR, yR, zR = X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)

# define source coordinates
if single_source:
    xS, yS, zS = sx[0]*np.ones_like(X.reshape(-1,1)), sy[0]*np.ones_like(X.reshape(-1,1)), sz[0]*np.ones_like(X.reshape(-1,1))
else:
    xS, yS, zS = np.tile(sx,(X.size//sx.shape[0]+1, 1))[:X.size].reshape(-1,1), np.tile(sy,(Y.size//sy.shape[0]+1,1))[:Y.size].reshape(-1,1), np.tile(sz,(Z.size//sz.shape[0]+1,1))[:Z.size].reshape(-1,1)

# define inputs and output
Xp = np.hstack((xS, yS, zS, xR, yR, zR))
yp = vel_gap.reshape(-1,1)

# input for database
Xb = np.copy(Xp)
yb = np.copy(yp)

if permutate:
    perm_idx = torch.randperm(X.size).numpy()
    Xb[:, :3] = Xb[perm_idx, :3]

# random sampling
X_train, X_test, y_train, y_test = train_test_split(
    Xp[np.logical_not(np.isnan(yp))[:,0]], 
    yp[np.logical_not(np.isnan(yp))].reshape(-1,1), 
    test_size=0.1,
    random_state=1335
)

# remove source information
if try_first:
    X_starf = [X_train[:,3].reshape(-1,1), X_train[:,4].reshape(-1,1), X_train[:,5].reshape(-1,1)]
    sids,_ = np.where((np.isclose(X_starf[0], sx)) & (np.isclose(X_starf[1], sy)) & (np.isclose(X_starf[2], sz)))
    print("before")
    print(X_train.shape)
    print(sids)
    print(X_starf[0][sids,0])
    print(X_starf[1][sids,0])
    print(X_starf[2][sids,0])
    print(sx,sy,sz)
    X_tmp, y_tmp = np.zeros_like(X_train[:-1,:]), np.zeros_like(y_train[:-1,:])
    for i in range(6):
        X_tmp[:,i] = np.delete(X_train[:,i], sids[0])
    y_tmp[:,0] = np.delete(y_train[:,0], sids[0])
    X_train, y_train = X_tmp, y_tmp
    print(X_train.shape)

# find source location id in X_star
X_starf = [X_train[:,3].reshape(-1,1), X_train[:,4].reshape(-1,1), X_train[:,5].reshape(-1,1)]

# sids,_ = np.where((np.isclose(X_starf[0], sx)) & (np.isclose(X_starf[1], sy)) & (np.isclose(X_starf[2], sz)))

sids = [np.where((np.isclose(X_starf[0], sx[i], atol=1e-16)) & (np.isclose(X_starf[1], sy[i], atol=1e-16)) & (np.isclose(X_starf[2], sz[i], atol=1e-16))) for i in range(len(sx))]
sids = np.asarray(sids)[:,0]

print(sids.shape)

sids = sids[sids.astype(bool)].astype(int)

print("after")
print(sids)
print(X_starf[0][sids,0])
print(X_starf[1][sids,0])
print(X_starf[2][sids,0])
print(sx,sy,sz)
