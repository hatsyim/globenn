'''
losing parameters and rerun no more!
@hatsyim
'''

# chain the prediction results
from test import *

T_pred, V_pred = T_pred.detach().numpy(), V_pred.detach().numpy()

# numpy to hdf5
predictions = [
    T_pred,
    V_pred
    
]
name = [
    'T_pred',
    'V_pred'
]
print(np.min(T_pred) , np.min(T_pred)==0)
for i in range(len(predictions)):
    h5f = h5py.File(predictions_path + name[i]+'.h5', 'w')
    h5f.create_dataset(name[i], data=predictions[i])
    h5f.close()