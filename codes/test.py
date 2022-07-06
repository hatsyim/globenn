"""
plotting
@hatsyim
"""

from model import *

if offline:
    model_path = "/home/taufikmh/KAUST/summer_2021/global_pinns/10_incorporating_us_array/models/2001_21_1e-06_331_91_181_ELU"
    # model_path = "/home/taufikmh/KAUST/summer_2021/global_pinns/07_pytorch_implementation/models/2001_20_1e-06_83_91_181_<class 'torch.nn.modules.activation.ELU'>_False_1367093_256_sphere_gladm25_512_cartesian_1_<class 'torch.optim.adam.Adam'>_MinMax_6_cuda"
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

print("Loaded")

if offline and old:
    Xt = torch.utils.data.DataLoader(
    torch.from_numpy(Xb).to(torch.float).to(torch.device('cuda')),
    batch_size=int(Xb.shape[0])
    )
    T_pred = model.traveltimes(Xt, projection=False, normalization=False).cpu().reshape(vel_all.shape)
    V_pred = model.velocity(Xt, projection=False, normalization=False).cpu().reshape(vel_all.shape)
else:
    Xt = torch.utils.data.DataLoader(
    torch.from_numpy(Xb).to(torch.float).to(torch.device('cuda')),
    batch_size=int(Xb.shape[0]//10)
    )
    T_pred = model.traveltimes(Xt, projection=False, normalization=True).cpu().reshape(vel_all.shape)
    V_pred = model.velocity(Xt, projection=False, normalization=True).cpu().reshape(vel_all.shape)

print("Predicted")