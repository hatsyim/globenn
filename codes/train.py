'''
training using Keras
@hatsyim
'''

# chain previous files
from model import *

# device = torch.device("cuda")

model = Model(model_path,VelocityClass=VelocityFunction(),device="cuda:0")

# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   model = torch.nn.DataParallel(model)
#   model.to(device)

model.load('/home/taufikmh/KAUST/summer_2021/global_pinns/07_pytorch_implementation/models/bacvel_10_shallow/Model_Epoch_00160_ValLoss_0.003003701814964081.pt')

model.train()
