from DeepConvSurv import DeepConvSurv
from Loader import PatchData
from NegativeLogLikelihood import NegativeLogLikelihood
from score import c_index, concordance_index

import os
import torch
import torch.nn as nn
from torchvision import transforms
from skorch import NeuralNetRegressor
from skorch.helper import predefined_split
from torch.optim import SGD
import resource
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

if device.type == 'cuda':
    print(torch.cuda.get_device_name(GPU_NUM))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM)/1024**3,1), 'GB')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

transfers = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.57806385, 0.57806385, 0.57806385],[0.00937135, 0.00937135, 0.00937135])
    ])
train0, train1, train2, train3, train4, train5, train6, train7, train8, train9 = PatchData.split_cluster('/home/smu1/test_Park/cluster_info.csv', 'Train', transfer=transfers)
valid0, valid1, valid2, valid3, valid4, valid5, valid6, valid7, valid9, valid9 = PatchData.split_cluster('/home/smu1/test_Park/cluster_info.csv', 'Valid', transfer=transfers)

train0df, valid0df = train0.split_train_valid()

model = DeepConvSurv()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0,1,6])
model.to(device)

net = NeuralNetRegressor(model, criterion=NegativeLogLikelihood,
                         lr = 0.0005, batch_size=512,
                         max_epochs=100, optimizer=SGD,
                         optimizer__momentum=0.9, optimizer__weight_decay=0.001,
                         iterator_train__shuffle = True, iterator_train__num_workers = 10,
                         iterator_valid__shuffle = True, iterator_valid__num_workers = 10,
                         train_split=predefined_split(valid0df),
                         device = 'cuda:0,1,6')

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

print("Fitting")
net.fit(train0df, y=None)
print("Fit completed")
history = net.history
train_loss0 = history[:, 'train_loss']
valid_loss0 = history[:, 'valid_loss']
ax1.plot(train_loss0)
ax1.plot(valid_loss0)
ax1.legend(['train_loss', 'valid_loss'])

net.save_params(f_params='dcs0_0005.pkl',
                f_optimizer='dcs0_0005_optimizer.pkl',
                f_history='dcs0_0005_history.json')

pred = net.predict_proba(valid0)
label = valid0.get_label()
accuracy = concordance_index(pred, label)
print(accuracy)

net1 = NeuralNetRegressor(model, criterion=NegativeLogLikelihood,
                          lr = 0.00001, batch_size=512,
                          max_epochs=100, optimizer=SGD,
                          optimizer__momentum=0.9, optimizer__weight_decay=0.001,
                          iterator_train__shuffle = True, iterator_train__num_workers = 10,
                          iterator_valid__shuffle = True, iterator_valid__num_workers = 10,
                          train_split=predefined_split(valid0df),
                          device = 'cuda:0,1,6')

print("Fitting")
net1.fit(train0df, y=None)
print("Fit completed")
history = net1.history
train_loss1 = history[:, 'train_loss']
valid_loss1 = history[:, 'valid_loss']
ax2.plot(train_loss1)
ax2.plot(valid_loss1)
ax2.legend('train_loss', 'valid_loss')

net1.save_params(f_params='dcs0_00001.pkl',
                 f_optimizer='dcs0_00001_optimizer.pkl',
                 f_history='dcs0_00001_history.json')

pred = net1.predict_proba(valid0)
label = valid0.get_label()
accuracy = concordance_index(pred, label)
print(accuracy)


fig.show()