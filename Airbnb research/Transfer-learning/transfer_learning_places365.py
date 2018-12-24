# Source: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# coding: utf-8

# In[2]:


#imports torch version 0.3.1.post2 is used for this (torch.__version__)
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy


# In[4]:


# Data augmentation and normalization for training
# Just normalization for validation
# check the below url to know more about transforms in pytorch
# https://stackoverflow.com/questions/50002543/transforms-compose-meaning-pytorch
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# directory with images
data_dir = 'data'


# In[5]:


data_dir


# In[6]:


# A generic data loader  with dir path and function to transpose data
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
# print(image_datasets)
# batch_size (int, optional) – how many samples per batch to load (default: 1).
# shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
# num_workers (int, optional) – how many subprocesses to use for data loading. 
# 0 means that the data will be loaded in the main process. (default: 0)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# returns a list of classes 
class_names = image_datasets['train'].classes
# Context-manager that changes the selected device.
# device index to select. It’s a no-op if this argument is negative.
#device = torch.cuda.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.cuda.device(0)
print(class_names)


# Training the model
# ------------------
# 
# Now, let's write a general function to train a model. Here, we will
# illustrate:
# 
# -  Scheduling the learning rate
# -  Saving the best model
# 
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.
# 
# 

# In[16]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs=torch.squeeze(Variable(inputs))
                labels = Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
#                  requires_grad=True
#                 with torch.set_grad_enabled(phase == 'train'):
                if phase =='train' :
#                         torch.requires_grad=True       
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            outputs.data.zero_()
                            optimizer.step()

                # statistics
                running_loss += loss * inputs.size(0)
                running_corrects += torch.sum(preds.float().sum() == preds.float().sum())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {} Acc: {}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
#             import ipdb; ipdb.set_trace()
            if phase == 'val' and float(epoch_acc) > best_acc:
                best_acc = float(epoch_acc)
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Finetuning the convnet
# ----------------------
# 
# Load a pretrained model and reset final fully connected layer.
# 
# 
# 

# In[17]:


def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))
def load_model():
    # this model has a last conv feature map as 14x14

    model_file = 'whole_wideresnet18_places365.pth.tar'
    model_file = 'resnet18_places365.pth.tar'
    #if not os.access(model_file, os.W_OK):
    #    os.system('C:\\Users\\vivek4\\Downloads\\wget.exe   http://places2.csail.mit.edu/models_places365/' + model_file)
    #    os.system('C:\\Users\\vivek4\\Downloads\\wget.exe  https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')
    useGPU = 0
    if useGPU == 1:
        model = torch.load(model_file)
    else:
        model = torch.load(model_file, map_location=lambda storage, loc: storage) # allow cpu

    ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
    # from functools import partial
    # import pickle
    # pickle.load = partial(pickle.load, encoding="latin1")
    # pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    # model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
#     model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model


# In[18]:


features_blobs = []
model_ft=load_model()


# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
# 
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
# 
# 
# 

# In[21]:


model_conv = model_ft
for param in model_conv.parameters():
    param.requires_grad = False
#     param.volatile =True

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
# Applies a linear transformation to the incoming data: y=xAT+b
model_conv.fc = nn.Linear(num_ftrs, 14)

# model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
# 
# On CPU this will take about half the time compared to previous scenario.
# This is expected as gradients don't need to be computed for most of the
# network. However, forward does need to be computed.
# 
# 
# 

# In[20]:


model_conv = train_model(model_conv, criterion, optimizer_conv,exp_lr_scheduler, num_epochs=10)


# In[64]:


#print all the layers
model_conv


# In[65]:


#print all the weights
model_conv.state_dict()

