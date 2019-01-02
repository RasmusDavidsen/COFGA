# # COFGA 
# 
# ### Created by
# #### Rasmus Davidsen & Lukkas Hamann

import numpy as np
import pandas as pd
from skimage.io import imread
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torchvision import transforms
import time
import csv

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import WeightedRandomSampler


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# custom libraries
from mAP import mAP_score
from COFGA_dataset import CofgaDataset


# ## Define run

# define name of output files
results_name = "results_resnet152_dropOut_wD_bN_noColor_map.csv"
results_name_AP_val = "Results_resnet152_dropOut_wD_bN_noColor_AP_val.csv"
results_name_AP_train = "Results_resnet152_dropOut_wD_bN_noColor_AP_train.csv"


# ## Data augmentation

# random rotation
degrees = 5
rotation_transform = transforms.Compose([transforms.RandomRotation(degrees), transforms.ToTensor(),])

# perfoming horizontal flip with a given probability
prob = 0.8
hoz_transform = transforms.Compose([transforms.RandomHorizontalFlip(prob), transforms.ToTensor(),])

#performing vertical flip with a gven probability
prob = 0.8
vert_transform = transforms.Compose([transforms.RandomVerticalFlip(prob), transforms.ToTensor(),])


# allows to chose randomly from the different transformations
transform_list = transforms.RandomChoice([rotation_transform, hoz_transform,
                                          vert_transform])


# ## Loading the data

# loading the custom dataset
dataset = CofgaDataset(csv_file='dataset/train_preprocessed.csv',
                             root_dir='dataset/root/train/resized/',
                            transform = transform_list)


print("Total number of images: ",len(dataset))

COFGA_headers = pd.read_csv('dataset/train_preprocessed.csv')

COFGA_labels = COFGA_headers.columns.tolist()
COFGA_labels.pop(0)

COFGA_labels.insert(0, "epoch")


# ## Constructing trainLoader and validation loader
batch_size = 32

# fraction of dataset to be validation set
validation_split = 0.3

# shuffle dataset when taking the validation set
shuffle_dataset = True

# seed
random_seed = 1


dataset_size = len(dataset)
indices = list(range(dataset_size))

split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    

train_indices, val_indices = indices[split:], indices[:split]

#train_indices, val_indices = indices[:4], indices[4:6]

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)





train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=False,
                                           num_workers=1, sampler=train_sampler, pin_memory=True)


validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=False,
                                                num_workers=1, sampler=validation_sampler, pin_memory=True)


print("Number of train images: ", len(train_sampler))
print("Number of validation images: ", len(validation_sampler))
print("\nDataloader completed")


# ## Building the model

# ### Including cuda for GPU
use_cuda = torch.cuda.is_available()

def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()


# ### Defining the network

NUM_CLASSES = 37

class COFGA_NET(nn.Module):
    def __init__(self):
        super(COFGA_NET, self).__init__()
    
        # defining the pretrained network (pretrained on ImageNet data set) 
        self.model = models.resnet152(pretrained = True)
        num_final_in = self.model.fc.in_features
        
        # changin the last FC layer to fit the outputs for this data
        self.model.fc = nn.Linear(num_final_in, 1024)
        
        # Freezing all layers in resnet
        for param in self.model.parameters():
            param.requires_grad = True
        
        
        self.bn_1 = nn.BatchNorm1d(1024)
        
        self.relu1 = nn.ReLU(inplace=False)
        
        self.drop = nn.Dropout(p=0.2, inplace=False)
        
        self.dense_1 = nn.Linear(1024, 500)
        
        self.l_out = nn.Linear(500, NUM_CLASSES)   
        
       
        
    # The forward function defines the flow of the input data and thus decides which layer/chunk goes on top of what.
    def forward(self,x):
        
        x = self.model(x)
        
        x = torch.squeeze(x)
        
        x = self.bn_1(x)
        
        x = self.relu1(x)
        
        x = self.drop(x)
        
        x = self.dense_1(x)
        
        x = self.relu1(x)
        
        x = self.l_out(x)

        x = torch.sigmoid(x)
        return x
net = COFGA_NET()
if use_cuda:
    print("USING CUDA!")
    net.cuda()
    

#print(net)


print("Network constructed")


# ### Defining the loss function and the optimizer

import torch.optim as optim

# adddin the Binary cross entropy loss function
criterion = nn.BCELoss(reduction='sum')

# defining the optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)


# ## Train the network
from sklearn.metrics import accuracy_score

# number of epochs to train
num_epoch = 300

# train list for MAP_score 
train_MAP_list = [] # Total
train_AP_list = [] # Pr. class

# validation list for MAP_score 
val_MAP_list = [] # Total
val_AP_list = [] # Pr. class

# loss list
train_loss_list = []
val_loss_list = []

# list for epochs
epoch_list = []

#training the network
for epoch in range(num_epoch):  # loop over the dataset multiple times
    train_loss = 0.0
    val_loss = 0.0
    
    # allocating memory for train
    train_predicted = np.zeros(NUM_CLASSES)
    train_target = np.zeros(NUM_CLASSES)
    
    net.train()
    for i, data in enumerate(train_loader,0):
        
        # get inputs (by names in the custom dataset class)
        inputs = data['image']
        labels = data['labels']
        
        # wrap them in Variable
        inputs, labels = get_variable(Variable(inputs.type(torch.FloatTensor))), get_variable(Variable(labels.type(torch.FloatTensor)))
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # computing the outputs
        outputs = net(inputs)
        
        
        # outputs to numpy
        train_predicted = np.vstack((train_predicted,get_numpy(outputs)))
        
        
        # labels to numpy
        train_target = np.vstack((train_target,get_numpy(labels)))

        # computing the loss function
        loss = criterion(outputs, labels)

        # backpropagating
        loss.backward()

        # updating weights and biases in the network
        optimizer.step()
        
        # appending train loss (summed over the minibatch )
        train_loss += get_numpy(loss.data[0])
    
    
    # saving loss, MAP and AP for every 10 epoch  
    if int(epoch+1) % int(1) == 0 or epoch==num_epoch or epoch == 0:
        
        # removing first zero dummy row
        train_predicted = train_predicted[1:,:]
        train_target = train_target[1:,:]
        
        
        # evaluating network on validation set
        net.eval()
        # appending epoch to list for result outputs
        epoch_list.append(epoch+1)
        
        # appending train loss to list
        train_loss = train_loss/len(train_sampler)
        train_loss_list.append(train_loss)
        
        
        # allocating memory for validation MAP
        #val_predicted = np.zeros(( int(len(validation_sampler)), NUM_CLASSES))
        #val_target = np.zeros(( int(len(validation_sampler)), NUM_CLASSES))
        
        val_predicted = np.zeros(NUM_CLASSES)
        val_target = np.zeros(NUM_CLASSES)
        
        for i, data in enumerate(validation_loader,0):
            # get inputs (by names in the custom dataset class)
            inputs = data['image']
            labels = data['labels']

            # wrap them in Variable
            inputs, labels = get_variable(Variable(inputs.type(torch.FloatTensor))), get_variable(Variable(labels.type(torch.FloatTensor)))

            outputs = net(inputs)
            
            # computing the loss function
            loss = criterion(outputs, labels)
            
            val_loss += get_numpy(loss.data[0])
            
             
             # outputs to numpy
            val_predicted = np.vstack((val_predicted,get_numpy(outputs)))
        
        
            # labels to numpy
            val_target = np.vstack((val_target,get_numpy(labels)))
            
            
            # outputs to numpy
            outputs = get_numpy(outputs)

        # removing first zero dummy row
        val_predicted = val_predicted[1:,:]
        val_target = val_target[1:,:]
        
        # appending validation loss to list
        val_loss = val_loss/len(validation_sampler)                       
                               
        val_loss_list.append(val_loss)
        
        # calculating MAP
        val_MAP = mAP_score(val_predicted, val_target)
        train_MAP = mAP_score(train_predicted, train_target)

        # appending MAP score to MAP list
        val_MAP_list.append(val_MAP[0])
        train_MAP_list.append(train_MAP[0])
        
        # appending AP score to AP list for validation
        temp_val = val_MAP[1].tolist()
        temp_val.insert(0, epoch+1)
           
        val_AP_list.append(temp_val)
        
        # appending AP score to AP list for train
        temp_train = train_MAP[1].tolist()
        temp_train.insert(0, epoch+1)
           
        train_AP_list.append(temp_train)
        
        # printing loss train, validation loss, MAP
        print("[%d], Train loss: %.3f, Val loss: %.3f, Train mAP: %.3f"%
              (epoch+1, train_loss, val_loss, train_MAP[0]*100), "%,","Val mAP: %.3f"%(val_MAP[0]*100), "%")
    
    
print('\nFinished Training \n')

# saving results as pandas dataframe
# print('Results validation set\n')
results = pd.DataFrame({'Epoch':epoch_list,'Train loss': train_loss_list, 'Val loss': val_loss_list, 'Train_mAP': train_MAP_list, 'Val_mAP': val_MAP_list})
results.to_csv(results_name, sep=";", index=False)

# Saving AP for validation
with open(results_name_AP_val, "w") as val_AP_tocsv:
    writer = csv.writer(val_AP_tocsv, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    writer.writerow(COFGA_labels)
    for item in val_AP_list:
        writer.writerow(item[i] for i in range(0,len(COFGA_labels)))
        
# Saving AP for train
with open(results_name_AP_train, "w") as train_AP_tocsv:
    writer = csv.writer(train_AP_tocsv, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    writer.writerow(COFGA_labels)
    for item in train_AP_list:
        writer.writerow(item[i] for i in range(0,len(COFGA_labels)))


