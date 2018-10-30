
# coding: utf-8

# In[1]:


# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os ,sys
import numpy as np
from scipy.misc import imresize as imresize
import cv2
from PIL import Image
import pandas as pd


# In[2]:


def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(imresize(cam_img, size_upsample))
    return output_cam

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():
    # this model has a last conv feature map as 14x14

    model_file = 'whole_wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')
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

    model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model


# load the labels
classes, labels_IO, labels_attribute, W_attribute = load_labels()

# load the model
features_blobs = []
model = load_model()

# load the transformer
tf = returnTF() # image transformer

# get the softmax weight
params = list(model.parameters())
weight_softmax = params[-2].data.numpy()
weight_softmax[weight_softmax<0] = 0

#indoor outdoor classification 

path = "/Users/jaideep/Desktop/Scrapped images-1000/ScrappedImages-Jun-3-18/images"
dirs = os.listdir( path )
d=[];e=[];f=[];g=[];f1=[];g1=[];f2=[];g2=[];d1=[];f3=[];g3=[];f4=[];g4=[];error=[]
count=0;
for file in dirs:
    # load the test image
    #img_url = 'http://places2.csail.mit.edu/imgs/12.jpg'
    #os.system('wget %s -q -O test.jpg' % img_url)
    if not file.startswith('.') and file != 'Thumbs.db':
        try:
            img = Image.open(path+'/'+str(file))
            input_img = V(tf(img).unsqueeze(0), volatile=True)

            # forward pass
            logit = model.forward(input_img)
            h_x = F.softmax(logit).data.squeeze()
            probs, idx = h_x.sort(0, True)

            #print 'RESULT ON ' + img_url

            # output the IO prediction
            io_image = np.mean(labels_IO[idx[:10].numpy()]) # vote for the indoor or outdoor
            print io_image
            print str(file)
            if io_image < 0.5:
                count=count+1
                print count
                d.append(str(file).split(".")[0].split("_")[0])
                d1.append(str(file))
                e.append("indoor")
            else:
                count=count+1
                print count
                d.append(str(file).split(".")[0].split("_")[0])
                d1.append(str(file))
                e.append("outdoor")

            # output the prediction of scene category
        #     print '--SCENE CATEGORIES:'
        #     for i in range(0, 5):
        #         print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
            f.append(classes[idx[0]])
            g.append(probs[0])
            f1.append(classes[idx[1]])
            g1.append(probs[1])
            f2.append(classes[idx[2]])
            g2.append(probs[2])
            f3.append(classes[idx[3]])
            g3.append(probs[3])
            f4.append(classes[idx[4]])
            g4.append(probs[4])

            # output the scene attributes
            # responses_attribute = W_attribute.dot(features_blobs[1])
            # idx_a = np.argsort(responses_attribute)
            # print '--SCENE ATTRIBUTES:'
            # print ', '.join([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)])

            df_list = pd.DataFrame(
                {'id': d,
                 'list': e,
                 'file_name':d1,
                 'SCENE_CATEGORY_1':f,
                 'SCENE_CATEGORY_1_prob':g,
                    'SCENE_CATEGORY_2':f1,
                 'SCENE_CATEGORY_2_prob':g1,
                      'SCENE_CATEGORY_3':f2,
                 'SCENE_CATEGORY_3_prob':g2,
                     'SCENE_CATEGORY_4':f3,
                 'SCENE_CATEGORY_4_prob':g3,
                     'SCENE_CATEGORY_5':f4,
                 'SCENE_CATEGORY_5_prob':g4,
                })

            # # generate class activation mapping
            # print 'Class activation map is saved as cam.jpg'
            # CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

            # # render the CAM and output
            # img = cv2.imread('test.jpg')
            # height, width, _ = img.shape
            # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
            # result = heatmap * 0.4 + img * 0.5
            # cv2.imwrite('cam.jpg', result)
        except IOError:
            error.append(str(file))
            print(str(file)+' is not valid .')
                


# In[3]:


df_list
error


# In[4]:


df_list_sort=df_list.sort_values(['file_name'], ascending=[True])
df_list_sort


# In[6]:


df_list_sort.to_csv("/Users/jaideep/Desktop/Airbnb/Airbnb_Full_Project_Files_1/old/Airbnb_cnn_classification.csv", sep=',')

