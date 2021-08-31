#Imports
import os
import sys
import numpy as np
import shutil
from shutil import copyfile
from keras.preprocessing.image import save_img
import cv2


# #Please make sure all the folders names are correct acording to your directory structure and those folders are there at the locations

# # #Saving the Names of File in a dictionary

d={}
for filename in os.listdir('SmallSet/0/'):
  #desc=" ".join(filename.split(".png")[0].split("_")[1:])
  desc = filename
  #print(desc)
  id=int(filename.split("_")[0])
  if(id in list(d.keys())):
    d[id].append(desc)
  else:
    d[id]=list([desc])
#print(desc)


# # Creating folders and saving all the mask file for each image into individual folders

for i in d:
    os.mkdir("SmallSet/masks/"+str(i))
    for j in range(len(d[i])):
        #print(d[i][j])
        copyfile("SmallSet/0/"+d[i][j],"SmallSet/masks/"+str(i)+"/"+d[i][j])


# # Remove all Skin Masks


for i in range(0,1999):
    for filename in os.listdir('SmallSet/masks/'+str(i)+'/'):
        desc=" ".join(filename.split(".png")[0].split("_")[1:])
        if desc == 'skin':
            #print(filename)
            #print(i)
            os.remove('SmallSet/masks/'+str(i)+'/'+filename)


# # Clubbing all the masks and making a mask imag for all the folders:

a = []
for folders in os.listdir('SmallSet/masks/'):
    #print(folders)
    mask= np.zeros((512, 512,3))
    for filename in os.listdir('SmallSet/masks/'+folders+'/'):
        mask_image = cv2.imread('SmallSet/masks/'+folders+'/'+filename, -1)
        #print(type(mask_image))
        mask = np.maximum(mask, mask_image)
    save_img('SmallSet/masks/'+folders+'/'+folders+".png", mask)
    #imshow(mask)
    #plt.show()

# # Copy only the complete mask image from masks images and paste it in a new folder as ground truths

for folders in os.listdir('SmallSet/masks/'):
    for filename in os.listdir('SmallSet/masks/'+folders+'/'):
        d = filename.split(".png")[0]
        if d == folders:
            copyfile("SmallSet/masks/"+folders+'/'+filename,"SmallSet/images/GroundTruth/"+filename)
            #print(d)


# # Copy Files from GroundTruth to label

src_files = os.listdir('SmallSet/images/GroundTruth')
for file_name in src_files:
    full_file_name = os.path.join('SmallSet/images/GroundTruth', file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, 'SmallSet/images/train/label')  


# # Deleting all the disposable files and folders

for folders in os.listdir('SmallSet/masks/'):
    shutil.rmtree('SmallSet/masks/'+folders)

for file_name in os.listdir('SmallSet/images/GroundTruth'):
    os.remove('SmallSet/images/GroundTruth/'+file_name)



