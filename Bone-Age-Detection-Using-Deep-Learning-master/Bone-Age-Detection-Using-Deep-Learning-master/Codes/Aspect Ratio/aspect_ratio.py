import cv2
#import os
import glob
import numpy as np
from tqdm import tqdm



height = []
width= []
#folder = '/content/gdrive/My Drive/BE Project/Bone Age Detection/boneage-training-dataset/boneage-training-dataset'

#for filename in tqdm(os.listdir(folder)):
    #img = cv2.imread(os.path.join(folder,filename),0)

for img in tqdm(glob.glob("E:\\Bone Age Detection\\boneage-training-dataset\\boneage-training-dataset\\*.png")):
    cv_img = cv2.imread(img,0)
    height.append(cv_img.shape[0])
    width.append(cv_img.shape[1])


print("Average of Height : ", np.mean(height)) 
print("Average of Width : ", np.mean(width)) 