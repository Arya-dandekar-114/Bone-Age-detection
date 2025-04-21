#!/usr/bin/env python
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle


def main():
  DATASET_PATH = 'D:/Code_Me/Python/Bone_Age_Detection/TrainData/datasets_10832_15122_boneage-training-dataset.csv'
  TRAIN_DIRECTORY = 'D:/Code_Me/Python/Bone_Age_Detection/TrainData/PNGFormat'
  data_processing(DATASET_PATH, TRAIN_DIRECTORY)


def clahe_histogram(img):
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  cl1 = clahe.apply(img)
  res = np.array(cl1)
  return res

def data_processing(DSP, IDP):
  training_data = []
  count = 0
  Data_Train = pd.read_csv(DSP)
  
  for img in tqdm(os.listdir(IDP)[10:20]):
    #For Labels
    label_age = Data_Train["boneage"][count]
    if(Data_Train["male"][count] == True):
      label_gender = 1
    else:
      label_gender = 0
    path = os.path.join(IDP,img)
    
    img = cv2.imread(path,0)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_array = clahe_histogram(img)

    training_data.append([img_array, label_age, label_gender])
    
    count = count + 1
    #print(count) 

  shuffle(training_data)
  np.save('D:/Code_Me/Python/Bone_Age_Detection/train_data1.npy', training_data) 
  #print(training_data) 
  print("File Created")

if __name__ == "__main__":
    main()