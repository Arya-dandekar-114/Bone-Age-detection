#!/usr/bin/env python

import cv2
import numpy as np
import shutil
import imutils
import pandas as pd 


def main():

    #For Image 
 
    ORIGINAL_IMAGE_PATH = 'D:/Code_Me/Python/Bone_Age_Detection/TrainData/PNGFormat/'
    RESULT_IMAGE_PATH = 'D:/Code_Me/Python/Bone_Age_Detection/Augmented data/Train Images/'
    counter = 1
    csvcount = 0

    #For Creating a Modified CSV File

    ORIGINAL_DATA = pd.read_csv("D:/Code_Me/Python/Bone_Age_Detection/TrainData/datasets_10832_15122_boneage-training-dataset.csv")
    df = pd.DataFrame(
        {
            "id":[ ],
            "boneage":[ ],
            "male":[ ]
        }
    )

    #Converting Datatype of "male" column to Boolean
    convert_dict = {'male': bool, 
                } 
    df = df.astype(convert_dict)

    #Augments Images named 1377.png all the way upto 15611.png, by rotating images by 10 degrees from -90 to 90   
    for x in range(1377, 15611):

        Path = ORIGINAL_IMAGE_PATH + str(x) + '.png'
        Result_Path = RESULT_IMAGE_PATH + str(counter) + '.png'

        #A Try-Catch Block used to make the algorithm work even if there is a missing image name. 
        try:
            shutil.copyfile(Path, Result_Path)  
            image = cv2.imread(Path)
            df.at[counter-1, 'id'] = counter
            df.at[counter-1, 'boneage'] = ORIGINAL_DATA['boneage'][csvcount]
            df.at[counter-1, 'male'] = ORIGINAL_DATA['male'][csvcount]
             
            for y in range(-90,91,10):
                
                if (y == 0):
                    continue
                    
                else:
                    counter = counter + 1
                    df.at[counter-1, 'id'] = counter
                    df.at[counter-1, 'boneage'] = ORIGINAL_DATA['boneage'][csvcount]
                    df.at[counter-1, 'male'] = ORIGINAL_DATA['male'][csvcount]
                    rotated = imutils.rotate_bound(image, y)
                    Temp_String_Name = RESULT_IMAGE_PATH +str(counter)+ ".png"
                    cv2.imwrite(Temp_String_Name, rotated)
                                  
        except:
            continue

        counter = counter + 1     
        csvcount = csvcount + 1
        print(str(x) + " of 15611 " + str(counter))
    
    #Saving the CSV file
    df.to_csv("D:/Code_Me/Python/Bone_Age_Detection/Augmented data/AugCSV.csv")


if __name__ == "__main__":
    main()