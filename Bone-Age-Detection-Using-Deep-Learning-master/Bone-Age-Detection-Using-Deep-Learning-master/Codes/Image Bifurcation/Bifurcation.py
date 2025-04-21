import shutil
import pandas as pd 
import os


def main():

    #Entering the Image and CSV File Location.
    DATA_TRAIN = pd.read_csv(" ### Enter Your Path Here ### ")    #Example Path :- /content/gdrive/My Drive/Bone_Data/TrainData/datasets_10832_15122_boneage-training-dataset.csv 
    DATA_TEST = pd.read_csv(" ### Enter Your Path Here ### ")     #Example Path :- /content/gdrive/My Drive/Bone_Data/TrainData/datasets_10832_15122_boneage-test-dataset.csv
    
    ORIGINAL_IMAGE_PATH_TRAIN = " ### Enter Your Path Here ### "  #Example Path :- /content/gdrive/My Drive/Bone_Data/TrainData/PNGFormat 
    ORIGINAL_IMAGE_PATH_TEST = " ### Enter Your Path Here ### "   #Example Path :- /content/gdrive/My Drive/Bone_Data/TestData/PNGFormat


    #Creating the Directory for Storing the Bifurcated Images. NO NEED TO CHANGE ANYTHING FROM HERE.
    os.mkdir('/content/gdrive/My Drive/Bifurcated_Images')
    os.mkdir('/content/gdrive/My Drive/Bifurcated_Images/Test')
    os.mkdir('/content/gdrive/My Drive/Bifurcated_Images/Train')
    os.mkdir('/content/gdrive/My Drive/Bifurcated_Images/Test/Male')
    os.mkdir('/content/gdrive/My Drive/Bifurcated_Images/Train/Male')
    os.mkdir('/content/gdrive/My Drive/Bifurcated_Images/Test/Female')
    os.mkdir('/content/gdrive/My Drive/Bifurcated_Images/Train/Female')

    #Resultant Path for Storing Images.
    RESULT_IMAGE_PATH_TRAIN = '/content/gdrive/My Drive/Bifurcated_Images/Train/'
    RESULT_IMAGE_PATH_TEST = '/content/drive/My Drive/Bifurcated_Images/Test/'

    #Function written for Bifurcation
    Bifurcate(1377, 15611, DATA_TRAIN, ORIGINAL_IMAGE_PATH_TRAIN, RESULT_IMAGE_PATH_TRAIN)
    Bifurcate(4360, 4560, DATA_TEST, ORIGINAL_IMAGE_PATH_TEST, RESULT_IMAGE_PATH_TEST)

    print('Your File has been Saved to the Connected Drive /Bifurcated_Images')
   

   
#Function that Bifurcates Images named K.png all the way upto M.png, into Male and Female X-rays
def Bifurcate (k, m, DATA, Original_Image_Path, Result_Image_Path):
    csvcount = 0
    for x in range(k, m):
    
        #A Try-Catch Block used to make the algorithm work even if there is a missing image name. 
        try:
            Original_Path = Original_Image_Path + '/' + str(x) +'.png'

            if(DATA['male'][csvcount] == True):
                Result_Path = Result_Image_Path +'Male/'+ str(x) + '.png'
                shutil.copyfile(Original_Path, Result_Path)

            else:
                Result_Path = Result_Image_Path +'Female/'+ str(x) + '.png'
                shutil.copyfile(Original_Path, Result_Path)          
                                  
        except:
            continue

        csvcount = csvcount + 1
        Result_Path = ''


if __name__ == "__main__":
    main()