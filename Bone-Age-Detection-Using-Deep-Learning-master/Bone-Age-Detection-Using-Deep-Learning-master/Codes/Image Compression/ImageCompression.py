import cv2


def main():

    ORIGINAL_PATH = 'D:/Code_Me/Python/Bone_Age_Detection/TrainData/PNGFormat/'
    RESULT_IMAGE_PATH = 'D:/Code_Me/Python/Bone_Age_Detection/New Data Train/'
    counter = 1
    Quality = input("Input the Quality of JPEG Image from 0-99:")

    #Compresses Images named 1377.png all the way upto 15611.png to JPEG Format Using OpenCV
    for x in range(1377, 15611):

        Original_Image_Path = Path + str(x) + '.png'
        Result_Image_Path = ResultPath + counter + '.png'

        #A Try-Catch Block used to make the algorithm work even if there is a missing image name.   
        try:
            cv2.imwrite(Result_Image_Path, cv2.imread(Original_Image_Path), [int(cv2.IMWRITE_JPEG_QUALITY), Quality]) 
        except:
            continue
        print(x + " Images Done of 15611")

if __name__ == "__main__":
    main()