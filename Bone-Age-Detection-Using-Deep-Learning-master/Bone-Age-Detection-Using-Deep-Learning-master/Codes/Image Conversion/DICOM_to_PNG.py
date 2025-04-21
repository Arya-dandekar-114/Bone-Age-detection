import cv2
import os
import pydicom #Convert DICOM images to PNG.

#Directory of the Images.
inputdir = 'C:/New folder/'

#Output Directory for PNG images
outdir = 'D:/Code_Me/Python/Bone_Age_Detection'
#os.mkdir(outdir)

test_list = [ f for f in  os.listdir(inputdir)]

for f in test_list: 
    # read dicom image
    ds = pydicom.read_file(inputdir + f) 

     # get image array
    img = ds.PixelData
    cv2.imwrite(f.replace('.DCM','.png'),img)