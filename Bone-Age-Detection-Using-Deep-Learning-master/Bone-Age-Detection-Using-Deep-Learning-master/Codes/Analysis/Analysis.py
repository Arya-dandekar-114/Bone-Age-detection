import tensorflow as tf #For Loading and Predicting using the saved model
from keras_preprocessing import image 
import numpy as np # linear algebra
import cv2 
from PIL import Image
from PIL import ImageFilter #For Noise and Sharpening Filter
import os
import pandas as pd #For DataFrame
import matplotlib.pyplot as plt #For plotting the results.

def main():
    # Model Initializing 
    new_model = tf.keras.models.load_model('/content/drive/MyDrive/BE_Project/Model/best_model.h5', compile=False)

    #Input Directory will differ
    inputdir = "/content/drive/MyDrive/BE_Project/Images/Images for Analysis/"
    test_list = [ f for f in  os.listdir(inputdir)]

    data = [] 
    for f in test_list:
        ca = ""
        avgClahe = list()
        img = inputdir + f

        #Averaging CLAHE Filter
        for i in range(-30, 30, 2):
            avgClahe.append(Predict(CLAHE(img,i), new_model)) 

        #Extract the Chronological age from the name of the Image file. All images are saved in "'Chronological.Age'YO.png" Format.    
        for i in range(len(f)):
            if f[i]=="Y":
              break
            else:
              ca = ca + f[i]      
        data.append([Predict(img, new_model), Predict(Sharpen(img, 1), new_model), Predict(CLAHE(img, 2), new_model), Predict(CLAHE(Sharpen(img,1), 2), new_model), sum(avgClahe)/len(avgClahe), Predict(DeNoise(img), new_model), Predict(Sharpen(DeNoise(img),1), new_model), Predict(Sharpen(DeNoise(img),2), new_model), float(ca)])    
        df = pd.DataFrame(data, columns=['Normal', 'Sharpening', 'Clahe', 'Clahe+Sharpening', 'Averageing Clahe', 'Noise Reduction', 'Noise Reduction + Sharpening', 'Noise Reduction + Sharpening(Twice)', 'Chronological'])
    df.to_csv('Analysis.csv')

    #Analysing Using a Bar Graph by plotting Mean Average Error (MAE) of different filters.
    l1 = ['Normal', 'Sharpening', 'Clahe', 'Clahe+Sharpening', 'Averageing Clahe', 'Noise Reduction', 'Noise Reduction + Sharpening', 'Noise Reduction + Sharpening(Twice)']
    l2 = list()
    for x in l1: 
        print(x)
        df['error'] = abs((df['Chronological'] - df[x]))

        #mean age is
        l2.append(df['error'].mean())
        df['error'] = 0
    l3 = ['Normal', 'Sharpening', 'CLAHE', 'Sharpened\n'+'CLAHE', 'Averaged\n'+'CLAHE', 'Noise\n'+'Reduced', 'Noise\n'+ 'Reduced &\n' + 'Sharpened', 'Noise\n'+ 'Reduced &\n' + 'Sharpened\n'+'(Level-2)']
    fig = plt.figure(figsize = (15, 10))

    # creating the bar plot
    plt.bar(l3, l2, 
		    width = 0.7, color = ['red'])
    print(l2)
    plt.ylabel("Mean Average Error")
    plt.title("Mean Average Error per each 100 Bones with different Filters")
    plt.show()

#For Prediction.
def Predict(img, new_model):
    img=tf.keras.preprocessing.image.load_img(img,target_size=(256,256))
    img=tf.keras.preprocessing.image.img_to_array(img)
    img=tf.keras.applications.xception.preprocess_input(img)
    return ((127.3207517246848 + 41.18202139939618*(new_model.predict(np.array([img])))[0][0])/12.0)

#CLAHE Filter. Courtesy :- Pratik Poojary. Modified Slightly.
def CLAHE(img, clip_limit):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = clip_limit/10, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    #ret, thresh3 = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cl1 = cv2.cvtColor(cl1 ,cv2.COLOR_GRAY2RGB)
    res = np.array(cl1)
    out = "/content/try.png"
    cv2.imwrite(out, res)
    return out

#Sharpening Filter
def Sharpen(img, degree):
    img =  Image.open(img)
    sharpened = img.filter(ImageFilter.SHARPEN)
    if degree == 2:
        sharpened = sharpened.filter(ImageFilter.SHARPEN)
    out = "/content/try.png"
    sharpened.save(out)    
    return out

#DeNoising Filter
def DeNoise(img):
    img =  Image.open(img)
    img = img.filter(ImageFilter.MinFilter)
    out = "/content/try.png"
    img.save(out)
    return out

if __name__ == "__main__":
    main()