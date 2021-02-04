#-*-coding:utf-8-*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob, os

folders = sorted(glob.glob(os.path.join("data","ocr",'test','*')))
# print(folders)
path = "data/ocr/result"
i = 0
for folder in folders:
    for image in sorted(glob.glob(os.path.join(folder,'*.jpg'))):
        # print("asdasd",image)
        bgr = cv2.imread(image)


        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)

        # contrast limit가 2이고 title의 size는 8X8
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)
        bgr1 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        fig,ax = plt.subplots(1,1,figsize=(10,10))
        fig.suptitle("Given Image cumulative distribution function")
        ax.hist(bgr.flatten(),256,[0,256],color='b')
        ax.hist(bgr1.flatten(),256,[0,256],color='r')
        ax.legend(['CDF','PDF'],loc='upper right')
        plt.savefig('input_histogram.png')
        plt.show()
        foldername = folder.split('/')[-1]
        filename = image.split('/')[-1]
        filename = filename[:-4]
        directory = path+foldername
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: create dir')
        print(directory+filename+'.jpg')
        cv2.imwrite(directory+'/'+filename+'.jpg',bgr1)
        i=i+1
        