import os, glob
from PIL import Image, ImageEnhance, ImageChops
import numpy as np
import random
from PIL import ImageFilter
from PIL.ImageFilter import (
    BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
    EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
    )
import cv2 as cv
def salt_and_pepper(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col,_= X_imgs_copy.shape
    salt_vs_pepper = 0.3
    amount = 0.4
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
    
    random.seed(3)
    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_imgs.shape]
    X_imgs_copy[coords[0], coords[1], :] = 1
    random.seed(5)
    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_imgs.shape]
    X_imgs_copy[coords[0], coords[1], :] = 0
    return X_imgs_copy
    
files = sorted(glob.glob(os.path.join('..',"data","custom",'images','val','*')))
cnt = 0
path = "../data/custom/images/test/"
for img in files:
    image = Image.open(img)
    image_cv = cv.imread(img)
    #밝기
    enhancer = ImageEnhance.Brightness(image)
    brightness_image = enhancer.enhance(1.8)
    brightness_image.save(path+str(cnt)+'_bri.png')

    darkness_image = enhancer.enhance(0.5)
    darkness_image.save(path+str(cnt)+'_dark.png')

    #좌우 대칭
    horizonal_flip_image = image.transpose(Image.FLIP_LEFT_RIGHT) 
    horizonal_flip_image.save(path+str(cnt)+'_horizonal_flip.png')

    #상하 대칭
    vertical_flip_image = image.transpose(Image.FLIP_TOP_BOTTOM) 
    vertical_flip_image.save(path+str(cnt)+'_vertical_flip.png')

    # #좌우 이동
    # width, height = image.size
    # shift = random.randint(0, width * 0.2)
    # horizonal_shift_image = ImageChops.offset(image, shift, 0)
    # horizonal_shift_image.paste((0), (0, 0, shift, height))
    # horizonal_shift_image.save(path+str(cnt)+'_horizonal_shift.png')

    #상하 이동
    # width, height = image.size
    # shift = random.randint(0, height * 0.2)
    # vertical_shift_image = ImageChops.offset(image, 0, shift)
    # vertical_shift_image.paste((0), (0, 0, width, shift))
    # vertical_shift_image.save(path+str(cnt)+'_vertical_shift.png')

    #회전 
    rotate_image = image.rotate(random.randint(-30, 30))
    rotate_image.save(path+str(cnt)+'_rotate.png')

    #기울기
    #cx, cy = 0.1, 0
    #cx, cy = 0, 0.1
    cx, cy = 0, random.uniform(0.0, 0.3)
    shear_image = image.transform(
        image.size,
        method=Image.AFFINE,
        data=[1, cx, 0,
            cy, 1, 0,])
    shear_image.save(path+str(cnt)+'_shear.png')

    #확대 축소
    zoom = 0.7#random.uniform(0.7, 1.3) #0.7 ~ 1.3
    width, height = image.size
    x = width / 2
    y = height / 2
    crop_image = image.crop((x - (width / 2 / zoom), y - (height / 2 / zoom), x + (width / 2 / zoom), y + (height / 2 / zoom)))
    zoom_image = crop_image.resize((width, height), Image.LANCZOS)
    zoom_image.save(path+str(cnt)+'_zoom.png')
    

    #salt and pepper
    # print(image_cv.shape)
    # cv.imshow('asd',image_cv)
    sp = salt_and_pepper(image_cv)
    
    cv.imwrite(path+str(cnt)+"_salt_pepper.png",sp)
    
    #gaussianblur
    gb = ImageFilter.GaussianBlur(2)
    gbi = image.filter(gb)
    gbi.save(path+str(cnt)+'_gaussianblur.png')

    #image filters
    for f in (
    BLUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
    SMOOTH, SMOOTH_MORE, SHARPEN):
        
        image.filter(f).save(
            path+str(cnt)+"_{}.png".format(
                f.name.replace(" ", "-")))
    cnt+=1