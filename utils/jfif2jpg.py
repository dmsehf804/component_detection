from PIL import Image
import os, glob

files = sorted(glob.glob(os.path.join('..',"data","dataset",'*')))
# print(folders)
path = "../data/dataset_jpg/"
i = 0
# print(files)
cnt = 0
for image in files:
    print(image)
    img = Image.open(image)
    img_resize = img.resize((512,512),Image.LANCZOS)
    img_resize.save(path+str(cnt)+'.png')
    cnt+=1
    

 