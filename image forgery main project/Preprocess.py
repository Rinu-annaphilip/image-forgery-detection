
import pickle
import numpy as np
import os
# import itertools
# import matplotlib.pyplot as plt
import cv2
from imutils import paths
from PIL import Image, ImageChops, ImageEnhance
import random
import json
#converts input image to ela applied image
def convert_to_ela_image(path,quality):

    original_image = Image.open(path).convert('RGB')

    #resaving input image at the desired quality
    resaved_file_name = 'resaved_image.jpg'     #predefined filename for resaved image
    original_image.save(resaved_file_name,'JPEG',quality=quality)
    resaved_image = Image.open(resaved_file_name)

    #pixel difference between original and resaved image
    ela_image = ImageChops.difference(original_image,resaved_image)
    
    #scaling factors are calculated from pixel extremas
    extrema = ela_image.getextrema()
    max_difference = max([pix[1] for pix in extrema])
    if max_difference ==0:
        max_difference = 1
    scale = 350.0 / max_difference
    
    #enhancing elaimage to brighten the pixels
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    ela_image.save("ela_image.png")
    return ela_image



data=[]
labels=[]
print("[INFO] loading images...")
img_dir=sorted(list(paths.list_images("dataset2")))
random.shuffle(img_dir)
print(len(img_dir))
print("[INFO]  Preprocessing...")
count=0
maxlen=0
print("Total images==>",len(img_dir))
cnt=0
for i in img_dir:
    image_size = (128, 128)
    imgr= np.array(convert_to_ela_image(i, 90).resize(image_size)).flatten() / 255.0  
    data.append(imgr)
   
    lb=i.split(os.path.sep)[-2]
    if(lb=="Au"):
        labels.append(1)
    else:
        labels.append(0)
    cnt+=1
    print("Processed >",cnt)

# print(data)
# print(labels)
pickle.dump(data,open("datalist1.pkl","wb"))
pickle.dump(labels,open("labels1.pkl","wb"))