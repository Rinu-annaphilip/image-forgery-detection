import numpy as np

from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model
import cv2
model = load_model("cnn_model.h5")  
class_names = ["Forged", "Authentic"] 
print("model loaded")
def convert_to_ela_image(path,quality):
    
    original_image = Image.open(path).convert('RGB')
    print("original image")
    #resaving input image at the desired quality
    resaved_file_name = 'resaved_image.jpg'     #predefined filename for resaved image
    original_image.save(resaved_file_name,'JPEG',quality=quality)
    resaved_image = Image.open(resaved_file_name)
    print("resaved_image image") 
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




def predict_result(fname):
    image_size = (128, 128)
     # classification outputs
    test_image =  np.array(convert_to_ela_image(fname, 90).resize(image_size)).flatten()/255.0

    test_image = test_image.reshape(-1, 128, 128, 3)

    y_pred = model.predict(test_image)
    print(y_pred)
    # y_pred_class = round(y_pred[0][0])
    if(y_pred[0][0]==1):

        prediction = "Authentic"
    else:
        prediction = "Forged"
        # prediction = class_names[y_pred_class]
   
    return prediction

# impath="dataset2/Tp/Tp_D_CNN_M_B_nat10139_nat00059_11949.jpg"

# prdval=predict_result(impath)

# print(prdval)