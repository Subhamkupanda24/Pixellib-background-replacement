#Task 1

#Fastest way to load all the images at once with respect to their extension
import cv2
import os

for root, subdirs, files in os.walk('C:\\Users\\USER\\Desktop\\Carscan'):
    for f in files:
        if f.endswith('jpg'):
            # print(f)
            img = cv2.imread('C:\\Users\\USER\\Desktop\\Carscan\\' + f)


#Task 2
#Background removal

from pixellib.tune_bg import alter_bg

change_bg = alter_bg()
change_bg.load_pascalvoc_model("C:/Users/USER/Desktop/AI_Assignments/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

#For 1st image
change_bg.color_bg("C:/Users/USER/Desktop/Carscan/view1.jpeg", colors = (0,0,0), output_image_name="C:/Users/USER/Desktop/Carscan/bgr1.jpeg", detect = "car")

#For 2nd image

change_bg.color_bg("C:/Users/USER/Desktop/Carscan/view2.jpeg", colors = (0,0,0), output_image_name="C:/Users/USER/Desktop/Carscan/bgr2.jpeg", detect = "car")

#For 3rd image
change_bg.color_bg("C:/Users/USER/Desktop/Carscan/view3.jpeg", colors = (0,0,0), output_image_name="C:/Users/USER/Desktop/Carscan/bgr3.jpeg", detect = "car")


#Task 3
#Replacing background

#Required package
from pixellib.tune_bg import alter_bg

New_bground = alter_bg()
New_bground.load_pascalvoc_model("C:/Users/USER/Desktop/AI_Assignments/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

#for 1st image

New_bground.change_bg_img(f_image_path = "C:/Users/USER/Desktop/Carscan/view1.jpeg",b_image_path = "C:/Users/USER/Desktop/b_ground.jpg", output_image_name="C:/Users/USER/Desktop/Carscan/new_view.jpeg")

#For 2nd image

New_bground.change_bg_img(f_image_path = "C:/Users/USER/Desktop/Carscan/view2.jpeg",b_image_path = "C:/Users/USER/Desktop/b_ground.jpg", output_image_name="C:/Users/USER/Desktop/Carscan/new_view2.jpeg")


#for 3rd image

New_bground.change_bg_img(f_image_path = "C:/Users/USER/Desktop/Carscan/view3.jpeg",b_image_path = "C:/Users/USER/Desktop/b_ground.jpg", output_image_name="C:/Users/USER/Desktop/Carscan/new_view3.jpeg")












