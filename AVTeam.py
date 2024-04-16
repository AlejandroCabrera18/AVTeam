import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
from matplotlib import pyplot as plt
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
import numpy as np
import matplotlib.patches as mpatches
from numpy import asarray
from PIL import Image as ImageReader
import gluoncv
from posix import replace
import pandas as pd
import os

#Getting all the images to be predicted by the models in the images directory
directory_path = 'images/'
directory_files = os.listdir(directory_path)
poly_images = []
for filename in directory_files:
  if "gtFine" not in filename:
    poly_images.append(filename)
print(poly_images)

#creating a function which will get the name of the ground truth image by replacing the name of the original image 
#all the images should and will resprect the same pattern
def getGroundTruthImage(imageName):
  return imageName.replace('.jpg','_gtFine_color.png')

for myimg in poly_images:
  print(getGroundTruthImage(myimg))

from PIL import *
#Creating a function to get the number of pixels for the road 
def getRoadPixels(fileName): 
  im = ImageReader.open(fileName, 'r').convert('RGB')
  purple = 0
  for pixel in im.getdata():
    if pixel == (128,64,128):
      purple += 1
  return purple
  
def transformNonPurp(fileName): 
  im = Image.open(fileName).convert('RGB')
  # Extracting pixel map:
  pixel_map = im.load()
  width, height = im.size
  # taking half of the width:
  for i in range(width):
    for j in range(height):
      # getting the RGB pixel value.
        r, g, b = im.getpixel((i, j))
        if (int(r) != 128) and (int(g) != 64) and (int(b) != 128):
          pixel_map[i, j] = (0,0,0)
  im.save("images/purple_image.png",format="png")   
  
#Code for mIoU, Pixel Accuracy, and F1 Score
def mIoU(gtPixels,predPixels):
  #Shows all the pixels where prediction does not align with GT
  inacc_pixels = np.where(gtPixels != predPixels)
  #Shows all the pixels where prediction does align with GT
  acc_pixels = np.where(gtPixels == predPixels)
  #Shows all the pixels that are black in GT
  zero_pixels = np.where(gtPixels == 0)
  #Shows all the pixels that are purple in GT
  purp_pixels = np.where(gtPixels != 0)

  #Counts all pixels that say they are black, when they should be purple
  fn_pixels = np.setdiff1d(inacc_pixels,purp_pixels)
  #Counts all pixels that say they are purple, when they should be black
  fp_pixels = np.setdiff1d(inacc_pixels,zero_pixels)
  #Counts all pixels that say they are black, when they should be black
  tn_pixels = np.setdiff1d(acc_pixels,purp_pixels)
  #Counts all pixels that say they are purple, when they should be purple
  tp_pixels = np.setdiff1d(acc_pixels,zero_pixels)

  fp_score = len(fp_pixels)
  fn_score = len(fn_pixels)
  tp_score = len(tp_pixels)
  tn_score = len(tn_pixels)

  #Iou Score
  IoU = (tp_score)/(tp_score+fp_score+fn_score)
  print(f"IoU: {100*IoU}%")

  #Testing accuracy
  num_of_black = len(zero_pixels[0])
  num_of_purp = len(purp_pixels[0])
  print(f"Road Pixel Accuracy: {100*(tp_score/num_of_purp)}%")
  print(f"Non-Road Pixel Accuracy: {100*(tn_score/num_of_black)}%")
  print(f"Total Pixel Accuracy: {100*((tp_score+tn_score)/(num_of_black+num_of_purp))}%")

  #F1 Score
  f1_score = (tp_score)/(tp_score+(0.5*(fp_score+fn_score)))
  print(f"F1 Score: {f1_score}%")

#here is link to official rgb for city scapes label https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
def addLabel():
  road = mpatches.Patch(color= '#804080', label='road')
  plt.legend(handles=[road])
  
def fullProcess(fileName, modelName):
  img = image.imread(fileName)
  #normalize the image using dataset mean
  img = test_transform(img, ctx)
  #uncomment this if you want to see image before applying model
  #plt.imshow(img.asnumpy())
  #plt.show()
  #load the model
  model = gluoncv.model_zoo.get_model(modelName, pretrained=True)
  #make prediction using single scale
  output = model.predict(img)
  predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
  #Add color pallete for visualization
  mask = get_color_pallete(predict, 'citys')
  mask.save("output.png")
  #show predicted mask
  mmask = mpimg.imread("output.png")
  plt.title(f'{fileName} - {modelName}')
  plt.imshow(mmask)
  addLabel()
  plt.show()
  roadPixels = getRoadPixels("output.png")
  print(f'Predicted road pixels for {modelName} is {roadPixels}')

  transformNonPurp("output.png")  

  #Opening the ground truth and model generated photos
  gt_im = Image.open("images/FL_POLY_6_gtFine_color.png")
  pred_im = Image.open("images/purple_image.png")

  #Converting the photos to arrays
  gt_im_array = np.array(gt_im)
  pred_im_array = np.array(pred_im)
  #print(gt_im_array)
  #print(pred_im_array)

  #Flattening the arrays
  gt_pixels = gt_im_array.reshape(-1)
  pred_pixels = pred_im_array.reshape(-1)
  #Put all this into a function

  #Splitting the arrays by 3 (causing the steps to go by 3)
  gt_pixels = gt_pixels[::3]
  pred_pixels = pred_pixels[::3]

  #pixel accuracy is close but not exactly the same to mIoU output
  mIoU(gt_pixels,pred_pixels)

#desiredPixels = getRoadPixels('images/FL_POLY_6_gtFine_color.png')
#print(f'Desired  road pixels based on ground truth is {desiredPixels}')
models = ['deeplab_resnet101_citys']#,'psp_resnet101_citys','icnet_resnet50_citys'] #commented this out so it'd be faster
for model in models:
  fullProcess('images/FL_POLY_6.jpg', model)
  

