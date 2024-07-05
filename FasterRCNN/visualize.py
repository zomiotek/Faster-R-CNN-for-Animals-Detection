# Routines for visualizing model results and debug information.
#

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import pandas as pd

color_list1 = ['white', 'cyan', 'mediumspringgreen', 'orange', 'tomato', 'magenta', 'yellow']
color_list2 = ['white', 'red', 'red', 'red', 'red', 'red', 'red']
color_list = color_list2

def _draw_rectangle(ctx, corners, color, thickness = 4):
  y_min, x_min, y_max, x_max = corners
  ctx.rectangle(xy = [(x_min, y_min), (x_max, y_max)], outline = color, width = thickness)  

def _draw_text(image, text, position, color, scale = 1.0, offset_lines = 0):  
  font = ImageFont.truetype('arialbd.ttf', 15, encoding='unic')      
  text_size = font.getsize(text)
  text_image = Image.new(mode = "RGBA", size = text_size, color = (255, 255, 255, 100))
  ctx = ImageDraw.Draw(text_image)
  ctx.text(xy = (0, -1), text = text, font = font, fill = color)  
  scaled = text_image.resize((round(text_image.width * scale), round(text_image.height * scale)))
  position = (round(position[0]), round(position[1] + offset_lines * scaled.height))
  image.paste(im = scaled, box = position, mask = scaled)
  
def _class_to_color(class_index):    
    return color_list[class_index]    

def show_anchors(output_path, image, gt_boxes):
  ctx = ImageDraw.Draw(image, mode = "RGBA")  
  for box in gt_boxes:
    _draw_rectangle(ctx, corners = box.corners, color = (0, 255, 0), thickness = 4)      
  image.save(output_path)  
  
def show_detections(output_path, show_image, image, scored_boxes_by_class_index, class_index_to_name):  
  _, nazwaPliku = os.path.split(output_path)
  file_name, file_extension = os.path.splitext(nazwaPliku)          
  # Draw all results
  ctx = ImageDraw.Draw(image, mode = "RGBA")
  color_idx = 0
  df = pd.DataFrame(columns=['class_name', 'score', 'x_min', 'y_min', 'x_max', 'y_max'])        
  for class_index, scored_boxes in scored_boxes_by_class_index.items():        
    for i in range(scored_boxes.shape[0]):
      scored_box = scored_boxes[i,:]      
      class_name = class_index_to_name[class_index]      
            
      # SAVING THE PREDICTION DATA TO A FILE
      y_min=int(scored_box[0])
      x_min=int(scored_box[1])
      y_max=int(scored_box[2])
      x_max=int(scored_box[3])
      score=scored_box[4]      
      new_row = pd.DataFrame({'class_name': class_name, 'score': score, 'x_min': [x_min+1], 'y_min': [y_min+1], 'x_max': [x_max+1], 'y_max': [y_max+1]})                
      df = pd.concat([df, new_row])                         
      # *********************************
      
      text = " %s: %1.0f%s " % (class_name, 100*scored_box[4], '%')      
      color = _class_to_color(class_index = class_index)
      _draw_rectangle(ctx = ctx, corners = scored_box[0:4], color = color, thickness = 4)                 
      _draw_text(image = image, text = text, position = (scored_box[1], scored_box[0]-2), color = 'black', scale = 3, offset_lines = -1)            
  df.to_csv('./pred/' + file_name + '.txt', index=False, sep=' ', header=False)
  # Output
  if show_image:
    image.show()
  if output_path is not None:
    image.save(output_path)    
    print("Wrote detection results to '%s'" % output_path)
  return image
