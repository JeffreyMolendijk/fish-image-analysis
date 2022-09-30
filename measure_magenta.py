import os
import numpy as np
import cv2
import pandas as pd


def measure_magenta():

    files = [] # detect .tif files in crop folder
    for file in os.listdir(os.path.join('input', 'crop')):
        if file.endswith('.tif'): files += [file]
        if file.endswith('.png'): files += [file] 
    
    magenta_sum = pd.DataFrame(columns=['file', 'magenta', 'length', 'width'])
    
    # loop that makes thresh, then records the sum to a pd
    for file in files:
        img = cv2.imread(os.path.join('input', 'crop', file)) # read image in loop
        
        # threshold magenta
        lower = np.array([80, 0, 80])
        upper = np.array([255, 150, 255])
        thresh = cv2.inRange(img, lower, upper)
        
        fish_width_density = thresh.sum(axis=0)
        fish_width = np.where(fish_width_density > (fish_width_density.max()/20))[0][-1] - np.where(fish_width_density > (fish_width_density.max()/20))[0][1]
        fish_length_density = thresh.sum(axis=1)
        fish_length = np.where(fish_length_density > (fish_length_density.max()/20))[0][-1] - np.where(fish_length_density > (fish_length_density.max()/20))[0][1]
        
        magenta_sum = magenta_sum.append({'file': file, 'magenta': thresh.sum(), 'length': fish_length, 'width': fish_width}, ignore_index=True)
        cv2.imwrite(os.path.join('input', 'magenta', file), thresh)
    
    magenta_sum.to_excel(os.path.join('input', 'magenta', 'magenta.xlsx'))
