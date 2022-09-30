import matplotlib.pyplot as plt

# Optional settings for the plots. Comment out if needed.
import seaborn as sb
sb.set_context('poster')

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12., 9.6)


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, binary_closing
from skimage.io import imsave, imread
import pandas as pd
from fil_finder import FilFinder2D
import astropy.units as u
from PIL import Image




files = [] # detect .tif files in input folder
for file in os.listdir(os.path.join('input', 'crop')):
    if file.endswith('.tif'): files += [file] 

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












# Can analyze the axes to make density plots, but how to detect peaks?
plt.plot(thresh.sum(axis=0))
plt.show()

plt.plot(thresh.sum(axis=1))
plt.show()

for file in files:
    img = cv2.imread(os.path.join('input', 'crop', file)) # read image in loop
    
    # threshold magenta
    lower = np.array([20, 0, 20])
    upper = np.array([255, 150, 255])
    thresh = cv2.inRange(img, lower, upper)
    blur = cv2.blur(thresh, (2,2))
    skel = np.where(blur > 0.5, 1, 0)
    skeleton = skeletonize(skel) # perform skeletonization
    skeleton_closed = binary_closing(skeleton)
    
    # cv2.imshow("skel",img)
    # cv2.imshow("skel",thresh)
    # cv2.imshow("skel",blur)
    
    im = Image.fromarray(skeleton_closed)
    im.save(os.path.join('input', 'skeleton', file))





files = [] # detect .tif files in input folder
for file in os.listdir(os.path.join('input', 'skeleton')):
    if file.endswith('.tif'): files += [file] 

for file in files:
    img = cv2.imread(os.path.join('input', 'crop', file)) # read image in loop
    skeleton = cv2.imread(os.path.join('input', 'skeleton', file), 0) #in numpy array format
    fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton, beamwidth=0*u.pix)
    # fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=False, verbose=False, use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=10* u.pix, skel_thresh=500 * u.pix, prune_criteria='intensity')
    fil.exec_rht(verbose=True, save_png=True, save_name=os.path.join('input', 'skeleton', file) + '.png')





"""Run findfill"""
# https://stackoverflow.com/questions/53481596/python-image-finding-largest-branch-from-image-skeleton

skeleton = cv2.imread(os.path.join('input', 'skeleton', file), 0) #in numpy array format
fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
fil.preprocess_image(flatten_percent=85)
fil.create_mask(border_masking=False, verbose=False, use_existing_mask=True)
fil.medskel(verbose=True)
fil.analyze_skeletons(branch_thresh=10* u.pix, skel_thresh=500 * u.pix, prune_criteria='intensity')

# Show the longest path
plt.imshow(fil.skeleton, cmap='gray')
plt.contour(fil.skeleton_longpath, colors='r')
plt.axis('off')
plt.show()


fil.lengths() # Get length of branches

fil.exec_rht(verbose=True, save_png=True, save_name=os.path.join('input', 'skeleton', file) + '.png')

fil.find_widths(max_dist=0.2*u.pc)


fil.total_intensity()

fil.output_table()



"""Testing purposes"""
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(thresh, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

ax[2].imshow(skeleton_closed, cmap=plt.cm.gray)
ax[2].axis('off')
ax[2].set_title('skeleton_closed', fontsize=20)

fig.tight_layout()
plt.show()


file = files[3]

img = cv2.imread(os.path.join('input', 'crop', file))

# threshold magenta
lower = np.array([20, 0, 20])
upper = np.array([255, 150, 255])
thresh = cv2.inRange(img, lower, upper)

# Sum of all pixel values in specified range
thresh.sum()

blur = cv2.blur(thresh, (2,2))
skel = np.where(blur > 0.5, 1, 0)

# perform skeletonization
skeleton = skeletonize(skel)

skeleton_closed = binary_closing(skeleton)

cv2.imshow("skel",img)
cv2.imshow("skel",thresh)
cv2.imshow("skel",blur)


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(thresh, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

ax[2].imshow(skeleton_closed, cmap=plt.cm.gray)
ax[2].axis('off')
ax[2].set_title('skeleton_closed', fontsize=20)

fig.tight_layout()
plt.show()


# This skeletonize result looks fine, now measure total length + longest length?
# Use SKAN https://skeleton-analysis.org/stable/
# OR FillFinder

# Use fillfinder on existing skeleton, use this answer?
# https://stackoverflow.com/questions/53481596/python-image-finding-largest-branch-from-image-skeleton

fil = FilFinder2D(skeleton_closed, distance=250 * u.pc, mask=skeleton)
fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
fil.medskel(verbose=False)
fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length', max_prune_iter=50)

# Show the longest path
plt.imshow(fil.skeleton, cmap='gray')
plt.contour(fil.skeleton_longpath, colors='r')
plt.axis('off')
plt.show()



