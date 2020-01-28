
# coding: utf-8

# # Programming Project #1: Hybrid Images

# ## CS445: Computational Photography - Spring 2020

# ### Part I: Hybrid Images

# In[2]:


import cv2

import numpy as np
from matplotlib.colors import LogNorm
from scipy import signal

import utils


# In[3]:


get_ipython().magic(u'matplotlib notebook')
import matplotlib.pyplot as plt


# In[10]:


im1_file = './Nutmeg.jpg'
im2_file = './DerekPicture.jpg'

im1 = cv2.imread(im1_file, cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(im2_file, cv2.IMREAD_GRAYSCALE)


# In[11]:


pts_im1 = utils.prompt_eye_selection(im1)


# In[12]:


pts_im2 = utils.prompt_eye_selection(im2)


# In[13]:


im1, im2 = utils.align_images(im1_file, im2_file,pts_im1,pts_im2,save_images=False)


# In[14]:


# convert to grayscale
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) / 255.0
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) / 255.0


# In[15]:


#Images sanity check
fig, axes = plt.subplots(1, 2)
axes[0].imshow(im1,cmap='gray')
axes[0].set_title('Image 1'), axes[0].set_xticks([]), axes[0].set_yticks([])
axes[1].imshow(im2,cmap='gray')
axes[1].set_title('Image 2'), axes[1].set_xticks([]), axes[1].set_yticks([]);


# In[10]:


def hybridImage(im1, im2, cutoff_low, cutoff_high):
    '''
    Inputs:
        im1:    RGB (height x width x 3) or a grayscale (height x width) image
                as a numpy array.
        im2:    RGB (height x width x 3) or a grayscale (height x width) image
                as a numpy array.
        cutoff_low: standard deviation for the low-pass filter
        cutoff_high: standard deviation for the high-pass filter
        
    Output:
        Return the combination of both images, one filtered with a low-pass filter
        and the other with a high-pass filter.
    '''    


# In[ ]:


arbitrary_value = 20  # you should choose meaningful values; you might want to set to a fraction of image size
cutoff_low = arbitrary_value
cutoff_high = arbitrary_value

im_hybrid = hybridImage(im1, im2, cutoff_low, cutoff_high)


# In[ ]:


# Optional: Select top left corner and bottom right corner to crop image
# the function returns dictionary of 
# {
#   'cropped_image': np.ndarray of shape H x W
#   'crop_bound': np.ndarray of shape 2x2
# }
cropped_object = utils.interactive_crop(im_hybrid)


# ### Part II: Image Enhancement

# ##### Two out of three types of image enhancement are required.  Choose a good image to showcase each type and implement a method.  This code doesn't rely on the hybrid image part.

# #### Contrast enhancement

# #### Color enhancement 

# #### Color shift
