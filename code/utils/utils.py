import os, pdb
import sys
import time
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
#######################################################
# Auxiliary matrices used to solve DLT
Aux_M1  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)


Aux_M2  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ]], dtype=np.float64)



Aux_M3  = np.array([
          [0],
          [1],
          [0],
          [1],
          [0],
          [1],
          [0],
          [1]], dtype=np.float64)



Aux_M4  = np.array([
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)


Aux_M5  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)



Aux_M6  = np.array([
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ]], dtype=np.float64)


Aux_M71 = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)


Aux_M72 = np.array([
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ]], dtype=np.float64)



Aux_M8  = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ]], dtype=np.float64)


Aux_Mb  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , -1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)


########################################################
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
    return tot_time

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def save_correspondences_img(img1, img2, corr1, corr2, pred_corr2, results_dir, img_name):
  """ Save pair of images with their correspondences into a single image. Used for report"""
  # Draw prediction
  copy_img2 = img2.copy()
  copy_img1 = img1.copy()
  cv2.polylines(copy_img2, np.int32([pred_corr2]), 1, (5, 225, 225),3)

  point_color = (0,255,255)
  line_color_set = [(255,102,255), (51,153,255), (102,255,255), (255,255,0), (102, 102, 244), (150, 202, 178), (153,240,142), (102,0,51), (51,51,0) ]
  # Draw 4 points (ground truth)
  full_stack_images = draw_matches(copy_img1, corr1, copy_img2 , corr2, '/tmp/tmp.jpg', color_set = line_color_set, show=False)
  # Save image
  visual_file_name = os.path.join(results_dir, img_name)
  #cv2.putText(full_stack_images, 'RMSE %.2f'%h_loss,(800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
  cv2.imwrite(visual_file_name, full_stack_images)
  print('Wrote file %s', visual_file_name)





def draw_matches(img1, kp1, img2, kp2, output_img_file=None, color_set=None, show=True):
    """Draws lines between matching keypoints of two images without matches.
    This is a replacement for cv2.drawMatches
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        color_set: The colors of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    # Draw lines between points

    kp2_on_stack_image = (kp2 + np.array([img1.shape[1], 0])).astype(np.int32)

    kp1 = kp1.astype(np.int32)
    # kp2_on_stack_image[0:4,0:2]
    line_color1 = (2, 10, 240)
    line_color2 = (2, 10, 240)
    # We want to make connections between points to make a square grid so first count the number of rows in the square grid.
    grid_num_rows = int(np.sqrt(kp1.shape[0]))

    if output_img_file is not None and grid_num_rows >= 3:
        for i in range(grid_num_rows):
            cv2.line(new_img, tuple(kp1[i*grid_num_rows]), tuple(kp1[i*grid_num_rows + (grid_num_rows-1)]), line_color1, 1,  LINE_AA)
            cv2.line(new_img, tuple(kp1[i]), tuple(kp1[i + (grid_num_rows-1)*grid_num_rows]), line_color1, 1,  cv2.LINE_AA)
            cv2.line(new_img, tuple(kp2_on_stack_image[i*grid_num_rows]), tuple(kp2_on_stack_image[i*grid_num_rows + (grid_num_rows-1)]), line_color2, 1,  cv2.LINE_AA)
            cv2.line(new_img, tuple(kp2_on_stack_image[i]), tuple(kp2_on_stack_image[i + (grid_num_rows-1)*grid_num_rows]), line_color2, 1,  cv2.LINE_AA)

    if output_img_file is not None and grid_num_rows == 2:
        cv2.polylines(new_img, np.int32([kp2_on_stack_image]), 1, line_color2, 3)
        cv2.polylines(new_img, np.int32([kp1]), 1, line_color1, 3)
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 7
    thickness = 1

    for i in range(len(kp1)):
        key1 = kp1[i]
        key2 = kp2[i]
        # Generate random color for RGB/BGR and grayscale images as needed.
        try:
            c  = color_set[i]
        except:
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(key1).astype(int))
        end2 = tuple(np.round(key2).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness,  cv2.LINE_AA)
        cv2.circle(new_img, end1, r, c, thickness,  cv2.LINE_AA)
        cv2.circle(new_img, end2, r, c, thickness,  cv2.LINE_AA)
    # pdb.set_trace()
    if show:
        plt.figure(figsize=(15,15))
        if len(img1.shape) == 3:
            plt.imshow(new_img)
        else:
            plt.imshow(new_img)
        plt.axis('off')
        plt.show()
    if output_img_file is not None:
        cv2.imwrite(output_img_file, new_img)

    return new_img


def show_image(location, title, img, width=None):
    if width is not None:
        plt.figure(figsize=(width, width))
    plt.subplot(*location)
    plt.title(title, fontsize=8)
    plt.axis('off')
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    if width is not None:
        plt.show()
        plt.close()



def find_nine_grid_points(width, height, offset=50):
  y = np.linspace(offset,height-offset,3).astype(np.int32)
  x = np.linspace(offset,width-offset,3).astype(np.int32)
  #x = x[1:-1]
  #y = y[1:-1]
  [X, Y] = np.meshgrid(x, y)
  X =  X.flatten().reshape([9,1])
  Y = Y.flatten().reshape([9,1])
  show_nine_points = np.hstack((X, Y)).reshape([9,2])
  return show_nine_points




def removeHiddenfile(directory_list):
  if '.' in directory_list:
    directory_list.remove('.')
  if '..' in directory_list:
    directory_list.remove('.')
  if '.DS_Store' in directory_list:
    directory_list.remove('.DS_Store')
  return directory_list


def test_progress_bar():
    total = 20000
    for i in range(total):
        progress_bar(i, total, 'printing %d'%i)


def count_text_lines(file_path):
  f = open(file_path, 'r')
  lines =  f.readlines()
  f.close()
  return len(lines)


# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def get_average_grads(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
  # Note that each grad_and_vars looks like the following:
  #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def denorm_img(img):
  if len(img.shape) == 3:
    mean = np.array([118.93, 113.97, 102.60]).reshape([1,1,3])
    std  = np.array([69.85, 68.81, 72.45]).reshape([1,1,3])
  elif len(img.shape) == 2:
    mean = np.mean([118.93, 113.97, 102.60])
    std  = np.mean([69.85, 68.81, 72.45])
  return img*std + mean


def get_mesh_grid_per_img(c_w, c_h, x_start=0, y_start=0):
  """Get 1D array of indices of pixels in the image of size c_h x c_w"""
  x_t = tf.matmul(tf.ones([c_h, 1]),
            tf.transpose(\
                tf.expand_dims(\
                    tf.linspace(tf.cast(x_start,'float32'), tf.cast(x_start+c_w-1,'float32'), c_w), 1), [1,0]))
  y_t = tf.matmul(tf.expand_dims(\
                    tf.linspace(tf.cast(y_start,'float32'), tf.cast(y_start+c_h-1,'float32'), c_h), 1),
                  tf.ones([1, c_w]))
  x_t_flat = tf.reshape(x_t, [-1]) # 1 x D
  y_t_flat = tf.reshape(y_t, [-1])
  return  x_t_flat, y_t_flat

def get_batch_symetric_census(img, kernel_size=(15, 15), index=None, debug=False):
  """img = N x H x D x C or N x H x D or H x D x C"""
  img_shape = img.get_shape().as_list()
  if len(img_shape) == 2: # Single channel, single image
    img = tf.expand_dims(img, [0]) # 1 x H x D
  if len(img_shape) == 3 and img_shape[2]<=3: # Multiple channel, single image
    img = tf.reduce_mean(img, 2)
    img = tf.expand_dims(img, [0]) # 1 x H x D
  if len(img_shape) == 4:        # Multiple channel, multiple images
    img = tf.reduce_mean(img, 3) # N x H x D
  # Suppose that image size is H x W
  batch_size, img_h, img_w = img.get_shape().as_list()
  # Census kernel size
  c_h, c_w = kernel_size

  # Get meshgrid for the whole original image
  x_img_flat, y_img_flat = get_mesh_grid_per_img(img_w, img_h)

  # Reshape to (HxW, 1)
  x_img_col = tf.reshape(x_img_flat, [img_h*img_w, 1])
  y_img_col = tf.reshape(y_img_flat, [img_h*img_w, 1])

  # Zero pad the images
  p_h, p_w = int(c_h/2) , int(c_w/2)
  img = tf.pad(img, [[0,0],  [p_h, p_h], [p_w, p_w]])

  # Image now is bigger, after padding
  pad_img_w, pad_img_h = img_w + 2*p_w, img_h + 2*p_h

  # Get meshgrid for the base kernel
  x_kernel_flat, y_kernel_flat = get_mesh_grid_per_img(c_w, c_h)

  # In case we want to compute census transform for a batch of images, batch_indices needed
  # to take into account
  batch_indices_tensor = tf.matmul(
                                tf.expand_dims(
                                    tf.linspace(tf.cast(0, tf.float32),tf.cast((batch_size-1)*pad_img_h*pad_img_w,tf.float32),
                                           batch_size),
                                    1),
                                tf.ones([1, img_h*img_w*c_w*c_h])) #[batch_size, img_h*img_w*c_w*c_h]
  # Compute indices for img_h*img_w patches
  patch_indices = (y_img_col + y_kernel_flat)*pad_img_w + (x_img_col + x_kernel_flat) #[img_h*img_w, c_w*c_h]

  # x_start, y_start = 0, 0
  # patch_indices = (y_kernel_flat + y_start)*pad_img_w + (x_kernel_flat + x_start)

  patch_indices = tf.reshape(patch_indices, [1, -1]) #[1, img_h*img_w*c_w*c_h]
  patch_indices=  tf.cast(patch_indices + batch_indices_tensor, tf.int32)
  patch_indices_flat = tf.reshape(patch_indices, [-1]) #[batch_size*img_h*img_w*c_w*c_h, ]
  # Flatten the image
  img_flat = tf.reshape(img, [-1])

  # Obtain the patch
  patch_flat = tf.gather(img_flat, patch_indices_flat)

  # Reshap patch_flat
  patch_flat = tf.reshape(patch_flat, [batch_size*img_h*img_w, c_w*c_h])
  # Reverse the patch
  patch_flat_trans = tf.reverse(patch_flat, [1])


  # Get the census for every patch
  #patch_censuses = tf.greater_equal(patch_flat, patch_flat_trans)[:, 0:int(c_w*c_h/2)] # (NxHxW, 31)

    # Get the census for every patch
  #patch_censuses = tf.sigmoid((patch_flat - patch_flat_trans)/(255.0/6.0))[:, 0:int(c_w*c_h/2)] # (NxHxW, 31)
  patch_censuses = tf.identity((patch_flat - patch_flat_trans)/255.)[:, 0:int(c_w*c_h/2)] # (NxHxW, 31)

  ## Convert to binary
  pixel_censuses = tf.reduce_sum(tf.cast(tf.reverse(tensor=patch_censuses, axis=[1]), dtype=tf.float32)
    * 2 ** tf.range(tf.cast(int(c_w*c_h/2), dtype=tf.float32)), 1)/2**int(c_w*c_h/2)

  ## Compute census value number for every patch
  pixel_censuses = tf.reduce_sum(tf.cast(patch_censuses, dtype=tf.float32), 1)
  #pixel_censuses = patch_censuses

  # Reshape to original image size
  img_censuses = tf.reshape(pixel_censuses, [batch_size, img_h, img_w, 1])
  #img_censuses = tf.reshape(pixel_censuses, [batch_size, img_h, img_w, -1])

  if debug:
    return img, patch_flat[index], pixel_censuses[index], img_censuses
  else:
    return img_censuses

def get_symetric_census(img, kernel_size=(3, 3), index=None, debug=False):
  img_shape = img.get_shape().as_list()
  if len(img_shape) == 3:
    img = tf.reduce_mean(img, 2)
  if len(img_shape) == 4:
    img = tf.reduce_mean(img[0], 2)
  # Suppose that image size is H x W
  img_h, img_w = img.get_shape().as_list()
  # Census kernel size
  c_h, c_w = kernel_size

  # Get meshgrid for the whole original image
  x_img_flat, y_img_flat = get_mesh_grid_per_img(img_w, img_h)

  # Reshape to (HxW, 1)
  x_img_col = tf.reshape(x_img_flat, [img_h*img_w, 1])
  y_img_col = tf.reshape(y_img_flat, [img_h*img_w, 1])

  # Zero pad the images
  p_h, p_w = int(c_h/2) , int(c_w/2)
  img = tf.pad(img, [ [p_h, p_h], [p_w, p_w]])

  # Image now is bigger, after padding
  pad_img_w, pad_img_h = img_w + 2*p_w, img_h + 2*p_h

  # Get meshgrid for the base kernel
  x_kernel_flat, y_kernel_flat = get_mesh_grid_per_img(c_w, c_h)

  # Compute indices for img_h*img_w patches
  patch_indices = (y_img_col + y_kernel_flat)*pad_img_w + (x_img_col + x_kernel_flat)

  # x_start, y_start = 0, 0
  # patch_indices = (y_kernel_flat + y_start)*pad_img_w + (x_kernel_flat + x_start)

  patch_indices=  tf.cast(patch_indices, tf.int32)
  # Flatten the image
  img_flat = tf.reshape(img, [-1])

  # Obtain the patch
  patch_flat = tf.gather(img_flat, patch_indices)

  # Reverse the patch
  patch_flat_trans = tf.reverse(patch_flat, [1])


  # Get the census for every patch
  patch_censuses = tf.greater_equal(patch_flat, patch_flat_trans)[:, 0:int(c_w*c_h/2)] # (NxHxW, 31)

  # Convert to binary
  pixel_censuses = tf.reduce_sum(tf.cast(tf.reverse(tensor=patch_censuses, axis=[1]), dtype=tf.float32)
    * 2 ** tf.range(tf.cast(int(c_w*c_h/2), dtype=tf.float32)), 1)/2**int(c_w*c_h/2)

  # # Compute census value number for every patch
  # pixel_censuses = tf.reduce_sum(tf.cast(patch_censuses, dtype=tf.float32), 1)

  # Reshape to original image size
  img_censuses = tf.reshape(pixel_censuses, [img_h, img_w, 1])
  if debug:
    return img, patch_flat[index], pixel_censuses[index], img_censuses
  else:
    return img_censuses

def test_census_img():
  # Initialize an image for testing
  img_np = np.array([[1,  2,  3,  4,  5,  6,  7,  8,  9],
                   [11, 12, 13, 14, 15, 16, 17, 18, 19],
                   [1,  2,  3,  4,  5,  6,  7,  8,  9],
                   [1,  1,  1,  1,  1,  1,  1,  1,  1],
                   [5,  5,  5,  5,  5,  5,  5,  5,  5]]).reshape([5, 9, 1])
  img_np = np.tile(img_np, [1, 1, 3])

  img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32)

  # Which pixel want to compute censuse
  index_tf = tf.placeholder(dtype=tf.int32, shape=[])

  # Get the gray image
  gray_tf = tf.reduce_mean(img_tf, [2])


  # Get cencus image
  gray_tf, patch_img_tf, cencus_pixel_tf, census_img_tf = get_symetric_census(gray_tf, (3, 5),index_tf, True)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  # For debugging: compute census for HxW pixels, though get_symetric_census() can compute all at once
  for i in range(5):
    for j in range(9):
      index = i*9 + j
      print('\n========> Compute census for pixel %d with (y, x) = (%d, %d) (on original image)'%(index, i, j))
      img, patch_img, cencus_pixel, census_img = sess.run([gray_tf, patch_img_tf, cencus_pixel_tf, census_img_tf], feed_dict={index_tf:index})

      print('\nGray image after zero-padded: \n', img)
      print('\nPatch               :', patch_img)
      print('Reversed patch      :', patch_img[::-1])
      print('Corresponding census:', cencus_pixel)

  print('\n======> Census image:\n', census_img)

def test_batch_census_img():
  # Initialize an image for testing
  img_np = np.array([[1,  2,  3,  4,  5,  6,  7,  8,  9],
                   [11, 12, 13, 14, 15, 16, 17, 18, 19],
                   [1,  2,  3,  4,  5,  6,  7,  8,  9],
                   [1,  1,  1,  1,  1,  1,  1,  1,  1],
                   [5,  5,  5,  5,  5,  5,  5,  5,  5]]).reshape([5, 9, 1])
  img_np = np.tile(img_np, [1, 1, 3])

  img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32)

  # Which pixel want to compute censuse
  index_tf = tf.placeholder(dtype=tf.int32, shape=[])

  # Get the gray image
  #gray_tf = tf.reduce_mean(img_tf, [2])
  gray_tf = tf.expand_dims(img_tf, [0])
  gray_tf = tf.stack([img_tf, img_tf], 0)
  # Get cencus image
  gray_tf, patch_img_tf, cencus_pixel_tf, census_img_tf = get_batch_symetric_census(gray_tf, (3, 5),index_tf, True)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  # For debugging: compute census for HxW pixels, though get_symetric_census() can compute all at once
  for i in range(5):
    for j in range(9):
      index = i*9 + j
      print('\n========> Compute census for pixel %d with (y, x) = (%d, %d) (on original image)'%(index, i, j))
      img, patch_img, cencus_pixel, census_img = sess.run([gray_tf, patch_img_tf, cencus_pixel_tf, census_img_tf], feed_dict={index_tf:index})

      print('\nGray image after zero-padded: \n', img)
      print('\nPatch               :', patch_img)
      print('Reversed patch      :', patch_img[::-1])
      print('Corresponding census:', cencus_pixel)

  print('\n======> Census image:\n', census_img[:,:,:,0])


def find_percentile(x):
  x_sorted = np.sort(x)
  len_x = len(x_sorted)

  # Find mean, var of top 20:
  tops_list = [0.3, 0.6, 1]
  return_list = []
  start_index = 0
  for i in range(len(tops_list)):
    print('>> Top %.0f -  %.0f %%'%(tops_list[i-1]*100 if i >=1 else 0, tops_list[i]*100))
    stop_index = int(tops_list[i]*len_x)

    interval = x_sorted[start_index:stop_index]
    interval_mu = np.mean(interval)
    interval_std = np.std(interval)

    start_index = stop_index
    return_list.append([interval_mu, interval_std])
  return np.array(return_list)

def test_find_percentile():
  x = [10, 1.10,2 , 3 , 4 , 5,  6, 6, 7, 8, 9]
  print('===== Input ======\n', x)
  tops_list = find_percentile(x)
  print(tops_list)

if __name__=="__main__":
  test_find_percentile()
  #test_batch_census_img()
  #test_progress_bar()
