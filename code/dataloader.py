from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pdb, os
import argparse
from matplotlib import pyplot as plt
import numpy as np
from collections import namedtuple
import skimage.io as io
from utils.utils import get_mesh_grid_per_img, get_symetric_census

dataloader_params = namedtuple('parameters',
                              'data_path,'
                              'filenames_file,'
                              'pts1_file,'
                              'gt_file,'
                              'mode,'
                              'batch_size,'
                              'img_h,'
                              'img_w,'
                              'patch_size,'
                              'augment_list,'
                              'do_augment,')
# Create an extended version to deal with real data set where

# a data sample consist of images of full-size, small size and crop (patch)
extended_dataloader_params = namedtuple('extended_parameters',
                              'data_path,'
                              'filenames_file,'
                              'pts1_file,'
                              'gt_file,'
                              'mode,'
                              'batch_size,'
                              'img_h,'
                              'img_w,'
                              'patch_size,'
                              'augment_list,'
                              'do_augment,'
                              'full_img_h,'
                              'full_img_w')

def string_length_tf(t):
    """Wrap python functions to tensor function.
    This function returns len of a string"""
    return tf.py_func(len, [t], [tf.int64])




def read_img_and_gt(filenames_file, pts1_file, gt_file, params):

  with open(pts1_file, 'r') as pts1_f:
    pts1_array = pts1_f.readlines()
  pts1_array = [x.strip() for  x in pts1_array]
  pts1_array = [x.split() for x in pts1_array]
  pts1_array = np.array(pts1_array).astype('float64')

  with open(filenames_file, 'r') as img_f:
    img_array = img_f.readlines()
  img_array = [x.strip() for  x in img_array]
  img_array = [x.split() for x in img_array] # Use x.split()[0] if assuming image left and right have same name

  # In case there is not ground truth
  if not gt_file:
    return img_array, pts1_array, None

  with open(gt_file, 'r') as gt_f:
    gt_array = gt_f.readlines()
  gt_array = [x.strip() for  x in gt_array]
  gt_array = [x.split() for x in gt_array]
  gt_array = np.array(gt_array).astype('float64')

  return img_array, pts1_array, gt_array



class Dataloader(object):
  """Load synthetic data"""
  def __init__(self, params,shuffle=True):
    self.data_path = params.data_path
    self.params = params
    self.mode = params.mode
    self.patch_size = params.patch_size
    self.img_w = params.img_w
    self.img_h = params.img_h
    self.full_img_w = None
    self.full_img_h = None
    self.I1_batch = None
    self.I2_batch = None
    self.I1_aug_batch = None
    self.I2_aug_batch = None
    self.I_batch = None
    self.I_prime_batch = None
    self.pts1_batch = None
    self.gt_batch = None
    # Indices of pixels of the patch w.r.t the large image
    self.patch_indices_batch = None
    self.pts1_file = params.pts1_file
    self.gt_file =  params.gt_file
    self.mean_I = tf.constant([118.93, 113.97, 102.60], shape=[1, 1, 3])
    self.std_I = tf.constant([69.85, 68.81, 72.45], shape=[1, 1, 3])
    self.mean_I_prime = tf.constant([96.0, 91.38,  81.92], shape=[1, 1, 3])
    self.std_I_prime = tf.constant([77.45, 75.17, 75.18], shape=[1, 1, 3])
    self.mean_I1 = tf.constant(np.mean([118.93, 113.97, 102.60]), dtype=tf.float32)
    self.std_I1 = tf.constant(np.mean([69.85, 68.81, 72.45]), dtype=tf.float32)

    # Constants used to extract I1_aug, I2_aug from I_aug
    y_t = tf.range(0, self.params.batch_size*self.params.img_w*self.params.img_h, self.params.img_w*self.params.img_h)
    z =  tf.tile(tf.expand_dims(y_t,[1]),[1,self.params.patch_size*self.params.patch_size])
    self.batch_indices_tensor = tf.reshape(z, [-1]) # Add these value to patch_indices_batch[i] for i in range(num_pairs) # [BATCH_SIZE*WIDTH*HEIGHT]


    # Read to arrays
    img_np, pts1_np, gt_np = read_img_and_gt(self.params.filenames_file, self.pts1_file, self.gt_file, self.params)

    # Convert to tensor
    img_files = tf.convert_to_tensor(img_np, dtype=tf.string)
    pts1_tf    = tf.convert_to_tensor(pts1_np, dtype=tf.float32) # N x 2

    if self.gt_file:
      gt_tf    = tf.convert_to_tensor(gt_np, dtype=tf.float32)
      input_queue = tf.train.slice_input_producer([img_files, pts1_tf, gt_tf], shuffle=shuffle)
      gt_batch = input_queue[2]
    else:
      input_queue = tf.train.slice_input_producer([img_files, pts1_tf], shuffle=shuffle)

    # image names
    split_line  = tf.string_split(input_queue[0]).values
    filename_batch = split_line

    # Four points on the first image
    pts1_batch = input_queue[1]


    # Finding patch_indices_tf. Does not work with large data since out of memory
    ## Find indices of the pixels in the patch w.r.t the large image
    ## All patches have the same size so their pixels have the same base indices
    x_t_flat, y_t_flat = get_mesh_grid_per_img(params.patch_size, params.patch_size)



    # Read images file by file
    #if self.mode == 'test':
    I_path = tf.string_join([self.data_path,'I/', split_line[1]])
    I_prime_path = tf.string_join([self.data_path,'I_prime/', split_line[1]])

    # Check if input images are full one or not
    try:
      self.full_img_h = self.params.full_img_h
      self.full_img_w = self.params.full_img_w
    except:
    # If the input images are not full ones
      self.full_img_h = self.img_h
      self.full_img_w = self.img_w
      pass
    # Just obtain images with full_img_h x full_img_w, and check later if the size is identical
    full_I = self.read_image(I_path, [self.full_img_h, self.full_img_w], channels=3)
    full_I_prime = self.read_image(I_prime_path, [self.full_img_h, self.full_img_w], channels=3)

    full_I.set_shape([None, None, 3])
    full_I_prime.set_shape([None, None, 3])

    # Augment images with artifacts
    do_augment = tf.random_uniform([], 0, 1)
    # Training: use joint augmentation (images in one pair are inserted same noise)
    # Test: use disjoint augmentation (images in one pair are inserted different noise)
    if self.mode == 'train':
        full_I_aug, full_I_prime_aug = tf.cond(do_augment > (1-self.params.do_augment), lambda:self.joint_augment_image_pair(full_I, full_I_prime, 0, 255), lambda:(full_I, full_I_prime))
    else:
        full_I_aug, full_I_prime_aug = tf.cond(do_augment > (1-self.params.do_augment), lambda:self.disjoint_augment_image_pair(full_I, full_I_prime, 0, 255), lambda:(full_I, full_I_prime))

    # Standardize images
    if 'normalize' in self.params.augment_list:
      full_I       = self.norm_img(full_I, self.mean_I, self.std_I)
      full_I_prime = self.norm_img(full_I_prime, self.mean_I, self.std_I)
      # These are augmented large images which will be used
      full_I_aug = self.norm_img(full_I_aug, self.mean_I, self.std_I)
      full_I_prime_aug = self.norm_img(full_I_prime_aug, self.mean_I, self.std_I)

    if 'per_image_normalize' in self.params.augment_list:
      # Two redundances:
      full_I = tf.image.per_image_standardization(I)
      full_I_prime = tf.image.per_image_standardization(I_prime)

      # These are augmented large images which will be used
      full_I_aug       = tf.image.per_image_standardization(full_I_aug)
      full_I_prime_aug = tf.image.per_image_standardization(full_I_prime_aug)

    # Now, check if full_img_h and img_h are identical. If not => resize full_img to get img
    if self.full_img_h == self.img_h:
      I           = full_I
      I_prime     = full_I_prime

      I_aug       = full_I_aug
      I_prime_aug = full_I_prime_aug
    else:
      I           = tf.image.resize_images(full_I,[self.img_h, self.img_w], tf.image.ResizeMethod.AREA)
      I_prime     = tf.image.resize_images(full_I_prime,[self.img_h, self.img_w], tf.image.ResizeMethod.AREA)

      I_aug       = tf.image.resize_images(full_I_aug,[self.img_h, self.img_w], tf.image.ResizeMethod.AREA)
      I_prime_aug = tf.image.resize_images(full_I_prime_aug,[self.img_h, self.img_w], tf.image.ResizeMethod.AREA)

    # Read patch_indices
    x_start_tf = pts1_batch[0] # 1,
    y_start_tf = pts1_batch[1]  # (1, )
    patch_indices_tf = (y_t_flat + y_start_tf)*self.params.img_w + (x_t_flat + x_start_tf)

    patch_indices_tf = tf.cast(patch_indices_tf, tf.int32)

    # Obtain I1, I2, I1_aug and I2_aug
    I_flat           = tf.reshape(tf.reduce_mean(I, 2), [-1]) # I: HxWx3
    I_prime_flat     = tf.reshape(tf.reduce_mean(I_prime, 2), [-1]) # I_prime: HxWx3
    I_aug_flat       = tf.reshape(tf.reduce_mean(I_aug, 2), [-1]) # I_aug: HxWx3
    I_prime_aug_flat = tf.reshape(tf.reduce_mean(I_prime_aug, 2), [-1])

    patch_indices_flat = tf.reshape(patch_indices_tf, [-1])
    pixel_indices      =  patch_indices_flat


    I1_flat     = tf.gather(I_flat, pixel_indices)
    I2_flat     = tf.gather(I_prime_flat, pixel_indices)
    I1_aug_flat = tf.gather(I_aug_flat, pixel_indices)
    I2_aug_flat = tf.gather(I_prime_aug_flat, pixel_indices)

    I1          = tf.reshape(I1_flat, [self.params.patch_size, self.params.patch_size, 1])
    I2          = tf.reshape(I2_flat, [self.params.patch_size, self.params.patch_size, 1])
    I1_aug      = tf.reshape(I1_aug_flat, [self.params.patch_size, self.params.patch_size, 1])
    I2_aug      = tf.reshape(I2_aug_flat, [self.params.patch_size, self.params.patch_size, 1])

    # If ground truth of homography is given (in synthetic case - both training and testing, in aerial image case - only test)
    if self.gt_file:
      self.full_I_batch, self.full_I_prime_batch, self.I1_batch, self.I2_batch, self.I1_aug_batch, self.I2_aug_batch, self.I_batch, self.I_prime_batch, self.pts1_batch, self.gt_batch, self.patch_indices_batch = tf.train.shuffle_batch([full_I_aug, full_I_prime_aug, I1, I2, I1_aug, I2_aug, I_aug, I_prime_aug, pts1_batch, gt_batch, patch_indices_tf], self.params.batch_size,
                         self.params.batch_size*10, self.params.batch_size*4, 20) # number of threads = 20

    else:
      self.full_I_batch, self.full_I_prime_batch, self.I1_batch, self.I2_batch, self.I1_aug_batch, self.I2_aug_batch, self.I_batch, self.I_prime_batch, self.pts1_batch, self.patch_indices_batch = tf.train.shuffle_batch([full_I_aug, full_I_prime_aug,I1, I2, I1_aug, I2_aug, I_aug, I_prime_aug, pts1_batch, patch_indices_tf], self.params.batch_size, self.params.batch_size*10, self.params.batch_size*4, 20) # number of threads = 20


  def read_image(self, image_path, out_size, channels=3):

    path_length = string_length_tf(image_path)[0]
    file_extension = tf.substr(image_path, path_length-3, 3)
    file_cond = tf.equal(file_extension, 'jpg')

    image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path), channels=channels), lambda: tf.image.decode_png(tf.read_file(image_path), channels=channels))
    image = tf.cast(image, tf.float32)
    #image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, out_size, tf.image.ResizeMethod.AREA)
    return image


  def get_mean_std(self):
    img_np, _, _ = read_img_and_gt(self.params.filenames_file, self.pts1_file, self.gt_file, self.params)

    # Convert to tensor
    img_files = tf.convert_to_tensor(img_np, dtype=tf.string)
    input_queue = tf.train.slice_input_producer([img_files], shuffle=False)

    # image names
    split_line  = tf.string_split(input_queue[0]).values
    filename_batch = split_line

    I_path = tf.string_join([self.data_path,'I/', split_line[1]])
    I_prime_path = tf.string_join([self.data_path,'I_prime/', split_line[1]])

    I = self.read_image(I_path, [self.params.img_h, self.params.img_w], channels=3)
    I_prime = self.read_image(I_prime_path, [self.params.img_h, self.params.img_w], channels=3)

    I.set_shape([None, None, 3])
    I_prime.set_shape([None, None, 3])

    batch_size = 100

    I_batch, I_prime_batch = tf.train.shuffle_batch([I, I_prime], batch_size, batch_size*10, 0, 32) # 32 threads
    mean_I_batch, std_I_batch = tf.nn.moments(I_batch, [0, 1,2])
    mean_I_prime_batch, std_I_prime_batch = tf.nn.moments(I_prime_batch, [0, 1,2])

    std_I_batch = tf.sqrt(std_I_batch)
    std_I_prime_batch = tf.sqrt(std_I_prime_batch)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    sum_all_mean_I_value = np.zeros(3)
    sum_all_mean_I_prime_value = np.zeros(3)
    sum_all_std_I_value = np.zeros(3)
    sum_all_std_I_prime_value = np.zeros(3)

    total_iters = int(len(img_np)/batch_size)
    for i in range(total_iters):
      print('%d / %d'%(i, total_iters))
      m_I, std_I, m_I_prime, std_I_prime = sess.run([mean_I_batch, std_I_batch, mean_I_prime_batch, std_I_prime_batch])
      sum_all_mean_I_value += m_I
      sum_all_std_I_value  += std_I
      sum_all_mean_I_prime_value += m_I_prime
      sum_all_std_I_prime_value  +=std_I_prime

    global_mean_I = sum_all_mean_I_value/total_iters
    global_std_I  = sum_all_std_I_value/total_iters
    global_mean_I_prime = sum_all_mean_I_prime_value/total_iters
    global_std_I_prime  = sum_all_std_I_prime_value/total_iters

    print('===> Final results=======')
    print('mean I', global_mean_I)
    print('std I', global_std_I)

    print('mean I prime', global_mean_I_prime)
    print('std I prime', global_std_I_prime)
    print('===> End =======')

  def norm_img(self, img, mean, std):
    return (img - mean)/std

  def denorm_img(self, img, mean, std):
    return img*std + mean


  def disjoint_augment_image_pair(self, img1, img2, min_val=0, max_val=255):
    # Randomly shift gamma
    random_gamma = tf.random_uniform([], 0.8, 1.2)
    img1_aug = img1**random_gamma
    random_gamma = tf.random_uniform([], 0.8, 1.2)
    img2_aug = img2**random_gamma

    # Randomly shift brightness
    random_brightness = tf.random_uniform([], 0.5, 2.0)
    img1_aug = img1_aug * random_brightness
    random_brightness = tf.random_uniform([], 0.5, 2.0)
    img2_aug = img2_aug * random_brightness

    # Randomly shift color
    random_colors = tf.random_uniform([3], 0.8, 1.2)
    white = tf.ones([tf.shape(img1)[0], tf.shape(img1)[1]])
    color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
    img1_aug  *= color_image

    random_colors = tf.random_uniform([3], 0.8, 1.2)
    white = tf.ones([tf.shape(img1)[0], tf.shape(img1)[1]])
    color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
    img2_aug  *= color_image

    # Saturate
    img1_aug  = tf.clip_by_value(img1_aug,  min_val, max_val)
    img2_aug  = tf.clip_by_value(img2_aug, min_val, max_val)

    return img1_aug, img2_aug

  def joint_augment_image_pair(self, img1, img2, min_val=0, max_val=255):
    # Randomly shift gamma
    random_gamma = tf.random_uniform([], 0.8, 1.2)
    img1_aug = img1**random_gamma
    img2_aug = img2**random_gamma

    # Randomly shift brightness
    random_brightness = tf.random_uniform([], 0.5, 2.0)
    img1_aug = img1_aug * random_brightness
    img2_aug = img2_aug * random_brightness

    # Randomly shift color
    random_colors = tf.random_uniform([3], 0.8, 1.2)
    white = tf.ones([tf.shape(img1)[0], tf.shape(img1)[1]])
    color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
    img1_aug  *= color_image
    img2_aug  *= color_image

    # Saturate
    img1_aug  = tf.clip_by_value(img1_aug,  min_val, max_val)
    img2_aug  = tf.clip_by_value(img2_aug, min_val, max_val)

    return img1_aug, img2_aug

if __name__=="__main__":
	test_synthetic_dataloader()
