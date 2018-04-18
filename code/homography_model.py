from __future__ import print_function, absolute_import, division
import tensorflow as tf
import numpy as np
from utils import utils
import tensorflow.contrib.slim as slim
from utils.utils import *
from utils.tf_spatial_transformer import transformer
import pdb
from tensorflow.contrib.layers.python.layers import initializers
from collections import namedtuple
from utils.utils import get_symetric_census, get_batch_symetric_census
homography_model_params = namedtuple('parameters',
                          'mode,'
                          'batch_size,'
                          'patch_size,'
                          'img_w,'
                          'img_h,'
                          'loss_type,'
                          'use_batch_norm,'
                          'augment_list,'
                          'leftright_consistent_weight,'
                          )

def tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)




class HomographyModel(object):
  def __init__(self, args, I1, I2, I1_aug, I2_aug, I_aug, I_prime_aug, h4p, gt, patch_indices, reuse_variables=None, model_index=0):
    self.params = args
    self.mode = args.mode
    self.is_training = True if self.mode=='train' else False
    self.I1 = I1
    self.I2 = I2
    self.I1_aug = I1_aug
    self.I2_aug = I2_aug
    # I and I_prime are augmented by default
    self.I = I_aug
    self.I_prime = I_prime_aug
    self.pts_1 = h4p
    self.gt = gt
    self.use_batch_norm=args.use_batch_norm
    self.patch_indices = patch_indices
    self.reuse_variables = reuse_variables
    self.model_collection = ['model_' + str(model_index)]
    # Constants and variables used for spatial transformer
    M = np.array([[self.params.img_w/2.0, 0., self.params.img_w/2.0],
                  [0., self.params.img_h/2.0, self.params.img_h/2.0],
                  [0., 0., 1.]]).astype(np.float32)

    M_tensor  = tf.constant(M, tf.float32)
    self.M_tile   = tf.tile(tf.expand_dims(M_tensor, [0]), [self.params.batch_size, 1,1])
    # Inverse of M
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    self.M_tile_inv   = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [self.params.batch_size,1,1])

    y_t = tf.range(0, self.params.batch_size*self.params.img_w*self.params.img_h, self.params.img_w*self.params.img_h)
    z =  tf.tile(tf.expand_dims(y_t,[1]),[1,self.params.patch_size*self.params.patch_size])
    self.batch_indices_tensor = tf.reshape(z, [-1]) # Add these value to patch_indices_batch[i] for i in range(num_pairs) # [BATCH_SIZE*WIDTH*HEIGHT]

    # Constant for ssim loss
    self.ssim_window =  tf_fspecial_gauss(size=3, sigma=0.5) # window shape [size, size]

    self.build_model()
    self.solve_DLT()
    self.transform()
    self.build_losses()
    self.build_summaries()


  def _conv2d(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu, scope=''):
    p = np.floor((kernel_size -1)/2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    out_conv =  slim.conv2d(inputs=p_x, num_outputs=num_out_layers, kernel_size=kernel_size, stride=stride, padding="VALID", activation_fn=activation_fn, scope=scope)
    if self.use_batch_norm:
      out_conv = slim.batch_norm(out_conv, self.is_training)
    return out_conv

  def _conv_block(self, x, num_out_layers, kernel_sizes, strides):
    conv1 = self._conv2d(x, num_out_layers[0], kernel_sizes[0], strides[0], scope='conv1')
    conv2 = self._conv2d(conv1, num_out_layers[1], kernel_sizes[1], strides[1], scope='conv2')

    return conv2

  def _maxpool2d(self, x, kernel_size, stride):
    p = np.floor((kernel_size -1)/2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.max_pool2d(p_x, kernel_size, stride=stride)

  def _vgg(self):
    with tf.variable_scope('conv_block1', reuse=self.reuse_variables): # H
      conv1 = self._conv_block(self.model_input, ([64, 64]), (3, 3), (1, 1))
      maxpool1 = self._maxpool2d(conv1, 2, 2) # H/2
    with tf.variable_scope('conv_block2', reuse=self.reuse_variables):
      conv2 = self._conv_block(maxpool1, ([64, 64]), (3, 3), (1, 1))
      maxpool2 = self._maxpool2d(conv2, 2, 2) # H/4
    with tf.variable_scope('conv_block3', reuse=self.reuse_variables):
      conv3 = self._conv_block(maxpool2, ([128, 128]), (3, 3), (1, 1))
      maxpool3 = self._maxpool2d(conv3, 2, 2) # H/8
    with tf.variable_scope('conv_block4', reuse=self.reuse_variables):
      conv4 = self._conv_block(maxpool3, ([128, 128]), (3, 3), (1, 1))
      # Dropout
      keep_prob = 0.5 if self.mode=='train' else 1.0
      dropout_conv4 = slim.dropout(conv4, keep_prob)

    # Flatten dropout_conv4
    out_conv_flat = slim.flatten(dropout_conv4)

    # Two fully-connected layers
    with tf.variable_scope('fc1'):
      fc1 = slim.fully_connected(out_conv_flat, 1024, scope='fc1')
      dropout_fc1 = slim.dropout(fc1, keep_prob)
    with tf.variable_scope('fc2'):
      fc2 = slim.fully_connected(dropout_fc1, 8, scope='fc2', activation_fn=None) #BATCH_SIZE x 8

    self.pred_h4p = fc2


  def _L1_smooth_loss(self, x, y):
    abs_diff = tf.abs(x-y)
    abs_diff_lt_1 = tf.less(abs_diff, 1)
    return tf.reduce_mean(tf.where(abs_diff_lt_1, 0.5*tf.square(abs_diff), abs_diff-0.5))

  def _SSIM_loss(self, x, y, size=3):
    # C = (K*L)^2 with K = max of intensity range (i.e. 255). L is very small
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, size, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, size, 1, 'VALID')

    sigma_x  = slim.avg_pool2d(x ** 2, size, 1, 'VALID') - mu_x ** 2
    sigma_y  = slim.avg_pool2d(y ** 2, size, 1, 'VALID') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y , size, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


  def _NCC_loss(self, x, y):
    """Consider x, y are vectors. Take L2 of the difference
       of the them after being normalized by their length"""
    len_x = tf.sqrt(tf.reduce_sum(tf.square(x)))
    len_y = tf.sqrt(tf.reduce_sum(tf.square(y)))
    return tf.sqrt(tf.reduce_sum(tf.square(x/len_x - y/len_y)))


  def solve_DLT(self):

    batch_size = self.params.batch_size
    pts_1_tile = self.pts_1_tile
    # Solve for H using DLT
    pred_h4p_tile = tf.expand_dims(self.pred_h4p, [2]) # BATCH_SIZE x 8 x 1
    # 4 points on the second image
    pred_pts_2_tile = tf.add(pred_h4p_tile, pts_1_tile)


    # Auxiliary tensors used to create Ax = b equation
    M1 = tf.constant(Aux_M1, tf.float32)
    M1_tensor = tf.expand_dims(M1, [0])
    M1_tile = tf.tile(M1_tensor,[batch_size,1,1])

    M2 = tf.constant(Aux_M2, tf.float32)
    M2_tensor = tf.expand_dims(M2, [0])
    M2_tile = tf.tile(M2_tensor,[batch_size,1,1])

    M3 = tf.constant(Aux_M3, tf.float32)
    M3_tensor = tf.expand_dims(M3, [0])
    M3_tile = tf.tile(M3_tensor,[batch_size,1,1])

    M4 = tf.constant(Aux_M4, tf.float32)
    M4_tensor = tf.expand_dims(M4, [0])
    M4_tile = tf.tile(M4_tensor,[batch_size,1,1])

    M5 = tf.constant(Aux_M5, tf.float32)
    M5_tensor = tf.expand_dims(M5, [0])
    M5_tile = tf.tile(M5_tensor,[batch_size,1,1])

    M6 = tf.constant(Aux_M6, tf.float32)
    M6_tensor = tf.expand_dims(M6, [0])
    M6_tile = tf.tile(M6_tensor,[batch_size,1,1])


    M71 = tf.constant(Aux_M71, tf.float32)
    M71_tensor = tf.expand_dims(M71, [0])
    M71_tile = tf.tile(M71_tensor,[batch_size,1,1])

    M72 = tf.constant(Aux_M72, tf.float32)
    M72_tensor = tf.expand_dims(M72, [0])
    M72_tile = tf.tile(M72_tensor,[batch_size,1,1])

    M8 = tf.constant(Aux_M8, tf.float32)
    M8_tensor = tf.expand_dims(M8, [0])
    M8_tile = tf.tile(M8_tensor,[batch_size,1,1])

    Mb = tf.constant(Aux_Mb, tf.float32)
    Mb_tensor = tf.expand_dims(Mb, [0])
    Mb_tile = tf.tile(Mb_tensor,[batch_size,1,1])

    # Form the equations Ax = b to compute H
    # Form A matrix
    A1 = tf.matmul(M1_tile, pts_1_tile) # Column 1
    A2 = tf.matmul(M2_tile, pts_1_tile) # Column 2
    A3 = M3_tile                   # Column 3
    A4 = tf.matmul(M4_tile, pts_1_tile) # Column 4
    A5 = tf.matmul(M5_tile, pts_1_tile) # Column 5
    A6 = M6_tile                   # Column 6
    A7 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M72_tile, pts_1_tile)# Column 7
    A8 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M8_tile, pts_1_tile)# Column 8

    A_mat = tf.transpose(tf.stack([tf.reshape(A1,[-1,8]),tf.reshape(A2,[-1,8]),\
                                   tf.reshape(A3,[-1,8]),tf.reshape(A4,[-1,8]),\
                                   tf.reshape(A5,[-1,8]),tf.reshape(A6,[-1,8]),\
         tf.reshape(A7,[-1,8]),tf.reshape(A8,[-1,8])],axis=1), perm=[0,2,1]) # BATCH_SIZE x 8 (A_i) x 8
    print('--Shape of A_mat:', A_mat.get_shape().as_list())
    # Form b matrix
    b_mat = tf.matmul(Mb_tile, pred_pts_2_tile)
    print('--shape of b:', b_mat.get_shape().as_list())

    # Solve the Ax = b
    H_8el = tf.matrix_solve(A_mat , b_mat)  # BATCH_SIZE x 8.
    print('--shape of H_8el', H_8el)


    # Add ones to the last cols to reconstruct H for computing reprojection error
    h_ones = tf.ones([batch_size, 1, 1])
    H_9el = tf.concat([H_8el,h_ones],1)
    H_flat = tf.reshape(H_9el, [-1,9])
    self.H_mat = tf.reshape(H_flat,[-1,3,3])   # BATCH_SIZE x 3 x 3

  def transform(self):
    # Transform H_mat since we scale image indices in transformer
    H_mat = tf.matmul(tf.matmul(self.M_tile_inv, self.H_mat), self.M_tile)
    # Transform image 1 (large image) to image 2
    out_size = (self.params.img_h, self.params.img_w)
    warped_images, _ = transformer(self.I, H_mat, out_size)
    # TODO: warp image 2 to image 1


    # Extract the warped patch from warped_images by flatting the whole batch before using indices
    # Note that input I  is  3 channels so we reduce to gray
    warped_gray_images = tf.reduce_mean(warped_images, 3)
    warped_images_flat = tf.reshape(warped_gray_images, [-1])
    patch_indices_flat = tf.reshape(self.patch_indices, [-1])
    pixel_indices =  patch_indices_flat + self.batch_indices_tensor
    pred_I2_flat = tf.gather(warped_images_flat, pixel_indices)

    self.pred_I2 = tf.reshape(pred_I2_flat, [self.params.batch_size, self.params.patch_size, self.params.patch_size, 1])

  def build_losses(self):
    I2 = self.I2_aug

    if self.params.mode == 'test':
      try:
        batch_h_loss = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(self.pred_h4p - self.gt), axis=1)))
        h_loss_identity = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(self.gt),axis=1)))
        is_failure = tf.cast(tf.greater_equal(batch_h_loss, h_loss_identity), tf.float32)
        self.num_fail = tf.reduce_sum(is_failure)
        # If it is a fail, use identity matrix
        self.bounded_h_loss = tf.reduce_mean(batch_h_loss*(1-is_failure) + is_failure*h_loss_identity)
      except:
        pass

    with tf.variable_scope('losses', reuse=self.reuse_variables):
      if self.params.loss_type=='h_loss':
        try:
          self.h_loss = tf.sqrt(tf.reduce_mean(tf.square(self.pred_h4p - self.gt)))
        except:
          pass
        self.rec_loss = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(self.pred_I2 - I2))))
        self.ssim_loss = tf.stop_gradient(self._SSIM_loss(self.pred_I2, I2))
        self.ssim_loss = tf.stop_gradient(tf.reduce_mean(self.ssim_loss))
        self.l1_loss = tf.stop_gradient(tf.reduce_mean(tf.abs(self.pred_I2 - I2)))
        self.l1_smooth_loss = tf.stop_gradient(self._L1_smooth_loss(self.pred_I2, I2))
        self.ncc_loss = tf.stop_gradient(self._NCC_loss(I2, self.pred_I2))

      elif self.params.loss_type=='rec_loss':
        try:
          self.h_loss = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(self.pred_h4p - self.gt))))
        except:
          pass
        self.rec_loss = tf.sqrt(tf.reduce_mean(tf.square(self.pred_I2 - I2)))
        self.ssim_loss = tf.stop_gradient(self._SSIM_loss(self.pred_I2, I2))
        self.ssim_loss = tf.stop_gradient(tf.reduce_mean(self.ssim_loss))
        self.l1_loss = tf.stop_gradient(tf.reduce_mean(tf.abs(self.pred_I2 - I2)))
        self.l1_smooth_loss = tf.stop_gradient(self._L1_smooth_loss(self.pred_I2, I2))
        self.ncc_loss = tf.stop_gradient(self._NCC_loss(I2, self.pred_I2))

      elif self.params.loss_type=='ssim_loss':
        try:
          self.h_loss = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(self.pred_h4p - self.gt))))
        except:
          pass
        self.rec_loss = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(self.pred_I2 - I2))))
        self.ssim_loss = tf.reduce_mean(self._SSIM_loss(self.pred_I2, I2))
        self.l1_loss = tf.stop_gradient(tf.reduce_mean(tf.abs(self.pred_I2 - I2)))
        self.l1_smooth_loss = tf.stop_gradient(self._L1_smooth_loss(self.pred_I2, I2))
        self.ncc_loss = tf.stop_gradient(self._NCC_loss(I2, self.pred_I2))

      elif self.params.loss_type=='l1_loss':
        try:
          self.h_loss = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(self.pred_h4p - self.gt))))
        except:
          pass
        self.rec_loss = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(self.pred_I2 - I2))))
        self.ssim_loss = tf.stop_gradient(tf.reduce_mean(self._SSIM_loss(self.pred_I2, I2)))
        self.l1_loss = tf.reduce_mean(tf.abs(self.pred_I2 - I2))
        self.l1_smooth_loss = tf.stop_gradient(self._L1_smooth_loss(self.pred_I2, I2))
        self.ncc_loss = tf.stop_gradient(self._NCC_loss(I2, self.pred_I2))

      elif self.params.loss_type=='l1_smooth_loss':
        try:
          self.h_loss = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(self.pred_h4p - self.gt))))
        except:
          pass
        self.rec_loss = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(self.pred_I2 - I2))))
        self.ssim_loss = tf.stop_gradient(tf.reduce_mean(self._SSIM_loss(self.pred_I2, I2)))
        self.l1_loss = tf.stop_gradient(tf.reduce_mean(tf.abs(self.pred_I2 - I2)))
        self.l1_smooth_loss =  self._L1_smooth_loss(self.pred_I2, I2)
        self.ncc_loss = tf.stop_gradient(self._NCC_loss(I2, self.pred_I2))

      elif self.params.loss_type=='ncc_loss':
        try:
          self.h_loss = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(self.pred_h4p - self.gt))))
        except:
          pass
        self.rec_loss = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(self.pred_I2 - I2))))
        self.ssim_loss = tf.stop_gradient(tf.reduce_mean(self._SSIM_loss(self.pred_I2, I2)))
        self.l1_loss = tf.stop_gradient(tf.reduce_mean(tf.abs(self.pred_I2 - I2)))
        self.l1_smooth_loss =  tf.stop_gradient(self._L1_smooth_loss(self.pred_I2, I2))
        self.ncc_loss = self._NCC_loss(I2, self.pred_I2)

  def build_model(self):
    # Declare types of activation function, weight_initialization of conv layer. We can set for each conv by setting locally later
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self.is_training), \
              slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
      with tf.variable_scope('model', reuse=self.reuse_variables):
        self.model_input = tf.concat([self.I1_aug, self.I2_aug], 3)
        self.pts_1_tile = tf.expand_dims(self.pts_1, [2]) # BATCH_SIZE x 8 x 1
        self._vgg()

  def build_summaries(self):
    # In the follows, only display 1 sample in each iteration
    with tf.device('/cpu:0'):
      with tf.name_scope('Input_larges'):
        tf.summary.image('I', self.I, 1) # Maximum batch is 1
        tf.summary.image('I_prime', self.I_prime, 1) # Maximum batch is 1
      with tf.name_scope('Input_patches'):
        tf.summary.image('I1_aug', self.I1_aug, 1) # Maximum batch is 1
        tf.summary.image('I2_aug', self.I2_aug, 1) # Maximum batch is 1
        tf.summary.image('I1', self.I1, 1) # Maximum batch is 1
        tf.summary.image('I2', self.I2, 1) # Maximum batch is 1

      with tf.name_scope('Outputs'):
        tf.summary.image('pred_I2', self.pred_I2, 1) # Maximum batch is 1




