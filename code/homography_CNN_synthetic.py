from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pdb, os, shutil
import argparse
from matplotlib import pyplot as plt
import numpy as np
from dataloader import Dataloader, dataloader_params
from utils import utils
import math
from homography_model import HomographyModel, homography_model_params
import time, timeit

# Size of synthetic image and the pertubation range (RH0)
HEIGHT = 240 #
WIDTH = 320
RHO = 45
PATCH_SIZE = 128

# Synthetic data directories
DATA_PATH = "/home/tynguyen/pose_estimation/data/synthetic/" + str(RHO) + '/'

if not os.path.exists(DATA_PATH):
  os.makedirs(DATA_PATH)

I_DIR = DATA_PATH + 'I/' # Large image
I_PRIME_DIR = DATA_PATH + 'I_prime/' # Large image
PTS1_FILE = os.path.join(DATA_PATH,'pts1.txt')
FILENAMES_FILE = os.path.join(DATA_PATH,'train_synthetic.txt')
GROUND_TRUTH_FILE = os.path.join(DATA_PATH,'gt.txt')

TEST_PTS1_FILE = os.path.join(DATA_PATH,'test_pts1.txt')
TEST_FILENAMES_FILE = os.path.join(DATA_PATH,'test_synthetic.txt')
TEST_GROUND_TRUTH_FILE = os.path.join(DATA_PATH,'test_gt.txt')

# Log and model directories
MAIN_LOG_PATH = '/media/tynguyen/'
LOG_DIR       = MAIN_LOG_PATH + "docker_folder/pose_estimation/logs/"
MODEL_DIR     = MAIN_LOG_PATH + "docker_folder/pose_estimation/models/"

# Where to save visualization images (for report)
RESULTS_DIR   = MAIN_LOG_PATH + "docker_folder/pose_estimation/results/synthetic/report/"

# List of augmentations to the data
AUGMENT_LIST = ['normalize']

def str2bool(s):
  return s.lower() == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='Train or test', choices=['train', 'test'])
parser.add_argument('--loss_type', type=str, default='l1_loss', help='Loss type', choices=['h_loss', 'rec_loss', 'ssim_loss', 'l1_loss', 'l1_smooth_loss', 'ncc_loss'])
parser.add_argument('--use_batch_norm', type=str2bool, default='True', help='Use batch_norm?')
parser.add_argument('--leftright_consistent_weight', type=float, default=0, help='UUse left right consistent in loss function? Set a small weight for loss(I2_to_I1 - I1)')
parser.add_argument('--augment_list', nargs='+', default=AUGMENT_LIST, help='List of augmentations')
parser.add_argument('--do_augment',     type=float, default=0.5, help='Possibility of augmenting image: color shift, brightness shift...')
parser.add_argument('--num_gpus', type=int, default=2, help='Number of splits')

parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='The log path')
parser.add_argument('--results_dir', type=str, default=RESULTS_DIR, help='Store visualization for report')
parser.add_argument('--model_dir', type=str, default=MODEL_DIR, help='The models path')
parser.add_argument('--model_name', type=str, default='model.ckpt', help='The model name')

parser.add_argument('--data_path',     type=str, default=DATA_PATH, help='The raw data path.')
parser.add_argument('--I_dir',  			 type=str, default=I_DIR, help='The training image path')
parser.add_argument('--I_prime_dir',   type=str, default=I_PRIME_DIR, help='The training image path')
parser.add_argument('--pts1_file',      type=str, default=PTS1_FILE, help='The training path to 4 corners on the first image - training dataset')
parser.add_argument('--test_pts1_file', type=str, default=TEST_PTS1_FILE, help='The test path to 4 corners on the first image - test dataset')
parser.add_argument('--gt_file',       type=str, default=GROUND_TRUTH_FILE, help='The training ground truth file')
parser.add_argument('--test_gt_file',  type=str, default=TEST_GROUND_TRUTH_FILE, help='The test ground truth file')
parser.add_argument('--filenames_file',     type=str, default=FILENAMES_FILE, help='File that contains all names of files, for training')
parser.add_argument('--test_filenames_file',     type=str, default=TEST_FILENAMES_FILE, help='File that contains all names of files for evaluation')

parser.add_argument('--visual',   type=str2bool, default='false', help='Visualize obtained images to debug')
parser.add_argument('--save_visual',   type=str2bool, default='True', help='Save visual images for report')

parser.add_argument('--img_w',         type=int, default=WIDTH)
parser.add_argument('--img_h',         type=int, default=HEIGHT)
parser.add_argument('--patch_size',    type=int, default=PATCH_SIZE)
parser.add_argument('--batch_size',    type=int, default=128)
parser.add_argument('--max_epoches',   type=int, default=150)
parser.add_argument('--lr',            type=float, default=1e-4, help='Max learning rate')
parser.add_argument('--min_lr',        type=float, default=.9e-4, help='Min learning rate')

parser.add_argument('--resume', type=str2bool, default='False', help='True: restore the existing model. False: retrain')
parser.add_argument('--retrain', type=str2bool, default='False', help='True: restore the existing model, use max learning rate')

print('<==================== Loading raw data ===================>\n')
args = parser.parse_args()
# Update model_dir
model_prefix_name = args.loss_type
for augment_type in args.augment_list:
  model_prefix_name += '_' + augment_type

if args.mode=='test':
  args.log_dir = os.path.join(args.log_dir, model_prefix_name + 'test/')

args.model_dir = os.path.join(args.model_dir, model_prefix_name)
args.log_dir = os.path.join(args.log_dir, model_prefix_name)
if args.mode=='test':
  args.log_dir = os.path.join(args.log_dir, model_prefix_name + 'test/')

if not args.resume:
  try:
    shutil.rmtree(args.log_dir)
  except:
    pass

if not os.path.exists(args.model_dir):
  os.makedirs(args.model_dir)
if not os.path.exists(args.log_dir):
  os.makedirs(args.log_dir)

if not os.path.exists(args.results_dir):
  os.makedirs(args.results_dir)

# Visualize online
if args.visual:
  plt.ion()

train_dataloader_params=dataloader_params(data_path=args.data_path,
                                        filenames_file=args.filenames_file,
                                        pts1_file=args.pts1_file,
                                        gt_file=args.gt_file,
                                        mode=args.mode,
                                        batch_size=args.batch_size,
                                        img_h=args.img_h,
                                        img_w=args.img_w,
                                        patch_size=args.patch_size,
                                        augment_list=args.augment_list,
                                        do_augment=args.do_augment)

num_test_data = utils.count_text_lines(args.test_filenames_file)

print('===> There are totally %d test files'%(num_test_data))

test_batch_size=np.min([num_test_data, args.batch_size])

test_dataloader_params=dataloader_params(data_path=args.data_path,
                                        filenames_file=args.test_filenames_file,
                                        pts1_file=args.test_pts1_file,
                                        gt_file=args.test_gt_file,
                                        mode='test',
                                        batch_size=test_batch_size,
                                        img_h=args.img_h,
                                        img_w=args.img_w,
                                        patch_size=args.patch_size,
                                        augment_list=args.augment_list,
                                        do_augment=args.do_augment)


def train():
  # Overrides the current default graph for the lifetime of the context
  with tf.Graph().as_default(), tf.device('/gpu:0'): # Use GPU 0
    global_step = tf.Variable(0, trainable=False)

    # Training parameters
    # Count the number of training & eval data
    num_data = utils.count_text_lines(args.filenames_file)
    print('===> Train: There are totally %d training files'%(num_data))

    num_total_steps = 150000

    # Optimizer. Use exponential decay: decayed_lr = lr* decay_rate^ (global_steps/ decay_steps)
    decay_rate = 0.96

    decay_steps = (math.log(decay_rate) * num_total_steps)/math.log(args.min_lr*1.0/args.lr)
    print('args lr:', args.lr, args.min_lr)
    print('===> Decay steps:', decay_steps)
    learning_rate = tf.train.exponential_decay(args.lr, global_step, int(decay_steps), decay_rate, staircase=True)

    # Due to slim.batch_norm docs:
    # Note: when training, the moving_mean and moving_variance need to be updated.
    # By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
    # need to be added as a dependency to the `train_op`. For example:

    # ```python
    #   update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #   with tf.control_dependencies(update_ops):
    #     train_op = optimizer.minimize(loss)
    # ```
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      opt_step = tf.train.AdamOptimizer(learning_rate)

    # Load data
    data_loader = Dataloader(train_dataloader_params, shuffle=True) # shuffle
    # Debug test train_dataloader
    # test_synthetic_dataloader(data_loader, True)

    I1_batch =  data_loader.I1_batch
    I2_batch =  data_loader.I2_batch
    I1_aug_batch =  data_loader.I1_aug_batch
    I2_aug_batch =  data_loader.I2_aug_batch
    I_batch  =  data_loader.I_batch
    I_prime_batch = data_loader.I_prime_batch
    pts1_batch     = data_loader.pts1_batch
    gt_batch      = data_loader.gt_batch
    patch_indices_batch = data_loader.patch_indices_batch

    # Split on multiple GPU
    I1_splits = tf.split(I1_batch, args.num_gpus, 0)
    I2_splits = tf.split(I2_batch, args.num_gpus, 0)
    I1_aug_splits = tf.split(I1_aug_batch, args.num_gpus, 0)
    I2_aug_splits = tf.split(I2_aug_batch, args.num_gpus, 0)
    I_splits  = tf.split(I_batch, args.num_gpus, 0 )
    I_prime_splits = tf.split(I_prime_batch, args.num_gpus, 0)
    pts1_splits     = tf.split(pts1_batch, args.num_gpus, 0)
    gt_splits      = tf.split(gt_batch, args.num_gpus, 0 )
    patch_indices_splits = tf.split(patch_indices_batch, args.num_gpus, 0 )

    # Train on multiple GPU:
    multi_grads = []
    reuse_variables = None
    h_losses = []
    rec_losses = []
    ssim_losses = []
    l1_losses = []
    l1_smooth_losses = []
    ncc_losses = []
    model_params = homography_model_params(mode=args.mode,
                               batch_size=int(args.batch_size/args.num_gpus),
                               patch_size=args.patch_size,
                               img_h=args.img_h,
                               img_w=args.img_w,
                               loss_type=args.loss_type,
                               use_batch_norm=args.use_batch_norm,
                               augment_list=args.augment_list,
                               leftright_consistent_weight=args.leftright_consistent_weight
                               )
    # Deal with sharable variables
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(args.num_gpus):
        with tf.device('/gpu:%d'%i):
          model = HomographyModel(model_params, I1_splits[i], I2_splits[i], I1_aug_splits[i], I2_aug_splits[i], I_splits[i], I_prime_splits[i],
                                  pts1_splits[i], gt_splits[i], patch_indices_splits[i], reuse_variables=reuse_variables, model_index=i)
          # Debug test splits
          #test_synthetic_dataloader(data_loader, True, I1_splits[i], I2_splits[i], I_splits[i], I_prime_splits[i], pts1_splits[i], gt_splits[i], patch_indices_splits[i])

          h_loss = model.h_loss
          rec_loss = model.rec_loss
          ssim_loss = model.ssim_loss
          l1_loss = model.l1_loss
          l1_smooth_loss = model.l1_smooth_loss
          ncc_loss = model.ncc_loss

          pred_I2 = model.pred_I2
          I2 = model.I2
          H_mat = model.H_mat
          I1 = model.I1
          I = model.I

          I1_aug = model.I1_aug
          I2_aug = model.I2_aug

          h_losses.append(h_loss)
          rec_losses.append(rec_loss)
          ssim_losses.append(ssim_loss)
          l1_losses.append(l1_loss)
          l1_smooth_losses.append(l1_smooth_loss)
          ncc_losses.append(ncc_loss)

          reuse_variables = True
          if args.loss_type=='h_loss':
            grads = opt_step.compute_gradients(h_loss)
          elif args.loss_type=='rec_loss':
            grads = opt_step.compute_gradients(rec_loss)
          elif args.loss_type=='ssim_loss':
            grads = opt_step.compute_gradients(ssim_loss)
          elif args.loss_type=='l1_loss':
            grads = opt_step.compute_gradients(l1_loss)
          elif args.loss_type=='l1_smooth_loss':
            grads = opt_step.compute_gradients(l1_smooth_loss)
          elif args.loss_type=='ncc_loss':
            grads = opt_step.compute_gradients(ncc_loss)
          else:
            print('===> Loss type does not exist!')
            exit(0)
          print('====> Use loss type: ', args.loss_type)
          time.sleep(2)
          multi_grads.append(grads)
    # Take average of the grads
    grads = utils.get_average_grads(multi_grads)
    apply_grad_opt = opt_step.apply_gradients(grads, global_step=global_step)
    total_h_loss = tf.reduce_mean(h_losses)
    total_rec_loss = tf.reduce_mean(rec_losses)
    total_ssim_loss = tf.reduce_mean(ssim_losses)
    total_l1_loss = tf.reduce_mean(l1_losses)
    total_l1_smooth_loss = tf.reduce_mean(l1_smooth_losses)
    total_ncc_loss = tf.reduce_mean(ncc_losses)
    with tf.name_scope('Losses'):
      tf.summary.scalar('Learning_rate', learning_rate)
      tf.summary.scalar('Total_h_loss', total_h_loss)
      tf.summary.scalar('Total_rec_loss', total_rec_loss)
      tf.summary.scalar('Total_ssim_loss', total_ssim_loss)
      tf.summary.scalar('Total_l1_loss', total_l1_loss)
      tf.summary.scalar('Total_l1_smooth_loss', total_l1_smooth_loss)
      tf.summary.scalar('Total_ncc_loss', total_ncc_loss)
    summary_opt = tf.summary.merge_all()
    # Create a session
    gpu_options = tf.GPUOptions(allow_growth=True) # Does not pre-allocate large, increase if needed
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options) # soft_placement allows to work on CPUs if GPUs are not available

    sess = tf.Session(config=config)

    # Saver
    log_name = args.loss_type
    summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
    train_saver = tf.train.Saver(max_to_keep=5) # Keep maximum 5 models

    # Initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # Threads coordinator
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)


    # Restore
    if args.resume:
      train_saver.restore(sess,tf.train.latest_checkpoint(args.model_dir))
      if args.retrain:
        sess.run(global_step.assign(0))

    # Index of the image want to display
    index = 0
    h_total_loss_value = 0
    rec_total_loss_value = 0
    ssim_total_loss_value = 0
    l1_total_loss_value = 0
    l1_smooth_total_loss_value = 0
    ncc_total_loss_value = 0


    start_step = global_step.eval(session=sess)
    print('===> Start step:', start_step)

    # Start training
    for step in range(start_step, start_step + num_total_steps):
      if args.visual:
        _, h_loss_value,  rec_loss_value, ssim_loss_value, l1_loss_value, l1_smooth_loss_value, ncc_loss_value, lr_value, H_mat_value, pred_I2_value, I2_value, I1_value, I1_aug_value, I2_aug_value, I_value = sess.run([apply_grad_opt, total_h_loss, total_rec_loss, total_ssim_loss, total_l1_loss, total_l1_smooth_loss, total_ncc_loss, learning_rate, H_mat, pred_I2, I2, I1, I1_aug, I2_aug, I])
      elif args.loss_type=="l1_loss" and not args.visual:
        _, h_loss_value, l1_loss_value, l1_smooth_loss_value, lr_value = sess.run([apply_grad_opt, total_h_loss, total_l1_loss, total_l1_smooth_loss, learning_rate])
        h_total_loss_value += h_loss_value
        l1_total_loss_value += l1_loss_value
        l1_smooth_total_loss_value += l1_smooth_loss_value
        if step % 100 == 0:
          total_time = utils.progress_bar(step, num_total_steps + start_step - 1, 'Train: 1, h_loss %4.3f, l1_loss %.6f, l1_smooth_loss %.6f, lr %.6f'%(h_total_loss_value/(step -start_step +1), l1_total_loss_value/(step-start_step+1), l1_smooth_total_loss_value/(step-start_step +1), lr_value))

      else:
        _, h_loss_value,  rec_loss_value, ssim_loss_value, l1_loss_value, l1_smooth_loss_value, ncc_loss_value, lr_value = sess.run([apply_grad_opt, total_h_loss, total_rec_loss, total_ssim_loss, total_l1_loss, total_l1_smooth_loss, total_ncc_loss, learning_rate])
        h_total_loss_value += h_loss_value
        rec_total_loss_value += rec_loss_value
        ssim_total_loss_value += ssim_loss_value
        l1_total_loss_value += l1_loss_value
        l1_smooth_total_loss_value += l1_smooth_loss_value
        ncc_total_loss_value += ncc_loss_value
        if step % 100 == 0:
          total_time = utils.progress_bar(step, num_total_steps + start_step - 1, 'Train: 1, h_loss %4.3f, rec_loss %4.3f, ssim_loss %.6f. l1_loss %.6f, l1_smooth_loss %.6f, ncc_loss %.6f, lr %.6f'%(h_total_loss_value/(step -start_step +1), rec_total_loss_value/(step - start_step +1), ssim_total_loss_value/(step-start_step+1), l1_total_loss_value/(step-start_step+1), l1_smooth_total_loss_value/(step-start_step +1), ncc_total_loss_value/(step-start_step+1), lr_value))

      # Tensorboard
      if step % 1000 == 0:
        summary_str = sess.run(summary_opt)
        summary_writer.add_summary(summary_str, global_step=step)
      if step  and step % 1000 == 0:
        train_saver.save(sess, args.model_dir + args.model_name, global_step=step)

      if args.visual and step%1==0:
        if 'normalize' in args.augment_list:
          pred_I2_sample_value = utils.denorm_img(pred_I2_value[index,:,:,0]).astype(np.uint8)
          I2_sample_value = utils.denorm_img(I2_value[index, :, :, 0]).astype(np.uint8)
          I1_sample_value = utils.denorm_img(I1_value[index, :, :, 0]).astype(np.uint8)
          I1_aug_sample_value = utils.denorm_img(I1_aug_value[index, :, :, 0]).astype(np.uint8)
          I2_aug_sample_value = utils.denorm_img(I2_aug_value[index, :, :, 0]).astype(np.uint8)
          I_sample_value  = utils.denorm_img(I_value[index,...]).astype(np.uint8)
        else:
          pred_I2_sample_value = pred_I2_value[index,:,:,0].astype(np.uint8)
          I2_sample_value = I2_value[index, :, :, 0].astype(np.uint8)
          I1_sample_value = I1_value[index, :, :, 0].astype(np.uint8)
          I1_aug_sample_value = I1_aug_value[index, :, :, 0].astype(np.uint8)
          I2_aug_sample_value = I2_aug_value[index, :, :, 0].astype(np.uint8)
          I_sample_value  = I_value[index, ...].astype(np.uint8)
        plt.subplot(3,1,1)
        plt.imshow(np.concatenate([pred_I2_sample_value, I2_sample_value], 1), cmap='gray')
        plt.title('Pred I2 vs I2')
        plt.subplot(3,1,2)
        plt.imshow(np.concatenate([I1_aug_sample_value, I2_aug_sample_value], 1), cmap='gray')
        plt.title('I1_aug vs I2_aug')
        plt.subplot(3,1,3)
        plt.imshow(I_sample_value if I_sample_value.shape[2]== 3 else I_sample_value[:,:,0])
        plt.title('I')
        plt.show()
        plt.pause(0.05)
    # Save the final model
    train_saver.save(sess, args.model_dir + args.model_name, global_step=step)

class TestHomography(object):
  def __init__(self):
    # Overrides the current default graph for the lifetime of the context
    with tf.device('/gpu:0'): # Use GPU 0

      # Count the number of eval data
      num_data = utils.count_text_lines(args.test_filenames_file)
      print('===> Test: There are totally %d Test files'%(num_data))

      steps_per_epoch = np.ceil(num_data/args.batch_size).astype(np.int32)
      self.num_total_steps = 3*steps_per_epoch # Test 3 epoches

      # Load data
      data_loader = Dataloader(test_dataloader_params, shuffle=True) # No shuffle

      I1_batch =  data_loader.I1_batch
      I2_batch =  data_loader.I2_batch
      I1_aug_batch =  data_loader.I1_aug_batch
      I2_aug_batch =  data_loader.I2_aug_batch
      I_batch  =  data_loader.I_batch
      I_prime_batch = data_loader.I_prime_batch
      pts1_batch     = data_loader.pts1_batch
      gt_batch      = data_loader.gt_batch
      patch_indices_batch = data_loader.patch_indices_batch

      # Split on multiple GPU
      I1_splits = tf.split(I1_batch, args.num_gpus, 0)
      I2_splits = tf.split(I2_batch, args.num_gpus, 0)
      I1_aug_splits = tf.split(I1_aug_batch, args.num_gpus, 0)
      I2_aug_splits = tf.split(I2_aug_batch, args.num_gpus, 0)
      I_splits  = tf.split(I_batch, args.num_gpus, 0 )
      I_prime_splits = tf.split(I_prime_batch, args.num_gpus, 0)
      pts1_splits     = tf.split(pts1_batch, args.num_gpus, 0)
      gt_splits      = tf.split(gt_batch, args.num_gpus, 0 )
      patch_indices_splits = tf.split(patch_indices_batch, args.num_gpus, 0 )

      # Train on multiple GPU:
      reuse_variables = None
      h_losses = []
      rec_losses = []
      ssim_losses = []
      l1_losses = []
      l1_smooth_losses = []
      num_fails = []
      model_params = homography_model_params(mode='test',
                                 batch_size=int(args.batch_size/args.num_gpus),
                                 patch_size=args.patch_size,
                                 img_h=args.img_h,
                                 img_w=args.img_w,
                                 loss_type=args.loss_type,
                                 use_batch_norm=args.use_batch_norm,
                                 augment_list=args.augment_list,
                                 leftright_consistent_weight=args.leftright_consistent_weight)
      # Deal with sharable variables
      with tf.variable_scope(tf.get_variable_scope()):
        for i in range(args.num_gpus):
          with tf.device('/gpu:%d'%i):
            model = HomographyModel(model_params, I1_splits[i], I2_splits[i], I1_aug_splits[i], I2_aug_splits[i], I_splits[i], I_prime_splits[i],
                                    pts1_splits[i], gt_splits[i], patch_indices_splits[i], reuse_variables=reuse_variables, model_index=i)
            # Debug test splits
            #test_synthetic_dataloader(data_loader, True, I1_splits[i], I2_splits[i], I_splits[i], I_prime_splits[i], pts1_splits[i], gt_splits[i], patch_indices_splits[i])

            reuse_variables = True
            # In testing, use bounded_h_loss (under successful condition)
            h_loss = model.bounded_h_loss
            rec_loss = model.rec_loss
            ssim_loss = model.ssim_loss
            l1_loss = model.l1_loss
            l1_smooth_loss = model.l1_smooth_loss
            num_fail = model.num_fail

            self.pred_I2 = model.pred_I2
            self.I2 = model.I2
            self.H_mat = model.H_mat
            self.I1 = model.I1
            self.I1_aug = model.I1_aug
            self.I2_aug = model.I2_aug
            self.I  = model.I
            self.I_prime  = model.I_prime
            self.pts1   = model.pts_1
            self.gt     = model.gt
            self.pred_h4p = model.pred_h4p


            h_losses.append(h_loss)
            rec_losses.append(rec_loss)
            ssim_losses.append(ssim_loss)
            l1_losses.append(l1_loss)
            l1_smooth_losses.append(l1_smooth_loss)
            num_fails.append(num_fail)
      self.total_h_loss = tf.reduce_mean(h_losses)
      self.total_num_fail = tf.reduce_sum(num_fails)
      self.total_rec_loss = tf.reduce_mean(rec_losses)
      self.total_ssim_loss = tf.reduce_mean(ssim_losses)
      self.total_l1_loss = tf.reduce_mean(l1_losses)
      self.total_l1_smooth_loss = tf.reduce_mean(l1_smooth_losses)
      with tf.name_scope('Losses'):
        tf.summary.scalar('Total_h_loss', self.total_h_loss)
        tf.summary.scalar('Total_rec_loss', self.total_rec_loss)
        tf.summary.scalar('Total_ssim_loss', self.total_ssim_loss)
        tf.summary.scalar('Total_l1_loss', self.total_l1_loss)
        tf.summary.scalar('Total_l1_smooth_loss', self.total_l1_smooth_loss)
      self.summary_opt = tf.summary.merge_all()

  def run_test(self, model_index=0):
    # Create a session
    gpu_options = tf.GPUOptions(allow_growth=True) # Does not pre-allocate large, increase if needed
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options) # soft_placement allows to work on CPUs if GPUs are not available

    sess = tf.Session(config=config)

    # Saver
    log_name = args.loss_type
    summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
    train_saver = tf.train.Saver(max_to_keep=20) # Keep maximum 20 models

    # Initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # Threads coordinator
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)


    # Restore
    print(args.model_dir)
    train_saver.restore(sess,tf.train.latest_checkpoint(args.model_dir))
    # Index of the image want to display
    index = 0
    h_total_loss_value = 0
    rec_total_loss_value = 0
    ssim_total_loss_value = 0
    l1_total_loss_value = 0
    total_num_fail_value = 0
    # Start testing
    h_losses_array = []
    total_num_fail = 0

    for step in range(self.num_total_steps):
      num_fail_value, h_loss_value,  rec_loss_value, ssim_loss_value, l1_loss_value, l1_smooth_loss_value,  pred_I2_value, I1_aug_value, I2_aug_value, I_value, I_prime_value, pts1_value, gt_value, pred_h4p_value  = sess.run([self.total_num_fail, self.total_h_loss, self.total_rec_loss, self.total_ssim_loss, self.total_l1_loss, self.total_l1_smooth_loss, self.pred_I2, self.I1_aug, self.I2_aug, self.I, self.I_prime, self.pts1, self.gt, self.pred_h4p])

      h_total_loss_value += h_loss_value
      rec_total_loss_value += rec_loss_value
      ssim_total_loss_value += ssim_loss_value
      l1_total_loss_value += l1_loss_value
      total_num_fail_value += num_fail_value
      h_losses_array.append(h_loss_value)

      if args.save_visual:
        I_sample = utils.denorm_img(I_value[0]).astype(np.uint8)
        I_prime_sample = utils.denorm_img(I_prime_value[0]).astype(np.uint8)
        pts1_sample = pts1_value[0].reshape([4,2]).astype(np.float32)
        gt_h4p_sample = gt_value[0].reshape([4,2]).astype(np.float32)

        pts2_sample  = pts1_sample + gt_h4p_sample

        pred_h4p_sample = pred_h4p_value[0].reshape([4,2]).astype(np.float32)
        pred_pts2_sample = pts1_sample + pred_h4p_sample

        # Save
        visual_file_name = str(step*args.batch_size) + '_' + args.loss_type + '_loss_' + str(h_loss_value) + '.jpg'
        utils.save_correspondences_img(I_prime_sample, I_sample, pts1_sample, pts2_sample, pred_pts2_sample, args.results_dir, visual_file_name)


      if step % 10 == 0:
        print('===> This iteration num Fail: %d \n'%num_fail_value)
        total_time = utils.progress_bar(step, self.num_total_steps, 'Test, h_loss %4.3f, rec_loss %4.3f, ssim_loss %4.3f, l1_loss %4.3f, fail_percent %4.4f'%(h_total_loss_value/(step+1), rec_total_loss_value/(step+1), ssim_total_loss_value/(step+1), l1_total_loss_value/(step+1), total_num_fail_value/(step+1)/args.batch_size))

      if args.visual and step%10==0:
        plt.subplot(2,1,1)
        plt.imshow(np.concatenate([pred_I2_value[index,:,:,0].astype(np.uint8), I2_aug_value[index,:,:,0]], 1), cmap='gray')
        plt.title('Pred I2 vs I2')
        plt.subplot(2,1,2)
        plt.imshow(I1_aug_value[index,:,:,0], I2_aug_value[index,:,:,0], cmap='gray')
        plt.title('I1_aug vs I2_aug')
        plt.show()
        plt.pause(0.05)

    # Final result
    total_time = utils.progress_bar(step, self.num_total_steps, 'Test, h_loss %4.3f, rec_loss %4.3f, ssim_loss %4.3f, l1_loss %4.3f, fail_percent %4.4f'%(h_total_loss_value/(step+1), rec_total_loss_value/(step+1), ssim_total_loss_value/(step+1), l1_total_loss_value/(step+1),total_num_fail_value/(step+1)/args.batch_size))

    # Summarize results
    print('====> Result for RHO:', RHO,' loss ', args.loss_type, ' noise ', args.do_augment)
    print('|Steps  |   h_loss   |    l1_loss   |  Fail percent    |')
    print (step, h_total_loss_value/(step+1), l1_total_loss_value/(step+1), 100*total_num_fail_value/(step+1)/args.batch_size)

    tops_list = utils.find_percentile(h_losses_array)
    print('===> Percentile Values: (20, 50, 80, 100):')
    print(tops_list)
    print('======> End! ====================================')

def test_homography():
  test_obj = TestHomography()
  test_obj.run_test(0)
  exit()

def test_get_mean_std():
  train_data_loader = Dataloader(train_dataloader_params, shuffle=False)
  train_data_loader.get_mean_std()
  exit()


if __name__=="__main__":
  if args.mode=='train':
    train()
  else:
    test_homography()

