from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pdb, os, shutil
import argparse
from matplotlib import pyplot as plt
import numpy as np
from dataloader import Dataloader, dataloader_params, extended_dataloader_params
from utils import utils
from utils import RANSAC_homography as ransac_h
from utils import direct_homography as direct_h
import math
from homography_model import HomographyModel, homography_model_params
import time, timeit
import cv2

# Size of synthetic image and the pertubation range (RH0)
HEIGHT = 142 #
WIDTH = 190
RHO = 24
PATCH_SIZE = 128

FULL_HEIGHT = 240 #
FULL_WIDTH =  320

# Real data directories
DATA_PATH = "/Earthbyte/tynguyen/docker_folder/pose_estimation/data/real/" + str(RHO) + '/'
DATA_PATH = "/home/tynguyen/pose_estimation/data/real/" + str(RHO) + '/'
if not os.path.exists(DATA_PATH):
  os.makedirs(DATA_PATH)

I_DIR = DATA_PATH + 'I/' # Large image
I_PRIME_DIR = DATA_PATH + 'I_prime/' # Large image
PTS1_FILE = os.path.join(DATA_PATH,'pts1.txt')
FILENAMES_FILE = os.path.join(DATA_PATH,'train_real.txt')
GROUND_TRUTH_FILE = None

FULL_I_DIR = DATA_PATH + 'FULL_I/' # Large image
FULL_I_PRIME_DIR = DATA_PATH + 'FULL_I_prime/' # Large image

TEST_PTS1_FILE = os.path.join(DATA_PATH,'test_pts1.txt')
TEST_FILENAMES_FILE = os.path.join(DATA_PATH,'test_real.txt')

# Correspondences in test set
TEST_GROUND_TRUTH_FILE = os.path.join(DATA_PATH,'test_gt.txt')

# Log and model directories
LOG_DIR = "/Earthbyte/tynguyen/docker_folder/pose_estimation/logs/real_logs"
SAVE_MODEL_DIR = "/Earthbyte/tynguyen/docker_folder/pose_estimation/models/real_models"
# Where to save visualization images (for report)
RESULTS_DIR = "/Earthbyte/tynguyen/docker_folder/pose_estimation/results/real/"


# list of augmentations to the data
AUGMENT_LIST = ['normalize']

def str2bool(s):
  return s.lower() == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='test', help='Train or test', choices=['train', 'test'])
parser.add_argument('--method', type=str, default='SIFT', help='methods', choices=['SIFT', 'ORB', 'identity', 'direct'])
parser.add_argument('--augment_list', nargs='+', default=AUGMENT_LIST, help='List of augmentations')
parser.add_argument('--do_augment',     type=float, default=0.5, help='Possibility of augmenting image: color shift, brightness shift...')
parser.add_argument('--num_gpus', type=int, default=2, help='Number of splits')

parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='The log path')
parser.add_argument('--results_dir', type=str, default=RESULTS_DIR, help='The dir to store images for report')

parser.add_argument('--data_path',     type=str, default=DATA_PATH, help='The raw data path.')
parser.add_argument('--I_dir',  			 type=str, default=I_DIR, help='The training image path')
parser.add_argument('--I_prime_dir',   type=str, default=I_PRIME_DIR, help='The training image path')
parser.add_argument('--full_I_dir',  	 type=str, default=FULL_I_DIR, help='The training image path')
parser.add_argument('--full_I_prime_dir',   type=str, default=FULL_I_PRIME_DIR, help='The training image path')
parser.add_argument('--pts1_file',      type=str, default=PTS1_FILE, help='The training path to 4 corners')
parser.add_argument('--test_pts1_file', type=str, default=TEST_PTS1_FILE, help='The test path to 4 corners')
parser.add_argument('--gt_file',       type=str, default=GROUND_TRUTH_FILE, help='The training ground truth file')
parser.add_argument('--test_gt_file',  type=str, default=TEST_GROUND_TRUTH_FILE, help='The test ground truth file')
parser.add_argument('--filenames_file',     type=str, default=FILENAMES_FILE, help='File that contains all names of files, for training')
parser.add_argument('--test_filenames_file',     type=str, default=TEST_FILENAMES_FILE, help='File that contains all names of files for evaluation')

parser.add_argument('--visual',   type=str2bool, default='false', help='Visualize obtained images to debug')
parser.add_argument('--save_visual',   type=str2bool, default='True', help='Save images for report')
parser.add_argument('--artifact_mode', type=str, default='None', help='Add aftifacts to the images', choices=['noise', 'None'])

parser.add_argument('--img_w',         type=int, default=WIDTH)
parser.add_argument('--img_h',         type=int, default=HEIGHT)
parser.add_argument('--full_img_w',    type=int, default=FULL_WIDTH)
parser.add_argument('--full_img_h',    type=int, default=FULL_HEIGHT)
parser.add_argument('--patch_size',    type=int, default=PATCH_SIZE)
parser.add_argument('--batch_size',    type=int, default=128)
parser.add_argument('--max_epoches',   type=int, default=1)
parser.add_argument('--num_features',  type=int, default=8, help='Number of features used to do RANSAC')
parser.add_argument('--num_iterations',  type=int, default=100, help='Number of iterations used to do direct method')

parser.add_argument('--resume', type=str2bool, default='False', help='True: restore the existing model. False: retrain')

print('<==================== Loading raw data ===================>\n')
args = parser.parse_args()
# Update log_dir
log_prefix_name = args.method + '/'
for augment_type in args.augment_list:
  log_prefix_name += '_' + augment_type

log_prefix_name += '/' + 'RHO_' + str(RHO) + '_maxfeat_'  + str(args.num_features) + '_doaugment_' + str(args.do_augment) + '/'

args.log_dir     = os.path.join(args.log_dir, log_prefix_name + args.mode + '/')
args.results_dir = os.path.join(args.results_dir, log_prefix_name + args.mode + '/')

print('Save mean, std values to ', args.log_dir)
print('Save images to', args.results_dir)

if not args.resume:
  try:
    shutil.rmtree(args.log_dir)
  except:
    pass

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
# num_train_data = utils.count_text_lines(args.filenames_file)
# print('===> There are totally %d train files'%(num_train_data))
print('===> There are totally %d test files'%(num_test_data))
test_batch_size=np.min([num_test_data, args.batch_size])

test_dataloader_params=extended_dataloader_params(data_path=args.data_path,
                                        filenames_file=args.test_filenames_file,
                                        pts1_file=args.test_pts1_file,
                                        gt_file=args.test_gt_file,
                                        mode='test',
                                        batch_size=test_batch_size,
                                        img_h=args.img_h,
                                        img_w=args.img_w,
                                        patch_size=args.patch_size,
                                        augment_list=args.augment_list,
                                        do_augment=args.do_augment,
                                        full_img_h=args.full_img_h,
                                        full_img_w=args.full_img_w)


def test():
  with tf.Graph().as_default(), tf.device('/gpu:0'): # Use GPU 0
    # Training parameters
    # Count the number of training & eval data
    num_data = utils.count_text_lines(args.test_filenames_file)
    print('===> Train: There are totally %d test files'%(num_data))

    steps_per_epoch = np.ceil(num_data/args.batch_size).astype(np.int32)

    num_total_steps = args.max_epoches*steps_per_epoch
    # Load data
    data_loader = Dataloader(test_dataloader_params, shuffle=False) # no shuffle
    # Debug test train_dataloader
    # test_synthetic_dataloader(data_loader, True)

    I1_batch      =  data_loader.I1_batch
    I2_batch      =  data_loader.I2_batch
    I1_aug_batch  =  data_loader.I1_aug_batch
    I2_aug_batch  =  data_loader.I2_aug_batch
    I_batch       =  data_loader.I_batch
    I_prime_batch = data_loader.I_prime_batch
    full_I_batch       =  data_loader.full_I_batch
    full_I_prime_batch = data_loader.full_I_prime_batch
    pts1_batch    = data_loader.pts1_batch
    gt_batch      = data_loader.gt_batch
    patch_indices_batch = data_loader.patch_indices_batch

    # Train on multiple GPU:
    h_losses = []
    # Create a session
    gpu_options = tf.GPUOptions(allow_growth=True) # Does not pre-allocate large, increase if needed
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options) # soft_placement allows to work on CPUs if GPUs are not available

    sess = tf.Session(config=config)

    # Initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # Threads coordinator
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    num_samples = 0
    total_num_fail = 0
    h_losses = []
    total_time = 0


    # Start test
    for step in range(num_total_steps):
      full_I_value, full_I_prime_value, I2_value, I1_value, I1_aug_value, I2_aug_value, I_value, I_prime_value, pts_1_value, full_gt_corr_value = sess.run([full_I_batch, full_I_prime_batch, I2_batch, I1_batch, I1_aug_batch, I2_aug_batch, I_batch, I_prime_batch, pts1_batch, gt_batch])
      for i in range(args.batch_size):
        num_samples += 1
        down_ratio = 2
        I_sample = utils.denorm_img(full_I_value[i]).astype(np.uint8)
        I_prime_sample = utils.denorm_img(full_I_prime_value[i]).astype(np.uint8)
        corr1_sample = full_gt_corr_value[i, 0:8].reshape([4,2])/down_ratio # (gt is for 480x640. Here, use 240x320)
        corr2_sample = full_gt_corr_value[i, 8:16].reshape([4,2])/down_ratio
        # Use RANSAC_homography/ Direct method to find the homography (delta 4 points)
        sample_start_time = timeit.default_timer()
        if args.method == 'direct':
          pred_h4p, _, not_found = direct_h.find_homography(I_sample, I_prime_sample, corr1_sample, corr2_sample, visual=args.visual, method=args.method, num_iterations=args.num_iterations, return_h_inv=False)
        # RANSAC Methods
        else:
          pred_h4p, _, not_found  = ransac_h.find_homography(I_sample, I_prime_sample, corr1_sample, corr2_sample, visual=args.visual, method=args.method, min_match_count=args.num_features, return_h_inv=False)
        sample_run_time = timeit.default_timer() - sample_start_time
        total_time += sample_run_time
        # Set maximum value for every value of delta h4p
        pred_h4p[np.where(pred_h4p >= 80)] = 80
        pred_h4p[np.where(pred_h4p <=- 80)] = -80

        pred_corr2_sample = pred_h4p[0] + corr1_sample
        h_loss_value = np.sqrt(np.mean(np.square(pred_corr2_sample - corr2_sample )))
        # Evaluate the result
        # There are two cases of failure
        if not_found:  # Cannot find homography
          total_num_fail += 1
          print('===> Fail case 1: Not found homography')

        else:
          # H_loss if homography is identity matrix
          h_loss_identity = np.sqrt(np.mean(np.square(corr1_sample - corr2_sample)))
          if h_loss_identity < h_loss_value:
            print('===> Fail case 2:  error > identity')
            total_num_fail += 1
            h_loss_value = h_loss_identity
        h_losses.append(h_loss_value)

        _ = utils.progress_bar(step*args.batch_size+i, num_total_steps*args.batch_size, ' Test| image %d, h_loss %.3f, h_loss_average %.3f, fail %d/%d, time %.4f'%(i, h_loss_value, np.mean(h_losses), total_num_fail, num_samples, sample_run_time))

        # Save visualization
        if args.save_visual:
          # Query full images
          img1_with_4pts  = I_sample.astype(np.uint8)
          img2_with_4pts  = I_prime_sample.astype(np.uint8)
          # Draw prediction
          cv2.polylines(img2_with_4pts, np.int32([pred_corr2_sample]), 1, (5, 225, 225),3)

          point_color = (0,255,255)
          line_color_set = [(255,102,255), (51,153,255), (102,255,255), (255,255,0), (102, 102, 244), (150, 202, 178), (153,240,142), (102,0,51), (51,51,0) ]
          # Draw 4 points (ground truth)
          full_stack_images = utils.draw_matches(img1_with_4pts, corr1_sample ,  img2_with_4pts , corr2_sample, 'tmp.jpg', color_set = line_color_set, show=False)

          # Save image
          visual_file_name = os.path.join(args.results_dir, str(step*args.batch_size + i) + '_loss_' + str(h_loss_value) + '.jpg')
          #cv2.putText(full_stack_images, 'RMSE %.2f'%h_loss,(800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
          cv2.imwrite(visual_file_name, full_stack_images)
          print('Wrote file %s', visual_file_name)


    result_dict = {'method': args.method,
                   'h_losses': h_losses,
                   'h_loss_mu': np.mean(h_losses),
                   'h_loss_std': np.std(h_losses)}
    import cPickle as pickle
    result_file_dir = os.path.join(args.results_dir, 'h_losses.pkl')
    with open(result_file_dir, 'wb') as f:
      pickle.dump(result_dict, f)
      print('===> Successfully write results  to  %s'%result_file_dir)
    print ('==========================================================')
    mean_h_loss, std_h_loss = np.mean(np.array(h_losses)), np.std(np.array(h_losses))
    print ('===> H_loss:', mean_h_loss, '+/-', std_h_loss)
    print ('Running time:', total_time/num_samples)
    fail_percent = total_num_fail*1.0/(num_samples)
    print ('Failure %.3f'%(fail_percent))
    output_line = [num_samples, mean_h_loss, std_h_loss, fail_percent, total_time/num_samples]
    print ('output_line:', output_line)
    with open(os.path.join(args.log_dir, 'results.txt'), 'w') as f:
      np.savetxt(f, [output_line], delimiter= ' ',  fmt='%.5f')
      print('===> Wrote results to file %s'%os.path.join(args.log_dir, 'results.txt'))
    tops_list = utils.find_percentile(h_losses)
    print('===> Percentile Values: (20, 50, 80, 100):')
    print(tops_list)
    print('======> End! ====================================')


if __name__=="__main__":
  test()
