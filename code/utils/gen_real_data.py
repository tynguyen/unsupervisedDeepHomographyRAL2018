# gen_real_data.py
# Generate real image dataset
import os, pdb, shutil, argparse, glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from utils import progress_bar, removeHiddenfile, draw_matches
from utils import show_image, find_nine_grid_points
from numpy_spatial_transformer import numpy_transformer
from matplotlib import pyplot as plt
import time
import scipy.io as io

def query_gt_test_set():
  label_path = '/Earthbyte/tynguyen/real_rawdata/joe_data/test/labels/'
  mat_file_name_list = [label_path+'corresponences0_10.mat',
                        label_path+'correspondences11_21.mat',
                        label_path+'correspondences22_30.mat',
                        label_path+'correspondences31_40.mat',
                        label_path+'correspondences41_49.mat']
  for i in range(len(mat_file_name_list)):
    gt_array = io.loadmat(mat_file_name_list[i])
    corr1_array = gt_array['all_corr1']
    corr2_array = gt_array['all_corr2']
    if i == 0:
      complete_corr_array1 = corr1_array
      complete_corr_array2 = corr2_array
    else:
      complete_corr_array1 = np.concatenate((complete_corr_array1, corr1_array), axis=0)
      complete_corr_array2 = np.concatenate((complete_corr_array2, corr2_array), axis=0)
  # Return 200x2, 200x2 arrays.
  # To query 4 points on the first image, do:
  # complete_corr_array1[image_index*4:(image_index + 1)*4] => 4x2
  return complete_corr_array1, complete_corr_array2

def homographyGeneration(args, raw_image_path, index):

  rho = args.rho
  patch_size = args.patch_size
  height = args.img_h
  width =  args.img_w
  full_height = args.full_img_h
  full_width =  args.full_img_w


  # Text files to store numbers
  if args.mode=='train' and not args.debug:
    f_pts1 = open(args.pts1_file, 'wb')
    f_file_list = open(args.filenames_file, 'wb')
  elif not args.debug:
    f_pts1 = open(args.test_pts1_file, 'wb')
    f_file_list = open(args.test_filenames_file, 'wb')
    f_test_gt = open(args.test_gt_file, 'wb')

  # Query correspondences in test set
  if args.mode == 'test':
    corr1_array, corr2_array = query_gt_test_set()

  image_files = glob.glob(os.path.join(raw_image_path, '*.JPG'))
  image_files.sort()

  for num_files in range(len(image_files)-1):
    I_file = image_files[num_files]
    I_prime_file = image_files[num_files+1]

    I_img_id = int(I_file[len(I_file)- 8: len(I_file) - 4])
    print('===> Image ', I_file , 'vs', I_prime_file)

    if I_img_id in args.ignore_list:
      print('====> Ignore', I_img_id)
      continue
    else:
      print('====> Accept', I_img_id)


    I = cv2.imread(image_files[num_files])
    I_prime = cv2.imread(image_files[num_files+1])
    # Full images (size args.full_img_h x args.full_img_w): used for conventional algorithms
    full_I = cv2.resize(I, (full_width, full_height))
    full_I_prime = cv2.resize(I_prime, (full_width, full_height))

    # Large images (size args.img_h x args.img_w): used for deep learning algorithms
    I = cv2.resize(I, (width, height))
    I_prime = cv2.resize(I_prime, (width, height))

    I_gray  = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    I_prime_gray  = cv2.cvtColor(I_prime, cv2.COLOR_RGB2GRAY)

    # Ground truth of correspondences in test set
    if args.mode == 'test':
      corr1 = corr1_array[num_files*4:num_files*4+4].reshape([4,2])
      corr2 = corr2_array[num_files*4:num_files*4+4].reshape([4,2])
      if args.visual:
        visualization = cv2.cvtColor(full_I, cv2.COLOR_BGR2RGB)
        cv2.polylines(visualization, np.int32([corr1]), 1,  (0, 255, 0))
        show_image((1,2,1),"I", visualization)
        visualization = cv2.cvtColor(full_I_prime, cv2.COLOR_BGR2RGB)
        cv2.polylines(visualization, np.int32([corr2]), 1,  (0, 255, 255))
        show_image((1,2,2),"I PRIME", visualization)
        plt.show()
    
    for i in range(args.im_per_im):
      progress_bar(index, args.num_data + args.start_index, 'Real %d/%d, Gen %d'%(num_files,len(image_files), index ))

      # Pick the top left point of the patch on the real image
      # Randomize x to have more data
      x = random.randint(rho, width - rho - patch_size)  # col?
      # We can also randomize y. In our case, y can only vary in a small range. Thus, we just pick a constant value
      y = (height - patch_size)/2

      # define corners of image patch
      top_left_point = (x, y)
      bottom_left_point = (patch_size + x, y)
      bottom_right_point = (patch_size + x, patch_size + y)
      top_right_point = (x, patch_size + y)
      four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
      perturbed_four_points = four_points

      # grab image patches
      I1   = I_gray[y:y + patch_size, x:x + patch_size]
      I2   = I_prime_gray[y:y + patch_size, x:x + patch_size]

      if args.visual:

        plt.figure(figsize=(10, 8))
        # visualize patches on color image
        patches_visualization = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        cv2.polylines(patches_visualization, np.int32([perturbed_four_points]), 1,  (0, 255, 0))
        cv2.polylines(patches_visualization, np.int32([four_points]), 1, (0, 0, 255))
        show_image((2, 2, 1), "ORIGINAL IMAGE", patches_visualization)
        # visualize patch on warped image
        warped_visualization = cv2.cvtColor(I_prime, cv2.COLOR_BGR2RGB)
        cv2.polylines(warped_visualization, np.int32([perturbed_four_points]), 1,  (0, 255, 0))
        show_image((2, 2, 2), "WARPED IMAGE", warped_visualization)

        # visualize patch itself
        patch_warped_visualization = I1.copy()
        show_image((2, 2, 3), "ORIGINAL PATCH", patch_warped_visualization)
        # visualize warped patch itself
        patch_warped_visualization = I2.copy()
        show_image((2, 2, 4), "WARPED PATCH", patch_warped_visualization)
        plt.show()
        plt.axis()
        plt.close()

        ######################################################################################

      if args.debug:
        index += 1
        return index

      # Save real data
      large_img_path  = os.path.join(args.I_dir, str(index) + '.jpg')
      full_large_img_path  = os.path.join(args.full_I_dir, str(index) + '.jpg')

      if args.mode == 'train' and args.color==False:
        cv2.imwrite(large_img_path, I_gray)
        cv2.imwrite(full_large_img_path, full_I_gray)
      else:
        # cv2.imwrite(large_img_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)  )
        cv2.imwrite(large_img_path, I)
        cv2.imwrite(full_large_img_path, full_I)

      if args.I_prime_dir is not None:
        img_prime_path = os.path.join(args.I_prime_dir, str(index) + '.jpg')
        full_img_prime_path = os.path.join(args.full_I_prime_dir, str(index) + '.jpg')

        if args.mode == 'train' and args.color==False:
          cv2.imwrite(img_prime_path, I_prime_gray)
          cv2.imwrite(full_img_prime_path, full_I_prime_gray)
        else:
          cv2.imwrite(img_prime_path,  I_prime)
          cv2.imwrite(full_img_prime_path,  full_I_prime)

      pts1 = np.array(four_points).flatten().astype(np.float32) # Store the 4 points
      np.savetxt(f_pts1, [pts1], delimiter= ' ')

      if args.mode=='test':
        corr_flat = np.hstack([corr1.flatten(), corr2.flatten()])
        np.savetxt(f_test_gt, [corr_flat], delimiter= ' ')

      f_file_list.write('%s %s\n'%(str(index)  +'.jpg', str(index)  +'.jpg') )

      index += 1
      if index >= args.num_data + args.start_index:
        break


    if index >= args.start_index + args.num_data:
      break

  f_pts1.close()
  f_file_list.close()
  if args.mode=='test':
    f_test_gt.close()
  return index, 0


def dataCollection(args):
  if (args.resume == 'N' or args.resume == 'n') and args.start_index == 0 and args.mode=='train' and not args.debug:
    try:
      os.remove(args.gt_file)
      os.remove(args.pts1_file)
      os.remove(args.filenames_file)
      print'-- Current {} existed. Deleting..!'.format(args.gt_file)
      shutil.rmtree(args.I_dir, ignore_errors=True)
      if args.I_prime_dir is not None:
        shutil.rmtree(args.I_prime_dir, ignore_errors=True)
    except :
      print'-- Train: Current {} not existed yet!'.format(args.gt_file)
  else:
    print '--- Appending to existing data---'

  if (args.resume == 'N' or args.resume == 'n') and args.mode=='test' and not args.debug:
    try:
      os.remove(args.test_gt_file)
      os.remove(args.test_pts1_file)
      os.remove(args.test_filenames_file)
      print'-- Test: Current {} existed. Deleting..!'.format(args.test_gt_file)
    except :
      print'-- Test: Current {} not existed yet!'.format(args.test_gt_file)
  else:
    pass
  if not args.debug:
    if not os.path.exists(args.I_dir):
      os.makedirs(args.I_dir)
    if args.I_prime_dir is not None and not os.path.exists(args.I_prime_dir):
      os.makedirs(args.I_prime_dir)
    if not os.path.exists(args.full_I_dir):
      os.makedirs(args.full_I_dir)
    if not os.path.exists(args.full_I_prime_dir):
      os.makedirs(args.full_I_prime_dir)

  raw_image_list = removeHiddenfile(os.listdir(args.raw_data_path))
  index = args.start_index
  index = homographyGeneration(args, args.raw_data_path, index)



def main():
  RHO = 24 # Maximum range of pertubation

  DATA_NUMBER = 10000
  TEST_DATA_NUMBER = 1000
  IM_PER_REAL = 20 # Generate 20 different pairs of images from one single real image

  # Size of synthetic image
  HEIGHT = 142 #
  WIDTH = 190
  PATCH_SIZE = 128

  FULL_HEIGHT = 480 #
  FULL_WIDTH  =  640
  # Directories to files
  RAW_DATA_PATH = "/Earthbyte/tynguyen/real_rawdata/joe_data/train/" # Real images used for generating real dataset
  TEST_RAW_DATA_PATH = "/Earthbyte/tynguyen/real_rawdata/joe_data/test/" # Real images used for generating real test dataset

  # Data directories
  DATA_PATH = "/Earthbyte/tynguyen/docker_folder/pose_estimation/data/synthetic/" + str(RHO) + '/'
  if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

  I_DIR = DATA_PATH + 'I/' # Large image 240 x 320
  I_PRIME_DIR = DATA_PATH + 'I_prime/' # Large image 240 x 320

  FULL_I_DIR = DATA_PATH + 'FULL_I/' # Large image size 480 x 640
  FULL_I_PRIME_DIR = DATA_PATH + 'FULL_I_prime/' # Large image size 480 x 640

  PTS1_FILE = os.path.join(DATA_PATH,'pts1.txt')
  FILENAMES_FILE = os.path.join(DATA_PATH,'train_real.txt')
  GROUND_TRUTH_FILE = os.path.join(DATA_PATH,'gt.txt')
  TEST_PTS1_FILE = os.path.join(DATA_PATH,'test_pts1.txt')
  TEST_FILENAMES_FILE = os.path.join(DATA_PATH,'test_real.txt')
  # In real dataset, ground truth file consists of correspondences
  # Each row in the file contains 8 numbers:[corr1, corr2]
  TEST_GROUND_TRUTH_FILE = os.path.join(DATA_PATH,'test_gt.txt')

  def str2bool(s):
    return s.lower() == 'true'

  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default='train', help='Train or test', choices=['train', 'test'])
  parser.add_argument('--color',type=str2bool,default='true', help='Generate color or gray images')
  parser.add_argument('--debug',type=str2bool,default='false', help='Debug Mode. Will not delete or store any thing')

  parser.add_argument('--raw_data_path', type=str, default=RAW_DATA_PATH, help='The raw data path.')
  parser.add_argument('--test_raw_data_path',type=str, default=TEST_RAW_DATA_PATH, help='The test raw data path.')
  parser.add_argument('--data_path',     type=str, default=DATA_PATH, help='The raw data path.')
  parser.add_argument('--I_dir',  			 type=str, default=I_DIR, help='The training image path')
  parser.add_argument('--I_prime_dir',   type=str, default=I_PRIME_DIR, help='The training image path')
  parser.add_argument('--full_I_dir',         type=str, default=FULL_I_DIR, help='The training image path')
  parser.add_argument('--full_I_prime_dir',   type=str, default=FULL_I_PRIME_DIR, help='The training image path')
  parser.add_argument('--pts1_file',      type=str, default=PTS1_FILE, help='The training H4P path')
  parser.add_argument('--test_pts1_file', type=str, default=TEST_PTS1_FILE, help='The test H4P path')
  parser.add_argument('--num_data',      type=int, default=DATA_NUMBER, help='The data size for training')
  parser.add_argument('--im_per_im',      type=int, default=IM_PER_REAL, help='Each pair of real image can generate up to this number of pairs')
  parser.add_argument('--test_num_data', type=int, default=TEST_DATA_NUMBER, help='The data size for test')
  parser.add_argument('--gt_file',       type=str, default=GROUND_TRUTH_FILE, help='The ground truth file')
  parser.add_argument('--test_gt_file',  type=str, default=TEST_GROUND_TRUTH_FILE, help='The ground truth file')
  parser.add_argument('--filenames_file',     type=str, default=FILENAMES_FILE, help='File that contains all names of files')
  parser.add_argument('--test_filenames_file',type=str, default=TEST_FILENAMES_FILE, help='File that contains all names of files')
  parser.add_argument('--visual',   type=str2bool, default='false', help='Visualize obtained images to debug')
  parser.add_argument('--artifact_mode', type=str, default='None', help='Add aftifacts to the images', choices=['noise', 'None'])

  parser.add_argument('--img_w',         type=int, default=WIDTH)
  parser.add_argument('--img_h',         type=int, default=HEIGHT)
  parser.add_argument('--full_img_w',         type=int, default=FULL_WIDTH)
  parser.add_argument('--full_img_h',         type=int, default=FULL_HEIGHT)
  parser.add_argument('--rho',           type=int, default=RHO)
  parser.add_argument('--patch_size',    type=int, default=PATCH_SIZE)


  parser.add_argument('--resume', type=str, default='N', help='Y: append to existing data. N: delete old data, create new data')
  parser.add_argument('--start_index', type=int, default=0, help='start_index of the new sample')

  args = parser.parse_args()
  # Ignore some pairs of images with first image having the id as follows
  train_ignore_list = [59, 60, 91, 121, 122,  149, 150, 180, 181, 238, 239, 266, 267, 296, 297, 327]
  test_ignore_list = [30, 31]
  args.ignore_list = train_ignore_list

  print('<==================== Loading raw data ===================>\n')
  if args.mode =='test':
    args.start_index = args.num_data
    args.num_data    = args.test_num_data
    args.raw_data_path = args.test_raw_data_path
    args.ignore_list = test_ignore_list

  print '<================= Generating Data .... =================>\n'

  dataCollection(args)


if __name__ == '__main__':
  main()
  #query_gt_test_set()
