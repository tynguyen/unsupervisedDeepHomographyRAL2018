import os, pdb, shutil, argparse
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from shapely.geometry import Polygon, Point
from utils import progress_bar, removeHiddenfile, draw_matches
from utils import show_image, find_nine_grid_points
from numpy_spatial_transformer import numpy_transformer
from matplotlib import pyplot as plt

def homographyGeneration(args, raw_image_path, index):

  rho = args.rho
  patch_size = args.patch_size
  height = args.img_h
  width =  args.img_w

  try:
    color_image = cv2.imread(raw_image_path)
    color_image = cv2.resize(color_image, (width, height))
    gray_image  = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

  except:
    print 'Error with image:', raw_image_path
    return index, -1


  # Text files to store homography parameters (4 corners)
  if args.mode=='train' and not args.debug:
    f_pts1 = open(args.pts1_file, 'ab')
    f_gt = open(args.gt_file, 'ab')
    f_file_list = open(args.filenames_file, 'ab')
  elif not args.debug:
    f_pts1 = open(args.test_pts1_file, 'ab')
    f_gt = open(args.test_gt_file, 'ab')
    f_file_list = open(args.test_filenames_file, 'ab')


  name_suffix = 1
  i = 1
  num_zeros = 0



  while i < args.img_per_real + 1:
    # Randomly pick the top left point of the patch on the real image
    y = random.randint(rho, height - rho - patch_size)  # row?
    x = random.randint(rho, width - rho - patch_size)  # col?
    # x = 4
    # y = 2

    # define corners of image patch
    top_left_point = (x, y)
    bottom_left_point = (patch_size + x, y)
    bottom_right_point = (patch_size + x, patch_size + y)
    top_right_point = (x, patch_size + y)
    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
    perturbed_four_points = []
    for point in four_points:
      perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    # compute H
    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    try:
      H_inverse = inv(H)
    except:
      print "singular Error!"
      return index, 0

    iinv_warped_color_image = numpy_transformer(color_image, H_inverse, (width, height))
    inv_warped_image = numpy_transformer(gray_image, H_inverse, (width, height))

    # Extreact image patches
    original_patch = gray_image[y:y + patch_size, x:x + patch_size]
    warped_patch   = inv_warped_image[y:y + patch_size, x:x + patch_size]

    if args.visual:
      corr = np.array(four_points[0]) + 10
      corr_prime = cv2.perspectiveTransform(np.array([[corr]],dtype=np.float32), H_inverse)[0,0]

      patches_visualization = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
      cv2.polylines(patches_visualization, np.int32([perturbed_four_points]), 1,  (0, 255, 0))
      cv2.polylines(patches_visualization, np.int32([four_points]), 1, (0, 0, 255))
      cv2.circle(patches_visualization,tuple(corr), 2, (0,255,255), -1)
      show_image((2, 2, 1), "I1", patches_visualization)
      # visualize patch on warped image
      patch_warped_visualization = inv_warped_image.copy()
      cv2.polylines(patch_warped_visualization, np.int32([four_points]), 1, (0, 0, 0))
      cv2.polylines(patch_warped_visualization, np.int32([perturbed_four_points]), 1,  (0, 255, 0))
      cv2.circle(patch_warped_visualization, tuple(corr_prime), 2, (0), -1)
      show_image((2, 2, 2), "I2", patch_warped_visualization)

      # Warp image 2 back using inverse of (H_inverse)
      I2_to_I1 = numpy_transformer(inv_warped_color_image, H, (width, height))
      I2_to_I1 = cv2.cvtColor(I2_to_I1, cv2.COLOR_BGR2RGB)
      cv2.polylines(I2_to_I1, np.int32([perturbed_four_points]), 1,  (0, 255, 0))
      cv2.polylines(I2_to_I1, np.int32([four_points]), 1, (0, 0, 255))
      show_image((2, 2, 3), "I2 warped to I1", I2_to_I1)
      show_image((2, 2, 4), "(I2 warped to I1) vs I1", np.abs(np.mean(I2_to_I1, 2) - gray_image))
      plt.show()
      ######################################################################################

    if args.debug:
      index += 1
      return index, 0

    # Save synthetic data

    large_img_path    = os.path.join(args.I_dir, str(index) + '.jpg')

    if args.mode == 'train' and args.color==False:
      cv2.imwrite(large_img_path, gray_image)
    else:
      # cv2.imwrite(large_img_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)  )
      cv2.imwrite(large_img_path, color_image)

    if args.I_prime_dir is not None:
      img_prime_path = os.path.join(args.I_prime_dir, str(index) + '.jpg')
      if args.mode == 'train' and args.color==False:
        cv2.imwrite(img_prime_path, inv_warped_image)
      else:
        # cv2.imwrite(img_prime_path, cv2.cvtColor(inv_warped_color_image, cv2.COLOR_RGB2BGR) )
        cv2.imwrite(img_prime_path,  inv_warped_color_image)

    name_suffix += 1

    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    delta_h4p = np.array(H_four_points).flatten().astype(np.float32) # Store the delta of 4 points instead of h for less error later
    pts1 = np.array(four_points).flatten().astype(np.float32) # Store the 4 points

    # Debug
    # print 'delta_h4p:', delta_h4p
    # print 'h4p1:', h4p
    # print 'h4p2:', np.array(perturbed_four_points).reshape([-1])
    # print 'Error:', pts1 + delta_h4p - np.array(perturbed_four_points).reshape([-1])
    # print 'Res H:'
    # H_res = cv2.getPerspectiveTransform(np.array(four_points).astype(np.float32),np.array(perturbed_four_points).astype(np.float32) )
    # print 'H-res:\n', H_res
    # print 'Compare to origin:\n', H

    np.savetxt(f_gt, [delta_h4p], delimiter= ' ')
    np.savetxt(f_pts1, [pts1], delimiter= ' ')
    f_file_list.write('%s %s\n'%(str(index)  +'.jpg', str(index)  +'.jpg') )
    i += 1
    index += 1
    if index >= args.num_data + args.start_index:
      break
    if index %1000 == 0:
      print '--image number ', index

  f_gt.close()
  f_pts1.close()
  f_file_list.close()
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
    print '--- Train: Appending to existing data---'

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

  raw_image_list = removeHiddenfile(os.listdir(args.raw_data_path))
  index = args.start_index

  start_img_index = 0
  for img_index in range(len(raw_image_list)):
    if img_index < start_img_index:
      continue
    raw_img_name = raw_image_list[img_index]
    raw_image_path = os.path.join(args.raw_data_path,raw_img_name)
    index, error = homographyGeneration(args, raw_image_path, index)
    if error == -1:
      continue

    if index >= args.num_data + args.start_index:
      break



def main():
  RHO = 45 # The maximum value of pertubation

  DATA_NUMBER = 100000
  TEST_DATA_NUMBER = 5000
  IM_PER_REAL = 2 # Generate 2 different synthetic images from one single real image

  # Size of synthetic image
  HEIGHT = 240 #
  WIDTH = 320
  PATCH_SIZE = 128

  # Directories to files
  RAW_DATA_PATH = "/Earthbyte/tynguyen/rawdata/train/" # Real images used for generating synthetic data
  TEST_RAW_DATA_PATH = "/Earthbyte/tynguyen/rawdata/test/" # Real images used for generating test synthetic data

  # Synthetic data directories
  DATA_PATH = "/Earthbyte/tynguyen/docker_folder/pose_estimation/data/synthetic/" + str(RHO) + '/'
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

  def str2bool(s):
    return s.lower() == 'true'

  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default='test', help='Train or test', choices=['train', 'test'])
  parser.add_argument('--color',type=str2bool,default='true', help='Generate color or gray images')
  parser.add_argument('--debug',type=str2bool,default='false', help='Debug Mode. Will not delete or store any thing')

  parser.add_argument('--raw_data_path', type=str, default=RAW_DATA_PATH, help='The raw data path.')
  parser.add_argument('--test_raw_data_path',type=str, default=TEST_RAW_DATA_PATH, help='The test raw data path.')
  parser.add_argument('--data_path',     type=str, default=DATA_PATH, help='The raw data path.')
  parser.add_argument('--I_dir',  			 type=str, default=I_DIR, help='The training image path')
  parser.add_argument('--I_prime_dir',   type=str, default=I_PRIME_DIR, help='The training image path')
  parser.add_argument('--pts1_file',      type=str, default=PTS1_FILE, help='The training path to 4 corners on first image')
  parser.add_argument('--test_pts1_file', type=str, default=TEST_PTS1_FILE, help='The test path to 4 corners on second image')
  parser.add_argument('--num_data',      type=int, default=DATA_NUMBER, help='The data size for training')
  parser.add_argument('--test_num_data', type=int, default=TEST_DATA_NUMBER, help='The data size for test')
  parser.add_argument('--gt_file',       type=str, default=GROUND_TRUTH_FILE, help='The ground truth file')
  parser.add_argument('--test_gt_file',  type=str, default=TEST_GROUND_TRUTH_FILE, help='The ground truth file')
  parser.add_argument('--filenames_file',     type=str, default=FILENAMES_FILE, help='File that contains all names of files')
  parser.add_argument('--test_filenames_file',type=str, default=TEST_FILENAMES_FILE, help='File that contains all names of files')
  parser.add_argument('--visual',   type=str2bool, default='false', help='Visualize obtained images to debug')
  parser.add_argument('--artifact_mode', type=str, default='None', help='Add aftifacts to the images', choices=['noise', 'None'])

  parser.add_argument('--img_w',         type=int, default=WIDTH)
  parser.add_argument('--img_h',         type=int, default=HEIGHT)
  parser.add_argument('--rho',           type=int, default=RHO)
  parser.add_argument('--patch_size',    type=int, default=PATCH_SIZE)
  parser.add_argument('--img_per_real',   type=int, default=IM_PER_REAL)

  parser.add_argument('--resume', type=str, default='N', help='Y: append to existing data. N: delete old data, create new data')
  parser.add_argument('--start_index', type=int, default=0, help='start_index of the new created sample')

  args = parser.parse_args()
  print('<==================== Loading raw data ===================>\n')
  if args.mode =='test':
    args.start_index = args.num_data
    args.num_data    = args.test_num_data
    args.raw_data_path = args.test_raw_data_path

  print '<================= Generating Data .... =================>\n'

  dataCollection(args)


if __name__ == '__main__':
  main()
