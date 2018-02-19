# direct_homography.py
# Homography models using direct methods
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb, time

def find_homography(img1_color, img2_color, H4p1=None, H4p2_gt=None, visual=False, method='ECC', num_iterations=100, return_h_inv=True):
    if len(img1_color.shape) == 3:
      img1 = cv2.cvtColor(img1_color, cv2.COLOR_RGB2GRAY)
      img2 = cv2.cvtColor(img2_color, cv2.COLOR_RGB2GRAY)
      if visual:
          img1_color = cv2.cvtColor(img1_color, cv2.COLOR_RGB2BGR)
          img2_color = cv2.cvtColor(img2_color, cv2.COLOR_RGB2BGR)
    else:
      img1 = img1_color
      img2 = img2_color

    # Motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 3x3 matrice and initialize  the matrix to identity
    warp_matrix = np.eye(3,3, dtype = np.float32)

    # Number of iteration
    num_iters = num_iterations

    # Threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iters, termination_eps)

    # Rune the ECC algorithm.
    try:
      (cc, warp_matrix) = cv2.findTransformECC(img1, img2, warp_matrix, warp_mode, criteria)
    except:
      print("Error in convergence...")
      return  np.zeros([1,4,2]), np.eye(3), 1


    M = warp_matrix

    try:
      if return_h_inv:
        M_inv = np.linalg.inv(M)
      else:
        M_inv = M
    except:
      print 'Error in inverse H'
      return np.zeros([1,4,2]), np.eye(3), 1

    # Apply the perspective transformation to the source image corners
    h, w = img1.shape
    if len(H4p1) > 0:
        corners = np.float32(H4p1).reshape(1,4,2)
        transformedCorners = cv2.perspectiveTransform(corners,M_inv)
    else:
        corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(1,4,2)
        transformedCorners = cv2.perspectiveTransform(corners, M_inv)


    # Draw the matches
    if visual:
        img_warped = cv2.warpPerspective(img1_color, M, (img1_color.shape[1], img1_color.shape[0]))


        # Draw a polygon on the second image joining the transformed corners
        img2_color = cv2.polylines(img2_color, [np.int32(transformedCorners)], True, (0, 100, 255),3, cv2.LINE_AA)
        img1_color = cv2.polylines(img1_color, [np.int32(corners)], True, (125, 55, 125),3, cv2.LINE_AA)

        try:
            img2_color = cv2.polylines(img2_color, [np.int32(H4p2_gt)], True, (0, 255, 0),3, cv2.LINE_AA)
        except:
            pass
        # resize image
        #result = cv2.resize(result, (1248, 188))
        #img_warped = cv2.resize(img_warped, (624,188))
        ## Display the results
        #cv2.imshow('Homography', result)
        #cv2.imshow('Warped 1', img_warped)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        plt.subplot(3,1,2)
        plt.imshow(cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB))#,cmap='gray')
        plt.title('Warped Image 1')
        plt.axis('off')
        plt.subplot(3,1,3)
        plt.imshow( cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))#,cmap='gray')
        plt.title('Image 2')
        plt.axis('off')
        plt.show()
        plt.pause(0.05)
    return transformedCorners - corners, M, 0
