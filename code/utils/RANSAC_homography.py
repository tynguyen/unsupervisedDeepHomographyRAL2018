# RANSAC_homography.py
# Homography models using SIFT/ ORB 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb, time 
def find_homography(img1_color, img2_color, H4p1=None, H4p2_gt=None, visual=False, method='ORB', min_match_count = 25, return_h_inv=True):
    """By default, return h_inv for synthetic data"""
    print('===> Image size:', img1_color.shape) 
    if len(img1_color.shape) == 3:
      img1 = cv2.cvtColor(img1_color, cv2.COLOR_RGB2GRAY) 
      img2 = cv2.cvtColor(img2_color, cv2.COLOR_RGB2GRAY)
      if visual:
          img1_color = cv2.cvtColor(img1_color, cv2.COLOR_RGB2BGR)
          img2_color = cv2.cvtColor(img2_color, cv2.COLOR_RGB2BGR)
    else:
      img1 = img1_color
      img2 = img2_color
    
    # Create feature detectors
    if method=='ORB':
        ft_detector = cv2.ORB_create()
        # Create brute-force matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)
    elif method=='SIFT':
        ft_detector = cv2.xfeatures2d.SIFT_create()
        # Create brute-force matcher object
        bf = cv2.BFMatcher(crossCheck=False)
    else:
      return  np.zeros([1,4,2]), np.eye(3), 0 

    keyPoints1, descriptors1 = ft_detector.detectAndCompute(img1, None)
    keyPoints2, descriptors2 = ft_detector.detectAndCompute(img2, None)

    try:
      descriptors1 = descriptors1.astype(np.uint8)
      descriptors2 = descriptors2.astype(np.uint8)
    except:
      print('===> Error on finding descriptors') 
      return np.zeros([1,4,2]), np.eye(3), 1 
    
    
    if visual:
     print('===> No of keypoints:', len(keyPoints1), len(keyPoints2)) 
     kp_img1 = img1.copy() 
     kp_img1 = cv2.drawKeypoints(img1, keyPoints1, kp_img1)
     kp_img2 = img2.copy()
     kp_img2 = cv2.drawKeypoints(img2, keyPoints2, kp_img2)
     kp_pair = np.concatenate((kp_img1, kp_img2), axis=1)
     plt.imshow(kp_pair, cmap='gray') 
     plt.axis('off')
     plt.show()
     plt.pause(0.05)
    # Match the descriptors
    try:
        matches = bf.match(descriptors1, descriptors2)
    except:
        print('===> Error on Matching. Return identity matrix') 
        return np.zeros([1,4,2]), np.eye(3), 1 
   
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Check number of matches 
    if len(matches) <= 0:
      print('===> Not enough matched features. Return identity matrix')
      return np.zeros([1,4,2]), np.eye(3), 1 

    num_choosing = min_match_count+ 1 
    if num_choosing > len(matches):
        num_choosing = len(matches)
    
    #TODO: use all features or not
    # goodMatches = matches[0:num_choosing]
    
    # Synthetic:
    goodMatches = matches
    # Real:
    if not return_h_inv:
      goodMatches = matches[0:25]
    
    # Draw matches 
    if visual:
      #newKP2 = [keyPoints2[m.trainIdx] for m in goodMatches]
      print('===> No of matches, goodMatches:', len(matches), len(goodMatches)) 
      match_img = cv2.drawMatches(img1, keyPoints1, img2, keyPoints2, goodMatches, None, flags=2)
      plt.imshow(match_img)
      plt.axis('off') 
      plt.show()
      plt.pause(0.05)
    # Apply the homography transformation if we have enough good matches 
    # print('===> Applying RANSAC...')

    if len(goodMatches) >= min_match_count:
        # Get the good key points positions
        sourcePoints = np.float32([ keyPoints1[m.queryIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
        destinationPoints = np.float32([ keyPoints2[m.trainIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
        # Obtain homography
        try:   
          M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=5.0)
          matchesMask = mask.ravel().tolist()
        except:
          return np.zeros([1,4,2]), np.eye(3), 1
        try:
          if return_h_inv:
            M_inv = np.linalg.inv(M)
          else:
            M_inv = M 
        except:
          print('Error in inverse H', mask)
          return np.zeros([1,4,2]), np.eye(3), 1 
        
        # Apply the perspective transformation to the source image corners
        h, w = img1.shape
        if len(H4p1) > 0:
            corners = np.float32(H4p1).reshape(1,4,2)
            transformedCorners = cv2.perspectiveTransform(corners,M_inv)
        else:
            corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(1,4,2)
            transformedCorners = cv2.perspectiveTransform(corners, M_inv)


    else:
        print("-----Found - %d/%d" % (len(goodMatches), min_match_count))
        matchesMask = None
        return np.zeros([1,4,2]), np.eye(3), 1 

    # Draw the matches
    if visual:
        drawParameters = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
        img_warped = cv2.warpPerspective(img1_color, M, (img1_color.shape[1], img1_color.shape[0])) 
        
        
        # Draw a polygon on the second image joining the transformed corners  
        img2_color = cv2.polylines(img2_color, [np.int32(transformedCorners)], True, (0, 100, 255),3, cv2.LINE_AA)
        img1_color = cv2.polylines(img1_color, [np.int32(corners)], True, (125, 55, 125),3, cv2.LINE_AA)
   
        try: 
            img2_color = cv2.polylines(img2_color, [np.int32(H4p2_gt)], True, (0, 255, 0),3, cv2.LINE_AA)
        except:
            pass 
        result = cv2.drawMatches(img1_color, keyPoints1, img2_color, keyPoints2, goodMatches, None, **drawParameters)
        # resize image 
        #result = cv2.resize(result, (1248, 188))
        #img_warped = cv2.resize(img_warped, (624,188))   
        ## Display the results
        #cv2.imshow('Homography', result)
        #cv2.imshow('Warped 1', img_warped)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        plt.subplot(3,1,1)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) #cmap='gray')
        plt.title('Matching')
        plt.axis('off')
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
 
