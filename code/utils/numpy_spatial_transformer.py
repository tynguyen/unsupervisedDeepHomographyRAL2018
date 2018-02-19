# Simple version of spatial_transformer.py, work on a single image with multiple channels  
import numpy as np 
import cv2 
import pdb 
import matplotlib.pyplot as plt 
from skimage import io 
###############################################################
# Changable parameter
SCALE_H = True 
# scale_H:# The indices of the grid of the target output is
# scaled to [-1, 1]. Set False to stay in normal mode 
def _meshgrid(height, width, scale_H = SCALE_H):
    if scale_H:
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                        np.linspace(-1, 1, height))
        ones = np.ones(np.prod(x_t.shape))
        grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

    else:
        x_t, y_t = np.meshgrid(range(0,width), range(0,height))
        ones = np.ones(np.prod(x_t.shape))
        grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # print '--grid size:', grid.shape 
    return grid 


def _interpolate(im, x, y, out_size, scale_H = SCALE_H):
    # constants
    height = im.shape[0]
    width =  im.shape[1]


    height_f = float(height)
    width_f =  float(width)
    out_height = out_size[0]
    out_width = out_size[1]
    zero = np.zeros([], dtype='int32')
    max_y = im.shape[0] - 1
    max_x = im.shape[1] - 1

    if scale_H:
        # # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0)*(height_f) / 2.0

    # do sampling
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # print 'x0:', y0 
    # print 'x1:', y1 
    # Limit the size of the output image 
    x0 = np.clip(x0, zero, max_x)
    x1 = np.clip(x1, zero, max_x)
    y0 = np.clip(y0, zero, max_y)
    y1 = np.clip(y1, zero, max_y)
    

    
    Ia = im[ y0, x0, ... ]
    Ib = im[ y1, x0, ... ]
    Ic = im[ y0, x1, ... ]
    Id = im[ y1, x1, ... ]

    # print
    # plt.figure(2)
    # plt.subplot(221)
    # plt.imshow(Ia)
    # plt.subplot(222)
    # plt.imshow(Ib)
    # plt.subplot(223)
    # cv2.imshow('Ic', Ic)
    # plt.subplot(224)
    # plt.imshow(Id)
    # cv2.waitKey(0)

    wa = (x1 -x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    # print 'wabcd...', wa,wb, wc,wd 
    
    # Handle multi channel image 
    if im.ndim == 3:
      num_channels = im.shape[2]
      wa = np.expand_dims(wa, 2)
      wb = np.expand_dims(wb, 2)  
      wc = np.expand_dims(wc, 2) 
      wd = np.expand_dims(wd, 2) 
    out = wa*Ia + wb*Ib + wc*Ic + wd*Id
    # print '--shape of out:', out.shape
    return out 

def _transform(theta, input_dim, out_size):
    height, width = input_dim.shape[0], input_dim.shape[1]
    theta = np.reshape(theta, (3, 3))
    # print '--Theta:', theta 
    # print '-- Theta shape:', theta.shape  

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    out_height = out_size[0]
    out_width = out_size[1]
    grid = _meshgrid(out_height, out_width)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    T_g = np.dot(theta, grid)
    x_s = T_g[0,:]
    y_s = T_g[1,:]
    t_s = T_g[2,:]
    # print '-- T_g:', T_g 
    # print '-- x_s:', x_s 
    # print '-- y_s:', y_s
    # print '-- t_s:', t_s

    t_s_flat = np.reshape(t_s, [-1])
    # Ty changed 
    # x_s_flat = np.reshape(x_s, [-1])
    # y_s_flat = np.reshape(y_s, [-1])
    x_s_flat = np.reshape(x_s, [-1])/t_s_flat
    y_s_flat = np.reshape(y_s, [-1])/t_s_flat
    

    input_transformed =  _interpolate(input_dim, x_s_flat, y_s_flat, out_size) 
    if input_dim.ndim == 3:
      output = np.reshape(input_transformed, [out_height, out_width, -1])
    else: 
      output = np.reshape(input_transformed, [out_height, out_width])
      
    output = output.astype(np.uint8)
    return output


def numpy_transformer(img, H, out_size, scale_H = SCALE_H): 
    h, w = img.shape[0], img.shape[1]
    # Matrix M 
    M = np.array([[w/2.0, 0, w/2.0], [0, h/2.0, h/2.0], [0, 0, 1.]]).astype(np.float32)

    if scale_H:
        H_transformed = np.dot(np.dot(np.linalg.inv(M), np.linalg.inv(H)), M)
        # print 'H_transformed:', H_transformed 
        img2 = _transform(H_transformed, img, [h,w])
    else:
        img2 = _transform(np.linalg.inv(H), img, [h,w])
    return img2 


def test_transformer(scale_H = SCALE_H): 
    img = io.imread('/home/tynguyen/cis680/data/cifar10_transformed/imgs/05975.png')
    h, w = img.shape[0], img.shape[1]
    print( '-- h, w:', h, w ) 


    # Apply homography transformation 

    H = np.array([[2., 0.3, 5], [0.3, 2., 10.], [0.0001, 0.0002, 1.]]).astype(np.float32)
    img2 = cv2.warpPerspective(img, H, (w, h))


    # # Matrix M 
    M = np.array([[w/2.0, 0, w/2.0], [0, h/2.0, h/2.0], [0, 0, 1.]]).astype(np.float32)

    if scale_H:
        H_transformed = np.dot(np.dot(np.linalg.inv(M), np.linalg.inv(H)), M)
        print('H_transformed:', H_transformed)
        img3 = _transform(H_transformed, img, [h,w])
    else:
        img3 = _transform(np.linalg.inv(H), img, [h,w])

    print ( '-- Reprojection error:', np.mean(np.abs(img3 - img2))) 
    Reprojection = abs(img3 - img2)
    # Test on real image 
    count = 0 
    amount = 0 
    for i in range(48):
      for j in range(48):
        for k in range(2):
          if Reprojection[i, j, k] > 10:
            print(i, j, k, 'value', Reprojection[i, j, k])
            count += 1 
            amount += Reprojection[i, j, k]
    print('There is total %d > 10, over total %d, account for %.3f'%( count, 48*48*3,amount*1.0/count) ) 
    
    #io.imshow('img3', img3) 
    try:
        plt.subplot(221)
        plt.imshow(img)
        plt.title('Original image')

        plt.subplot(222)
        plt.imshow(img2)
        plt.title('cv2.warpPerspective')

        plt.subplot(223)
        plt.imshow(img3)
        plt.title('Transformer')

        plt.subplot(224)
        plt.imshow(Reprojection)
        plt.title('Reprojection Error')
        plt.show()
    except KeyboardInterrupt:
        plt.close()
        exit(1)


if __name__ == "__main__":
    test_transformer()
 