
'''
compute PSNR with tensorflow
'''
import tensorflow as tf
import os
import math
import numpy as np
import cv2
 
 
def read_img(path):
	return cv2.imread(path)
 
def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def _main():
    c = []
    b = os.listdir('./source')
    a=os.listdir('./target')
    for i in range(0,100):
    
        print(a[i])
        print('------------------------------------------')
        print(b[i])
        t1 = read_img('('./source/{}'.format(a[i]))
        t2 = read_img('./target/{}'.format(b[i]))
        y = psnr(t1, t2)
        c.append(y)
        print('The PSNR value is:',y)
    print(c)

 
if __name__ == '__main__':
    _main()

