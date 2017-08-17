import numpy as np  
from PIL import Image  
import matplotlib.pyplot as plt  
import sys     
import caffe  
import cv
import scipy.io
# import pydensecrf.densecrf as dcrf 
# from pydensecrf.utils import compute_unary, create_pairwise_bilateral,create_pairwise_gaussian, softmax_to_unary 
import pdb
# matplotlib inline  
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe  
im = Image.open('/home/li/Documents/fcn.berkeleyvision.org/nyud-fcn32s-color/test.png')  
in_ = np.array(im, dtype=np.float32)  
in_ = in_[:,:,::-1]  
in_ -= np.array((104.00698793,116.66876762,122.67891434))  
in_ = in_.transpose((2,0,1))  
  
# load net  
net = caffe.Net('/home/li/Documents/fcn.berkeleyvision.org/nyud-fcn32s-color/deploy.prototxt', '/home/li/Downloads/nyud-fcn32s-color-heavy.caffemodel', caffe.TEST)  
# shape for input (data blob is N x C x H x W), set data  
net.blobs['data'].reshape(1, *in_.shape)  
net.blobs['data'].data[...] = in_  
# run net and take argmax for prediction  
net.forward()  
# pdb.set_trace()
out = net.blobs['score'].data[0].argmax(axis=0) 
scipy.io.savemat('/home/li/Documents/fcn.berkeleyvision.org/nyud-fcn32s-color/out.mat',{'X':out}) 
#print "hello,python!"  
  
#plt.imshow(out,cmap='gray');  
plt.imshow(out)  
plt.axis('off')  
plt.savefig('testout_32s.png')  
