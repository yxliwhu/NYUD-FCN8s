import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

vgg_weights = '../ilsvrc-nets/VGG_ILSVRC_16_layers.caffemodel'
vgg_proto = '../ilsvrc-nets/VGG_ILSVRC_16_layers_deploy.prototxt'

# init
#caffe.set_device(int(sys.argv[1]))
#caffe.set_device(0)
#caffe.set_mode_gpu()
caffe.set_mode_cpu()

# solver = caffe.SGDSolver('solver.prototxt')
# solver.net.copy_from(weights)
solver = caffe.SGDSolver('solver.prototxt')
vgg_net = caffe.Net(vgg_proto, vgg_weights, caffe.TRAIN)
surgery.transplant(solver.net, vgg_net)
del vgg_net

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
test = np.loadtxt('../data/nyud/test.txt', dtype=str)

for _ in range(50):
    solver.step(2000)
    score.seg_tests(solver, False, test, layer='score')
