from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, \
        PoolingLayer, EltwiseLayer,Dw_convLayer, G_convLayer

#uncomment one of the groups to run the appropriate version fo mobilenet, one is vanilla version
#The second one is the group convolution version
'''
The format for specifying group convolution can be specified using the following format
G_convLayer(nifm, nofm, sofm, sfil,no_g,strd=1)

The format for specifying depthwise-convolution can be specified using the following format
G_convLayer(nifm, nofm, sofm, sfil,no_g,strd=1)
Dw_convLayer(nifm, nofm, sofm, sfil, strd=1)

nifm= number of input channels
nofm= number of output channels
sofm= output size (H or W)
sfil= filter dimension
strd= stride
no_g= 'g' or no. of groups as per group convolution literature
Inorder to get G=1 , we set no_g =nifm 
Inorder to get G=2 , we set no_g =nifm/2
.
.
. 

'''
NN= Network('Mobilenet')
NN.set_input_layer(InputLayer(3, 224))


#Network.add('conv_sqz1_{}_a'.format(i), ConvLayer(nifm,int(redu*nifm),inp_h,1,strd= strd),prevs=_prevs)

NN.add('conv1', ConvLayer(3, 32, 112, 3, 2))

NN.add('conv2_dw_a', Dw_convLayer(32, 32, 112, 3, 1))
NN.add('conv2_dw_b', ConvLayer(32, 64, 112, 1, 1))

NN.add('conv3_dw_a', Dw_convLayer(64, 64, 56, 3, 2))
NN.add('conv3_dw_b', ConvLayer(64, 128, 56, 1, 1))

NN.add('conv4_dw_a', Dw_convLayer(128, 128, 56, 3, 1))
NN.add('conv4_dw_b', ConvLayer(128, 128, 56, 1, 1))

NN.add('conv5_dw_a', Dw_convLayer(128, 128, 28, 3, 2))
NN.add('conv5_dw_b', ConvLayer(128, 256, 28, 1, 1))

NN.add('conv6_dw_a', Dw_convLayer(256, 256, 28, 3, 1))
NN.add('conv6_dw_b', ConvLayer(256, 256, 28, 1, 1))

NN.add('conv7_dw_a', Dw_convLayer(256, 256, 14, 3, 2))
NN.add('conv7_dw_b', ConvLayer(256, 512, 14, 1, 1))

prev= 'conv7_dw_b'

for i in range(5):
    NN.add('conv{}_dw_a'.format(i+8),Dw_convLayer(512, 512, 14, 3, 1),prevs= prev)
    NN.add('conv{}_dw_b'.format(i+8), ConvLayer(512, 512, 14, 1, 1))

    prev= 'conv{}_dw_b'.format(i+8)

NN.add('conv13_dw_a', Dw_convLayer(512, 512, 7, 3, 2),prevs= 'conv12_dw_b')
NN.add('conv13_dw_b', ConvLayer(512, 1024, 7, 1, 1))

NN.add('conv14_dw_a', Dw_convLayer(1024, 1024, 7, 3, 1))
NN.add('conv14_dw_b', ConvLayer(1024, 1024, 7, 1, 1))

NN.add('pool1', PoolingLayer(1024, 1, 7, 1))
NN.add('fc1', FCLayer(1024, 1000))
'''


'''
#Group conv version of mobile_net
NN= Network('Mobilenet')
NN.set_input_layer(InputLayer(3, 224))


#Network.add('conv_sqz1_{}_a'.format(i), ConvLayer(nifm,int(redu*nifm),inp_h,1,strd= strd),prevs=_prevs)

NN.add('conv1', ConvLayer(3, 32, 112, 3, 2))

NN.add('conv2_dw_a', G_convLayer(32, 32, 112, 3,no_g=32,strd=1))
NN.add('conv2_dw_b', ConvLayer(32, 64, 112, 1, 1))

NN.add('conv3_dw_a', G_convLayer(64, 64, 56, 3,no_g=64,strd=2))
NN.add('conv3_dw_b', ConvLayer(64, 128, 56, 1, 1))

NN.add('conv4_dw_a', G_convLayer(128, 128, 56, 3,no_g=128,strd=1))
NN.add('conv4_dw_b', ConvLayer(128, 128, 56, 1, 1))

NN.add('conv5_dw_a', G_convLayer(128, 128, 28, 3,no_g=128,strd=2))
NN.add('conv5_dw_b', ConvLayer(128, 256, 28, 1, 1))

NN.add('conv6_dw_a', G_convLayer(256, 256, 28, 3,no_g=256,strd=1))
NN.add('conv6_dw_b', ConvLayer(256, 256, 28, 1, 1))

NN.add('conv7_dw_a', G_convLayer(256, 256, 14, 3,no_g=256,strd=2))
NN.add('conv7_dw_b', ConvLayer(256, 512, 14, 1, 1))

prev= 'conv7_dw_b'

for i in range(5):
    NN.add('conv{}_dw_a'.format(i+8),G_convLayer(512, 512, 14, 3,no_g=512,strd=1),prevs= prev)
    NN.add('conv{}_dw_b'.format(i+8), ConvLayer(512, 512, 14, 1, 1))

    prev= 'conv{}_dw_b'.format(i+8)

NN.add('conv13_dw_a', G_convLayer(512, 512, 7, 3,no_g=512,strd=2),prevs= 'conv12_dw_b')
NN.add('conv13_dw_b', ConvLayer(512, 1024, 7, 1, 1))

NN.add('conv14_dw_a', G_convLayer(1024, 1024, 7, 3,no_g=1024,strd=1))
NN.add('conv14_dw_b', ConvLayer(1024, 1024, 7, 1, 1))

NN.add('pool1', PoolingLayer(1024, 1, 7, 1))
NN.add('fc1', FCLayer(1024, 1000))

'''
