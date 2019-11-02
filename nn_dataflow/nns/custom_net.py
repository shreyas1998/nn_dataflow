

from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, \
        PoolingLayer, EltwiseLayer,G_convLayer,Dw_convLayer

'''
Experiment performed to see variations of PE utilization with input size


#Experimenting with an input size of 28*28
NN = Network('custom_net')

NN.set_input_layer(InputLayer(64, 28))

NN.add('conv1',G_convLayer(64,64,28, 3,no_g=1,strd=1))

NN.add('conv2',G_convLayer(64,64,28, 3,no_g=2,strd=1))

NN.add('conv3',G_convLayer(64, 64, 28, 3,no_g=4,strd=1))

NN.add('conv4',G_convLayer(64, 64, 28, 3,no_g=8,strd=1))

NN.add('conv5',G_convLayer(64, 64, 28, 3,no_g=16,strd=1))

NN.add('conv6',G_convLayer(64, 64, 28, 3,no_g=32,strd=1))

NN.add('conv7',G_convLayer(64, 64, 28, 3,no_g=64,strd=1))



#Experimenting with an input size of 56*56
NN = Network('custom_net')

NN.set_input_layer(InputLayer(64, 56))

NN.add('conv1',G_convLayer(64,64,56, 3,no_g=1,strd=1))

NN.add('conv2',G_convLayer(64,64,56, 3,no_g=2,strd=1))

NN.add('conv3',G_convLayer(64, 64, 56, 3,no_g=4,strd=1))

NN.add('conv4',G_convLayer(64, 64, 56, 3,no_g=8,strd=1))

NN.add('conv5',G_convLayer(64, 64, 56, 3,no_g=16,strd=1))

NN.add('conv6',G_convLayer(64, 64, 56, 3,no_g=32,strd=1))

NN.add('conv7',G_convLayer(64, 64, 56, 3,no_g=64,strd=1))



#Experimenting with an input size of 112*112
NN = Network('custom_net')

NN.set_input_layer(InputLayer(64, 112))

NN.add('conv1',G_convLayer(64,64,112, 3,no_g=1,strd=1))

NN.add('conv2',G_convLayer(64,64,112, 3,no_g=2,strd=1))

NN.add('conv3',G_convLayer(64, 64, 112, 3,no_g=4,strd=1))

NN.add('conv4',G_convLayer(64, 64, 112, 3,no_g=8,strd=1))

NN.add('conv5',G_convLayer(64, 64, 112, 3,no_g=16,strd=1))

NN.add('conv6',G_convLayer(64, 64, 112, 3,no_g=32,strd=1))

NN.add('conv7',G_convLayer(64, 64, 112, 3,no_g=64,strd=1))


#Experimenting with an input size of 224*224

NN = Network('custom_net')

NN.set_input_layer(InputLayer(64, 224))

NN.add('conv1',G_convLayer(64,64,224, 3,no_g=1,strd=1))

NN.add('conv2',G_convLayer(64,64,224, 3,no_g=2,strd=1))

NN.add('conv3',G_convLayer(64, 64, 224, 3,no_g=4,strd=1))

NN.add('conv4',G_convLayer(64, 64, 224, 3,no_g=8,strd=1))

NN.add('conv5',G_convLayer(64, 64, 224, 3,no_g=16,strd=1))

NN.add('conv6',G_convLayer(64, 64, 224, 3,no_g=32,strd=1))

NN.add('conv7',G_convLayer(64, 64, 224, 3,no_g=64,strd=1))

'''

'''
#Experimenting for variations in input number of channels

#Experimenting with an input size of 56

NN = Network('custom_net')

NN.set_input_layer(InputLayer(128, 56))

NN.add('conv_1',G_convLayer(128,128,56, 3,no_g=1,strd=1))

NN.add('conv_2',G_convLayer(128,128,56, 3,no_g=2,strd=1))

NN.add('conv_3',G_convLayer(128, 128, 56, 3,no_g=4,strd=1))

NN.add('conv_4',G_convLayer(128, 128, 56, 3,no_g=8,strd=1))

NN.add('conv_5',G_convLayer(128, 128, 56, 3,no_g=16,strd=1))

NN.add('conv_6',G_convLayer(128, 128, 56, 3,no_g=32,strd=1))

NN.add('conv_7',G_convLayer(128, 128, 56, 3,no_g=64,strd=1))

NN.add('conv_8',G_convLayer(128, 128, 56, 3,no_g=128,strd=1))
'''

'''
NN = Network('custom_net')

NN.set_input_layer(InputLayer(256, 56))

NN.add('256_conv_1',G_convLayer(256,256,56, 3,no_g=1,strd=1))

NN.add('256_conv_2',G_convLayer(256,256,56, 3,no_g=2,strd=1))

NN.add('256_conv_3',G_convLayer(256, 256, 56, 3,no_g=4,strd=1))

NN.add('256_conv_4',G_convLayer(256, 256, 56, 3,no_g=8,strd=1))

NN.add('256_conv_5',G_convLayer(256, 256, 56, 3,no_g=16,strd=1))

NN.add('256_conv_6',G_convLayer(256, 256, 56, 3,no_g=32,strd=1))

NN.add('256_conv_7',G_convLayer(256, 256, 56, 3,no_g=64,strd=1))

NN.add('256_conv_8',G_convLayer(256, 256, 56, 3,no_g=128,strd=1))

NN.add('256_conv_9',G_convLayer(256, 256, 56, 3,no_g=256,strd=1))

'''
'''
NN = Network('custom_net')

NN.set_input_layer(InputLayer(512, 56))

NN.add('conv_1',G_convLayer(512,512,56, 3,no_g=1,strd=1))

NN.add('conv_2',G_convLayer(512,512,56, 3,no_g=2,strd=1))

NN.add('conv_3',G_convLayer(512, 512, 56, 3,no_g=4,strd=1))

NN.add('conv_4',G_convLayer(512, 512, 56, 3,no_g=8,strd=1))

NN.add('conv_5',G_convLayer(512, 512, 56, 3,no_g=16,strd=1))

NN.add('conv_6',G_convLayer(512, 512, 56, 3,no_g=32,strd=1))

NN.add('conv_7',G_convLayer(512, 512, 56, 3,no_g=64,strd=1))

NN.add('conv_8',G_convLayer(512, 512, 56, 3,no_g=128,strd=1))

NN.add('conv_9',G_convLayer(512, 512, 56, 3,no_g=256,strd=1))

NN.add('conv_90',G_convLayer(512, 512, 56, 3,no_g=512,strd=1))
'''

'''
#Experiment with ResNet like layers to see the interplay between channels and input dimension on PE utilization at a fixed G, This will be run for G=1 , G=64 and G=8

#G=64

NN = Network('custom_net')

NN.set_input_layer(InputLayer(64, 56))

NN.add('conv_1',G_convLayer(64,64, 56, 3,no_g=1,strd=1))

NN.add('conv_5',ConvLayer(64,128,28,1,strd=2))

NN.add('conv_2',G_convLayer(128,128, 28, 3,no_g=2,strd=1))

NN.add('conv_6',ConvLayer(128,256,14,1,strd=2))

NN.add('conv_3',G_convLayer(256, 256, 14, 3,no_g=4,strd=1))

NN.add('conv_7',ConvLayer(256,512,7,1,strd=2))

NN.add('conv_4',G_convLayer(512, 512, 7, 3,no_g=8,strd=1))

#G=1

NN = Network('custom_net')

NN.set_input_layer(InputLayer(64, 56))

NN.add('conv_1',G_convLayer(64,64, 56, 3,no_g=64,strd=1))

NN.add('conv_5',ConvLayer(64,128,28,1,strd=2))

NN.add('conv_2',G_convLayer(128,128, 28, 3,no_g=128,strd=1))

NN.add('conv_6',ConvLayer(128,256,14,1,strd=2))

NN.add('conv_3',G_convLayer(256, 256, 14, 3,no_g=256,strd=1))

NN.add('conv_7',ConvLayer(256,512,7,1,strd=2))

NN.add('conv_4',G_convLayer(512, 512, 7, 3,no_g=512,strd=1))

'''
#G=8

NN = Network('custom_net')

NN.set_input_layer(InputLayer(64, 56))

NN.add('conv_1',G_convLayer(64,64, 56, 3,no_g=8,strd=1))

NN.add('conv_5',ConvLayer(64,128,28,1,strd=2))

NN.add('conv_2',G_convLayer(128,128, 28, 3,no_g=16,strd=1))

NN.add('conv_6',ConvLayer(128,256,14,1,strd=2))

NN.add('conv_3',G_convLayer(256, 256, 14, 3,no_g=32,strd=1))

NN.add('conv_7',ConvLayer(256,512,7,1,strd=2))

NN.add('conv_4',G_convLayer(512, 512, 7, 3,no_g=64,strd=1))
