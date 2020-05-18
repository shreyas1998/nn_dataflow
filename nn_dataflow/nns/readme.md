# Creating Custom Neural networks

Adding custom networks is straightforward. The examples of network in nns directory can be seen for more details. Regarding the kinds of layers that are supported one can look at core/layer.py for layer interfaces and layer details. 
The additional layer interfaces that were added are as follows

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
In order to get G=1 , we set no_g =nifm 
In order to get G=2 , we set no_g =nifm/2
.
.
.
and so on.

For more information look at implementations of mobilenet.py, group_squeezenext.py for network examples and 
layer.py for information regardinglayer interfaces.