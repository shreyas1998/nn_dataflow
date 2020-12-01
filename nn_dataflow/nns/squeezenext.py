from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, \
        PoolingLayer, EltwiseLayer

'''
Squeezenext
'''

#This is according to the pytorch model, the specifications are not very clear in the paper

def sqznxt_blk(Network,nifm,nofm,inp_h,strd,no_sqz,_prevs):

    i= no_sqz
    redu=0.5
    if(strd==2):
        redu=1
    elif (nifm> nofm):
        redu=0.25

    Network.add('{}_sqz1_a'.format(i), ConvLayer(nifm,int(redu*nifm),inp_h,1,strd= strd),prevs=_prevs)
    Network.add('{}_sqz1_b'.format(i), ConvLayer(int(nifm*redu),int(nifm*redu*0.5),inp_h,1))

    Network.add('{}_sep3_w'.format(i), ConvLayer(int(nifm*redu*0.5),int(nifm*redu),inp_h,[3,1]))
    Network.add('{}_sep3_h'.format(i), ConvLayer(int(nifm*redu),int(nifm*redu),inp_h,[1,3]))

    Network.add('{}_exp1'.format(i), ConvLayer(int(nifm*redu),nofm,inp_h,1))

    if(strd==2 or (nifm!=nofm)):
        Network.add('{}_conv_br1'.format(i), ConvLayer(nifm,nofm,inp_h,1,strd= strd),prevs= _prevs)
        Network.add('{}_concat'.format(i), EltwiseLayer(nofm,inp_h,2),
           prevs=('{}_exp1'.format(i), '{}_conv_br1'.format(i)))
        #print("true"+" "+str(inp_h))
    else:
        #print(inp_h)
        Network.add('{}_concat'.format(i), EltwiseLayer(nofm,inp_h,2),
           prevs=(_prevs, '{}_exp1'.format(i)))
    #print("after"+" "+str(inp_h)+" "+str(nofm))

    return('{}_concat'.format(i))


NN= Network('squeeze_next')

num_bocks= [6,6,8,1]
strd_arr= [1,2,2,2]
num_blocks= [6,6,8,1]
nifm_arr= [64,32,64,128]
nofm_arr= [32,64,128,256]
inp_h_arr= [55,55,28,14]
#num_blocks is the number of blocks of squeezenets of a certain input resolution
#no_sqz is the number of squeezenet units in each block

NN.set_input_layer(InputLayer(3, 227))

NN.add('conv1', ConvLayer(3, 64, 111, 7, 2))
NN.add('pool1', PoolingLayer(64, 55, 3, 2))
prevs= 'pool1'

for no_bl in range(len(num_blocks)):
    strd= [strd_arr[no_bl]] + [1]*num_blocks[no_bl]
    nifm= nifm_arr[no_bl]
    nofm= nofm_arr[no_bl]
    inp_h= inp_h_arr[no_bl]

    for no_sqz in range(num_blocks[no_bl]):
        #print(str(inp_h)+str(strd[no_sqz]))
        if(strd[no_sqz])==2:
            inp_h= 1+((inp_h-1)/2)

        prevs= sqznxt_blk(Network= NN, nifm= nifm,nofm= nofm, inp_h= int(inp_h),strd= strd[no_sqz],
                            no_sqz= "bno"+str(no_bl+1)+"_"+str(no_sqz+1), _prevs=prevs)
        nifm= nofm
    #print("{}block_done".format(no_bl))
NN.add('conv2', ConvLayer(256, 128, 7, 1),prevs= prevs)
NN.add('pool2',PoolingLayer(128, 1, 7, 1))
NN.add('fc', FCLayer(128, 1000))
