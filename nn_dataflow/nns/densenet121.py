
from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, \
PoolingLayer, EltwiseLayer

'''Densenet 121'''

def add_dense_block(network,dense_id,gr_rate,no_layers_dlblock,bott_dim,nifm,_prevs):

    PREV1= _prevs
    den= 'dense_{}_'.format(dense_id)

    for i in range(no_layers_dlblock):

        network.add(den+'conv_{}_1'.format(i), ConvLayer(nifm+gr_rate*(i),4*gr_rate,bott_dim, 1, strd=1),
                prevs= PREV1)

        network.add(den+'conv_{}_3'.format(i), ConvLayer(4*gr_rate, gr_rate,bott_dim, 3, strd=1))

        #network.add(den+'concat_{}'.format(i), EltwiseLayer(nifm+gr_rate*(i+1),bott_dim,1),
        #        prevs= (PREV1,den+'conv_{}_3'.format(i),den+'conv_{}_3'))
        network.add(den+'concat_{}'.format(i), EltwiseLayer(nifm+gr_rate*(i+1),bott_dim,1),
                prevs= (PREV1,den+'conv_{}_3'.format(i)))

        PREV1= den+'concat_{}'.format(i)

    return(PREV1)
#missing batchnorm and relu in each of them

def add_trans_layer(network,trans_id,nifm_prev,gr_rate,no_dl_prev,bott_prev,_prev):
    trans= 'tran_{}_'.format(trans_id)
    network.add(trans+'conv', ConvLayer(nifm_prev+ gr_rate*no_dl_prev, nifm_prev*2,bott_prev, 1),prevs=_prev)

    #Maxpool or average pool?how to specify, or we don't need to worry about it

    pooled= network.add(trans+'pool', PoolingLayer(nifm_prev*2, bott_prev/2, 2, 2))

    return(trans+'pool')


NN= Network('Densenet')
NN.set_input_layer(InputLayer(3,224))


NN.add('conv1', ConvLayer(3, 64, 112, 7, 2))

NN.add('pool1', PoolingLayer(64, 56, 3, 2), prevs= ('conv1',))

_PREVS= 'pool1'


#db1
_PREVS= add_dense_block(network=NN, dense_id=1, gr_rate=32, no_layers_dlblock=6,
                                    bott_dim=56, nifm=64, _prevs=_PREVS )

_PREVS= add_trans_layer(network=NN,trans_id=1, nifm_prev=64, gr_rate=32, no_dl_prev=6,
                                    bott_prev=56, _prev=_PREVS)
#db2
_PREVS= add_dense_block(network=NN, dense_id=2, gr_rate=32, no_layers_dlblock=12,
                                    bott_dim=28, nifm=128,_prevs= _PREVS)

_PREVS= add_trans_layer(network=NN,trans_id=2, nifm_prev=128, gr_rate=32, no_dl_prev=12,
                                    bott_prev=28, _prev=_PREVS)
#db3
_PREVS= add_dense_block(network=NN, dense_id=3, gr_rate=32, no_layers_dlblock=24,
                                    bott_dim=14, nifm=256, _prevs=_PREVS )

_PREVS= add_trans_layer(network=NN,trans_id=3, nifm_prev=256, gr_rate=32, no_dl_prev=24,
                                    bott_prev=14, _prev=_PREVS)
#db4
_PREVS= add_dense_block(network=NN, dense_id=4, gr_rate=32, no_layers_dlblock=16,
                                    bott_dim=7, nifm=512, _prevs=_PREVS )

NN.add('pool6', PoolingLayer(1024, 1, 7),prevs=_PREVS)

NN.add('fc', FCLayer(1024, 1000))
