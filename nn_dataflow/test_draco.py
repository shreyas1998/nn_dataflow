""" $lic$
Copyright (C) 2016-2019 by The Board of Trustees of Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

'''
This is an extension of the nn dataflow simulator with the added functionality of depthwise
and group convolution layers

All the test cases have been inspired from the original tests which can be found in nn_dataflow/tests 
directory.

All the results of the DRACO paper were obtained from test_eyeriss_isca16 and test_eyeriss_isscc16

test_eyeriss_isca16 gives us the energy breakdown of various memeory levels present in eyeriss- DRAM,
Global buffer, PE spads, Array(the network), and ALU.

test_eyeriss_isscc16 gives us Power, Processing Latency, Ops, Active PEs, Filter size.

Inorder to change the map strategy, relative memory costs and no. of PEs,Global buffer size
and PE spad size, change these parameters in the setUp method.

Inorder to change the network chosen for simulation change it in the test_eyeriss_isca16 and
test_eyeriss_isscc16 respectively based on the paramter that needs to be measured.

To define a custom network , one needs to add it to the nns directory in the virtual environment
or the location where the nn-dataflow package is installed.

For usage of group and depthwise convolution please check the existing implementations in the nns
directory in the nn-dataflow library (extended). For more information regarding the layer interface
one can look into layer.py in core directory of the nn-dataflow library (extended).

Usage-
python -m unittest test_draco.TestNNDataflow.test_eyeriss_isca16    

python -m unittest test_draco.TestNNDataflow.test_eyeriss_isscc16

Both output pickled files containing the relevant output.

'''



import unittest
import sys
import StringIO

from nn_dataflow.core import Cost
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer
from nn_dataflow.core import MapStrategy, MapStrategyEyeriss
from nn_dataflow.core import MemHierEnum as me
from nn_dataflow.core import Network
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import NNDataflow
from nn_dataflow.core import Option
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import Resource

from nn_dataflow.nns import import_network

from nn_dataflow.nns import alex_net

import matplotlib.pyplot as plt
import pickle

class TestNNDataflow(unittest.TestCase):
    ''' Tests for NNDataflow module. '''

    def setUp(self):

        self.alex_net = import_network('alex_net')
        self.vgg_net = import_network('vgg_net')
        self.dense_net= import_network('densenet121')
        self.squeeze_next= import_network('squeezenext')
        self.res_net= import_network('resnet101')
        self.mobile_net= import_network('mobilenet')
        self.g_squeeze_next= import_network('group_squeezenext')
        self.g_res_net= import_network('g_resnet101')
        self.custom_net= import_network('custom_net')

        net = Network('simple')
        net.set_input_layer(InputLayer(4, 2))
        net.add('1', ConvLayer(4, 4, 2, 1))
        net.add('2', ConvLayer(4, 4, 2, 1))
        # Two more layers to avoid single-segment case.
        net.add('a1', ConvLayer(4, 1, 1, 1, strd=2))
        net.add('a2', ConvLayer(1, 1, 1, 1))
        self.simple_net = net

        net = Network('complex')
        net.set_input_layer(InputLayer(8, 8))
        net.add('1', ConvLayer(8, 8, 8, 1))
        net.add('2a', ConvLayer(8, 8, 8, 1), prevs=('1',))
        net.add('3a', ConvLayer(8, 8, 8, 1))
        net.add('2b', ConvLayer(8, 8, 8, 1), prevs=('1',))
        net.add('3b', ConvLayer(8, 8, 8, 1))
        net.add('4', ConvLayer(16, 8, 8, 1), prevs=('3a', '3b'))
        self.complex_net = net

        self.map_strategy = MapStrategyEyeriss

        self.resource = Resource(proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                                        dim=PhyDim2(1, 1),
                                                        type=NodeRegion.PROC),
                                 dram_region=NodeRegion(
                                     origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                     type=NodeRegion.DRAM),
                                 src_data_region=NodeRegion(
                                     origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                     type=NodeRegion.DRAM),
                                 dst_data_region=NodeRegion(
                                     origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                     type=NodeRegion.DRAM),
                                 dim_array=PhyDim2(16, 16),   
                                 size_gbuf= 128 * 1024 // 2,  # 128 kB
                                 size_regf=512 // 2,  # 512 B
                                 array_bus_width=float('inf'),
                                 dram_bandwidth=float('inf'),
                                 no_time_mux=False,
                                )

        self.cost = Cost(mac_op=1,
                         mem_hier=(200, 6, 2, 1),
                         noc_hop=0,
                         idl_unit=0)

        self.options = Option()

    def test_invalid_network(self):
        ''' Invalid network argument. '''
        with self.assertRaisesRegexp(TypeError, 'NNDataflow: .*network.*'):
            _ = NNDataflow(self.alex_net.input_layer(), 4,
                           self.resource, self.cost, self.map_strategy)

    def test_invalid_resource(self):
        ''' Invalid network argument. '''
        with self.assertRaisesRegexp(TypeError, 'NNDataflow: .*resource.*'):
            _ = NNDataflow(self.alex_net, 4,
                           self.resource.proc_region, self.cost,
                           self.map_strategy)

    def test_invalid_cost(self):
        ''' Invalid network argument. '''
        with self.assertRaisesRegexp(TypeError, 'NNDataflow: .*cost.*'):
            _ = NNDataflow(self.alex_net, 4,
                           self.resource, self.cost._asdict(),
                           self.map_strategy)

    def test_invalid_map_strategy(self):
        ''' Invalid map_strategy argument. '''
        class _DummyClass(object):  # pylint: disable=too-few-public-methods
            pass

        with self.assertRaisesRegexp(TypeError, 'NNDataflow: .*map_strategy.*'):
            _ = NNDataflow(self.alex_net, 4,
                           self.resource, self.cost, _DummyClass)

    def test_verbose(self):
        ''' Verbose mode. '''
        network = self.alex_net

        batch_size = 16

        options = Option(sw_gbuf_bypass=(True, True, True),
                         sw_solve_loopblocking=True,
                         verbose=True)

        nnd = NNDataflow(network, batch_size, self.resource, self.cost,
                         self.map_strategy)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout = StringIO.StringIO()
        sys.stderr = stderr = StringIO.StringIO()

        tops, _ = nnd.schedule_search(options)

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        stdout_value = stdout.getvalue()
        stderr_value = stderr.getvalue()
        stdout.close()
        stderr.close()

        self.assertTrue(tops)

        self.assertFalse(stdout_value)
        for layer in network:
            self.assertIn(layer, stderr_value)

    def test_pipelining(self):
        ''' Pipelining. '''
        network = self.alex_net
        batch_size = 1

        options = Option(hw_gbuf_save_writeback=True,
                         partition_interlayer=True)
        nnd = NNDataflow(network, batch_size, self.resource, self.cost,
                         self.map_strategy)

        tops, _ = nnd.schedule_search(options)
        self.assertTrue(tops)

    def test_fast_forward_infeasible(self):
        ''' Enter fast forward due to infeasible constraint. '''
        network = self.simple_net
        batch_size = 1

        # Very small gbuf size. Small fmap tpart is infeasible.
        resource = self.resource._replace(
            dim_array=PhyDim2(2, 2),
            size_gbuf=16)

        options = Option(hw_gbuf_save_writeback=True,
                         partition_interlayer=True)
        nnd = NNDataflow(network, batch_size, resource, self.cost,
                         self.map_strategy)

        tops, _ = nnd.schedule_search(options)
        self.assertTrue(tops)

        # No pipelining is feasible.
        for dtfl in tops:
            self.assertTupleEqual(dtfl['1'].sched_seq, (0, 0, 0))
            self.assertTupleEqual(dtfl['2'].sched_seq, (1, 0, 0))

    def test_fast_forward_found(self):
        ''' Enter fast forward due to early found. '''
        network = self.simple_net
        batch_size = 1

        # No time overhead limit.
        options = Option(hw_gbuf_save_writeback=True,
                         partition_interlayer=True,
                         layer_pipeline_time_ovhd=float('inf'))
        nnd = NNDataflow(network, batch_size, self.resource, self.cost,
                         self.map_strategy)

        tops, _ = nnd.schedule_search(options)
        self.assertTrue(tops)

    def test_fast_forward_crit_time(self):
        ''' Enter fast forward due to long critical time. '''
        network = self.simple_net
        batch_size = 1

        # Multiple nodes for spatial pipelining.
        resource = self.resource._replace(
            proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                   dim=PhyDim2(8, 8),
                                   type=NodeRegion.PROC),
            dim_array=PhyDim2(1, 1),
        )

        # Very strict time overhead limit.
        # At large fmap tpart, utilization decreases and critical time would
        # increase.
        options = Option(hw_gbuf_save_writeback=True,
                         partition_interlayer=True,
                         layer_pipeline_time_ovhd=1e-3)
        nnd = NNDataflow(network, batch_size, resource, self.cost,
                         self.map_strategy)

        tops, _ = nnd.schedule_search(options)
        self.assertTrue(tops)

    def test_fast_forward_frontier(self):
        ''' Enter fast forward due to off-frontier. '''
        network = self.simple_net
        batch_size = 16

        # Multiple nodes for spatial pipelining.
        resource = self.resource._replace(
            proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                   dim=PhyDim2(8, 8),
                                   type=NodeRegion.PROC),
            dim_array=PhyDim2(2, 2),
        )

        # No time overhead limit.
        options = Option(hw_gbuf_save_writeback=True,
                         partition_interlayer=True,
                         layer_pipeline_time_ovhd=float('inf'))
        nnd = NNDataflow(network, batch_size, resource, self.cost,
                         self.map_strategy)

        tops, _ = nnd.schedule_search(options)
        self.assertTrue(tops)

    def test_fmap_fwd(self):
        '''
        Fmap forward with shared mem sources or both on/off-chip destinations.
        '''
        network = self.complex_net
        batch_size = 16

        # Multiple nodes for spatial pipelining.
        resource = self.resource._replace(
            proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                   dim=PhyDim2(8, 8),
                                   type=NodeRegion.PROC),
        )

        # No time overhead limit.
        options = Option(hw_gbuf_save_writeback=True,
                         partition_interlayer=True,
                         layer_pipeline_time_ovhd=float('inf'))
        nnd = NNDataflow(network, batch_size, resource, self.cost,
                         self.map_strategy)

        tops, _ = nnd.schedule_search(options)
        self.assertTrue(tops)

    def test_sched_instance_sharing(self):
        ''' Scheduling instance sharing between layers. '''
        network = self.alex_net
        batch_size = 1

        nnd = NNDataflow(network, batch_size, self.resource, self.cost,
                         self.map_strategy)

        self.assertIs(nnd.layer_sched_dict['conv1_a'],
                      nnd.layer_sched_dict['conv1_b'])
        self.assertIs(nnd.layer_sched_dict['conv2_a'],
                      nnd.layer_sched_dict['conv2_b'])
        self.assertIs(nnd.layer_sched_dict['pool1_a'],
                      nnd.layer_sched_dict['pool1_b'])

    def test_opt_goal(self):
        ''' Optimization goal. '''
        network = self.alex_net

        batch_size = 8

        resource = self.resource._replace(
            proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                   dim=PhyDim2(8, 8),
                                   type=NodeRegion.PROC)
        )

        nnd = NNDataflow(network, batch_size, resource, self.cost,
                         self.map_strategy)

        options_e = Option(sw_gbuf_bypass=(True, True, True),
                           sw_solve_loopblocking=True,
                           partition_hybrid=True,
                           partition_batch=True,
                           opt_goal='e',
                           ntops=16)
        tops_e, _ = nnd.schedule_search(options_e)
        self.assertTrue(tops_e)

        options_d = Option(sw_gbuf_bypass=(True, True, True),
                           sw_solve_loopblocking=True,
                           partition_hybrid=True,
                           partition_batch=True,
                           opt_goal='d',
                           ntops=16)
        tops_d, _ = nnd.schedule_search(options_d)
        self.assertTrue(tops_d)

        options_ed = Option(sw_gbuf_bypass=(True, True, True),
                            sw_solve_loopblocking=True,
                            partition_hybrid=True,
                            partition_batch=True,
                            opt_goal='ed',
                            ntops=16)
        tops_ed, _ = nnd.schedule_search(options_ed)
        self.assertTrue(tops_ed)

        self.assertLess(tops_e[0].total_cost, tops_d[0].total_cost)
        self.assertLess(tops_e[0].total_cost, tops_ed[0].total_cost)

        self.assertLess(tops_d[0].total_time, tops_e[0].total_time)
        self.assertLess(tops_d[0].total_time, tops_ed[0].total_time)

        # Sum of the smallest ED may not be the smallest; allow for error.
        self.assertLess(tops_ed[0].total_cost * tops_ed[0].total_time,
                        tops_e[0].total_cost * tops_e[0].total_time * 1.05)
        self.assertLess(tops_ed[0].total_cost * tops_ed[0].total_time,
                        tops_d[0].total_cost * tops_d[0].total_time * 1.05)

    def test_ext_layer(self):
        ''' With external layers. '''
        network = self.alex_net

        network.add_ext('e0', InputLayer(4, 1))
        network.add('l1', FCLayer(1000, 4))
        network.add('l2', FCLayer(8, 4), prevs=('e0', 'l1'))

        batch_size = 16

        options = Option(sw_gbuf_bypass=(True, True, True),
                         sw_solve_loopblocking=True)

        nnd = NNDataflow(network, batch_size, self.resource, self.cost,
                         self.map_strategy)

        tops, _ = nnd.schedule_search(options)

        self.assertTrue(tops)

    def test_no_valid_dataflow(self):
        ''' No valid dataflow is found. '''

        # Very small REGF.
        self.resource = Resource(proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                                        dim=PhyDim2(4, 4),
                                                        type=NodeRegion.PROC),
                                 dram_region=NodeRegion(
                                     origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                     type=NodeRegion.DRAM),
                                 src_data_region=NodeRegion(
                                     origin=PhyDim2(0, 0), dim=PhyDim2(4, 4),
                                     type=NodeRegion.DRAM),
                                 dst_data_region=NodeRegion(
                                     origin=PhyDim2(0, 0), dim=PhyDim2(4, 4),
                                     type=NodeRegion.DRAM),
                                 dim_array=PhyDim2(16, 16),
                                 size_gbuf=128 * 1024 // 2,  # 128 kB
                                 size_regf=2,
                                 array_bus_width=float('inf'),
                                 dram_bandwidth=float('inf'),
                                 no_time_mux=False,
                                )

        nnd = NNDataflow(self.alex_net, 4, self.resource, self.cost,
                         self.map_strategy)
        tops, _ = nnd.schedule_search(self.options)

        self.assertFalse(tops)

        # With inter-layer pipelining.
        options = Option(hw_gbuf_save_writeback=True,
                         partition_interlayer=True)
        tops, _ = nnd.schedule_search(options)

        self.assertFalse(tops)

    def test_scheduling_failure(self):
        ''' Layer scheduling failure. '''
        network = self.alex_net

        batch_size = 16

        nnd = NNDataflow(network, batch_size, self.resource, self.cost,
                         MapStrategy)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout = StringIO.StringIO()
        sys.stderr = stderr = StringIO.StringIO()

        with self.assertRaises(NotImplementedError):
            _ = nnd.schedule_search(self.options)

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        stdout_value = stdout.getvalue()
        stderr_value = stderr.getvalue()
        stdout.close()
        stderr.close()

        self.assertFalse(stdout_value)
        self.assertIn('Failed', stderr_value)

    def test_eyeriss_isca16(self): #currently modified for the resnet global thing
        '''
        Reproduce Eyeriss ISCA'16 paper Fig. 10.
        '''

        network1= self.alex_net
        network2= self.res_net
        network3 = self.dense_net
        network4= self.squeeze_next
        network5= self.mobile_net
        network6= self.g_squeeze_next
        network7= self.g_res_net
        network= self.custom_net
        batch_size = 1

        nnd = NNDataflow(network, batch_size, self.resource, self.cost,
                         self.map_strategy)
        #runs with batch_size and network as defined in the current method. The map_strategy,
        #cost and resource are as defined in the setUp method (see class definition)

        tops, _ = nnd.schedule_search(self.options)
        self.assertTrue(tops)
        dfsch = tops[0]

        # Results as cost for each component:
        header = 'ALU, DRAM, Buffer, Array, RF'
        cost_bkdn = {}

        layer_list= [i for i in network.layer_dict.keys()]
        layer_list.remove('__INPUT__')

        for layer in layer_list:#['conv{}'.format(i) for i in range(1,15)]: #['bno1','bno2','bno3','bno4']:

                op_cost = 0
                access_cost = [0] * me.NUM

                for layer_part in network:
                    if not layer_part or not layer_part.startswith(layer):
                        continue
                    sr = dfsch[layer_part]
                    op_cost += sr.total_ops * self.cost.mac_op
                    access_cost = [ac + a * c for ac, a, c
                        in zip(access_cost, sr.total_accesses,
                                self.cost.mem_hier)]

                cost_bkdn[layer] = []

            # To 1e9.
                cost_bkdn[layer].append(op_cost / 1e9)
                cost_bkdn[layer].append(access_cost[me.DRAM] / 1e9)
                cost_bkdn[layer].append(access_cost[me.GBUF] / 1e9)
                cost_bkdn[layer].append(access_cost[me.ITCN] / 1e9)
                cost_bkdn[layer].append(access_cost[me.REGF] / 1e9)

        # Check the major parts: ALU, DRAM, RF.
        with open('custom_both_var_layerwise','wb') as pi:
            pickle.dump(cost_bkdn,pi,protocol= pickle.HIGHEST_PROTOCOL)
        print("pickling_memcost_done")

        for layer in cost_bkdn:
            print(str(layer)+"\n"+str(header)+"\n"+str(cost_bkdn[layer]))


    def test_eyeriss_isscc16(self): #done
        '''
        Reproduce Eyeriss ISSCC'16 paper Fig. 14.5.6, JSSC'17 paper Table V.
        '''
        network1 = self.alex_net
        network2= self.squeeze_next
        network3= self.res_net
        network= self.mobile_net
        network5= self.g_squeeze_next
        network6= self.g_res_net
        network7= self.dense_net
        network8= self.custom_net
        conv_list=[]


        layer_list= [i for i in network.layer_dict.keys()]

        

        resource = Resource(proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                                   dim=PhyDim2(1, 1),
                                                   type=NodeRegion.PROC),
                            dram_region=NodeRegion(
                                origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                type=NodeRegion.DRAM),
                            src_data_region=NodeRegion(
                                origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                type=NodeRegion.DRAM),
                            dst_data_region=NodeRegion(
                                origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                type=NodeRegion.DRAM),
                            dim_array=PhyDim2(12, 14),
                            size_gbuf=108 * 1024 //2,  # 108 kB     #have deleted that //2 factor and regf was 261
                            size_regf= 261,  # 225 + 12 + 24
                            array_bus_width=float('inf'),
                            dram_bandwidth=float('inf'),
                            no_time_mux=False,
                           )

        cost = Cost(mac_op=2e-12,
                    mem_hier=( 460e-12, 15e-12, 4e-12, 1e-12),  # pJ/16-b
                    noc_hop=0,
                    idl_unit=30e-3 / 200e6)  # 30 mW GBUF + REGF

        #nnd = NNDataflow(network, batch_size, resource, cost,
        #               self.map_strategy)

        batch_size = 1
        nnd= NNDataflow(network,batch_size, self.resource, self.cost,self.map_strategy)   #changeed from original case
        #runs with batch_size and network as defined in the current method. The map_strategy,
        #cost and resource are as defined in the setUp method (see class definition)

        tops, _ = nnd.schedule_search(self.options)
        self.assertTrue(tops)
        dfsch = tops[0]

        ## Check results.

        # Results as stats of the rows in the table.
        header = 'Power, Processing Latency, Ops, Active PEs, Filter size'
        stats = {}

        for layer in layer_list:
            onchip_cost = 0
            time = 0
            ops = 0
            fil_size = 0

            for layer_part in network:
                if(layer=='__INPUT__'):
                    continue
                if not layer_part or not layer_part.startswith(layer):
                    continue
                sr = dfsch[layer_part]
                onchip_cost += sr.total_cost \
                        - sr.total_accesses[me.DRAM] * cost.mem_hier[me.DRAM]
                time += sr.total_time
                ops += sr.total_ops

                #fil_size += network[layer_part].total_filter_size()

            if(time==0):
                continue
            power = onchip_cost / (time / 200e6) * 1e3  # mW
            active_pes = int(ops / time)

            stats[layer] = []
            stats[layer].append(power)
            stats[layer].append(time / 200.e3)  # cycles to ms
            stats[layer].append(ops / 1e6)  # to MOPs
            stats[layer].append(active_pes)
            #stats[layer].append(fil_size / 1e3)  # to k


            #print(str(layer)+"\n"+str(header)+"\n"+str(stats[layer])+"\n")
        with open('mobil_pe_dw','wb') as pi:
            pickle.dump(stats,pi,protocol= pickle.HIGHEST_PROTOCOL)
        print("power_pickling_done")

