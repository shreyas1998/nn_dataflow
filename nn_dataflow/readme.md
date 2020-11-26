# DRACO

[DRACO](https://arxiv.org/abs/2006.15103) is an extension of the nn-dataflow simulator capable of simulating group and depthwise convolutions. 
This is part of the paper (DRACO: Co-Optimizing Hardware Utilization, and
Performance of DNNs on Systolic Accelerator) which will be presented in [ISVLSI-20](http://www.isvlsi.org/).

## Installation

This has been tested on python 2.7.17.

Install [Microsoft visual c++ compiler for python 2.7](https://www.microsoft.com/en-us/download/details.aspx?id=44266)

Install matplotlib using [pip](https://pip.pypa.io/en/stable/).
Use [pip](https://pip.pypa.io/en/stable/) to install nn-dataflow. There is a newer version of nn-dataflow for python3, however at the time of development an older version was used.
```bash
pip install nn-dataflow==2.0
```

Once the simulator has been installed replace the core and nns directory in the library with the one given in the update branch of this repository.

## Testing
Use the test_draco.py to run tests. 

All the test cases have been inspired from the original tests which can be found in nn_dataflow/tests 
directory.

All the results of the DRACO paper were obtained from test_eyeriss_isca16 and test_eyeriss_isscc16

The test_eyeriss_isca16 method gives us the energy breakdown of various memeory levels present in eyeriss- DRAM, Global buffer, PE spads, Array(the network), and ALU.

The test_eyeriss_isscc16 method gives us Power, Processing Latency, Ops, Active PEs, Filter size.

In order to change the map strategy, relative memory costs and no. of PEs,Global buffer size and PE spad size, change these parameters in the setUp method.

In order to change the network chosen for simulation change it in the test_eyeriss_isca16 and
test_eyeriss_isscc16 respectively based on the parameter that needs to be measured.

To define a custom network , one needs to add it to the nns directory in the virtual environment
or the location where the nn-dataflow package is installed.

For usage of group and depthwise convolution please check the existing implementations in the nns
directory in the nn-dataflow library (extended). For more information regarding the layer interface
one can look into layer.py in core directory of the nn-dataflow library (extended).

Usage
```bash
python -m unittest test_draco.TestNNDataflow.test_eyeriss_isca16    

python -m unittest test_draco.TestNNDataflow.test_eyeriss_isscc16
```
Both output pickled files containing the relevant output.

