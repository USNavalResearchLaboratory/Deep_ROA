# DEEP ROA ESTIMATION (TORCH & LAVA)

## References
```
@article{scharzenberger2021deep_roa,
 title={{Learning To Estimate Regions Of Attraction Of Autonomous Dynamical Systems Using Physics-Informed Neural Networks}},
 author={Scharzenberger, Cody and Hays, Joe},
 journal={https://arxiv.org/abs/2111.09930},
 year={2021}
}
```
[Link to paper](https://arxiv.org/abs/2111.09930)

## [License](license.txt)

## Summary
This repository contains the code necessary to train a Deep ROA network to predict the boundary of the region of attraction (ROA) associated with a stable equilibrium point of a given autonomous dynamical system.  Our approach is best understood as an application of a modified physics-informed neural network (PINN) framework to a specific partial differential equation (PDE) (e.g., the Yuan-Li PDE) whose solution yields an implicit representation of the ROA boundary as time approaches infinity.  We refer to our framework as a "modified PINN" to emphasize the fact that we incorporate the standard PINN loss terms, such as the initial, boundary, and residual losses, as well as additional loss terms, including variational and monotonicity losses.  In the examples shown here we provide the system dynamics of interest directly in the form of differential equations, but this is not strictly necessary.  For example, the system dynamics could instead be represented by collected experimental data or a surrogate model, such as another neural network.

There are two different variations of our Deep ROA network provided in this repository.  The first is a non-spiking version implemented in Pytorch and located in the "ann" directory; the second is a spiking version implemented in Intel's Lava framework and is located in the "snn" directory.  While the principal of operation of these two Deep ROA implementations is the same, technical details such as the network architectures and requisite python libraries necessarily differ between each version.  These naunces are discussed briefly below.

## Install Instructions

### ANN Version
Simply pip install the "requirements.txt" file in the "ann" directory.

### SNN Version
Follow the instructions provided in the "python_environment_setup.txt" file in the "snn" directory.  These instructions are simple:

1. Install Lava per Intel's instructions: https://github.com/lava-nc/lava
2. Install Lava-DL per Intel's instructions: https://github.com/lava-nc/lava-dl


## Network Structure
The Deep ROA networks produced by this code are fully connected, feed-forward networks with a number of inputs equal to the dimension of the state space of the dynamical system being analyzed plus one for the single temporal variable and a single output representing the stability of the provided input states.  These networks have a number of hidden layers and associated widths as specified by the user.  The non-spiking Deep ROA implementation utilizies sigmoid activation functions on every layer, while the spiking version utilities an LIF model instead.  Input and output spike encoding and decoding is performed by the network itself by augmenting the input and output layers of the spiking Deep ROA network with additional synapse layers.  These layers allow the spiking Deep ROA network to learn an approriate encoding/decoding scheme during training.

When training is successful the network identifies stable states as those whose output is negative and unstable states as those whose output is positive.


## Repository Organization
This repository contains one example directory called "Closed_ROA" which itself contains the main script "main.py" as well as three subfolders: (1) Utilities, (2) Save, and (3) Load.

This repository contains two primary directories.  The first is the "ann" directory where the non-spiking Deep ROA networks are implemented, and the second is the "snn" directory where the spiking Deep ROA networks are implemented.  Each of these two main directories has nearly identical sub-directory structure, including six named example directories that utilitize the Deep ROA framework and a single utilities folder that implements the framework itself.

Since the non-spiking Deep ROA version is only dependent on Python modules that can be pip installed, an appropriate "requirements.txt" file is provided.  On the other hand, since the spiking Deep ROA version requires the use of Lava and Lava-DL, both of which have specific install instructions provided by Intel, a separate document titled "python_environment_setup.txt" is provied in lieu of a requirements file.

At the lowest level, the contents of all twelve example directories is the same.  Each contains a single "main.py" file that serves as the main file that is needed to run the example.  The accompanying "save" and "load" directories are provided so that the network has somewhere to save and load checkpoints, respectively.


### Utilities Directory
The Utilities folder contains the various classes required to implement this methodology.
These various classes implement the framework for training safety networks discussed above and do not themselves need to be edited. 

#### PINN Class
The deep roa class is the highest level Python wrapper that utilities all of the other lower level classes to build, train, and evaluate a safety network using the specified hyperparameters.  It encapsulates all of the functionality of the other classes and is the primary class that the user interacts with in their main script.

#### Project Options Class
The project options class contains all of the numerous parameters and settings that the user can select when building, training, and evaluating a safety network.  All of the parameters that directly impact the the safety network structure or training are contained in the hyperparameter class (see below), while the other non-essential options are left as attributes of the project options class.

#### Hyperparameters Class
The hyperparameters class contains all of the hyperparameters for building and training a safety network.  Changing any of these parameters will impact the quality of the resulting safety network ROA estimation.

#### Problem Specifications Class
The problem specifications class contains all of the parameters required to define the underlying dynamical system that is being analyzed, including the domain of analysis, the flow of the dynamical system, and the relevant initial and boundary conditions.

#### Neural Network Class
The network class contains all of the code necessary to implement out modified physics-informed network framework.  Specially, this class stores the network properties, such as its weights and biases, and implements the methods necessary to train and evaluate the network, such as the various loss functions used during training. 

#### PDE Class
The PDE class contains information related to the underlying dynamical system whose stable equilibrium we are analyzing.  This includes the explicit differential equations as well as methods for evaluating the flow of the system.

#### Initial-Boundary Condition Class
The initial-boundary condition class contains information related to the creation and application of the initial and boundary conditions that are relevant to the Yuan-Li PDE.

#### IBC / Residual / Variational Data Classes
The IBC, residual, and variational data classes contain information pertaining to the initial-boundary condition, residual, and variational training / testing data, respectively.

#### Finite Element Class
The finite element class contains the code necessary to implement the various finite element integration techniques that we use to evaluate the variational loss.  This includes such things as constructing finite elements and Gauss-Legendre integration.

#### Domain Class
The domain class simply contains information related to the domain over which we are analyzing the dynamical system. This includes such things as the limits of the domain as well as the spatiotemporal discretization. 

#### Utility Classes
In addition to the aforementioned classes, there are numerous classes whose names include the ``utilities'' keyword.  These classes do not themselves contain many properties but are instead organizational tools that collect many relevant, low level functions into a single utilities module.

The utilities classes are leverage by many of the previous classes to perform common basic functions.
Importantly, the model utilities class stores all of the possible dynamical systems of interest.
To study a new dynamical system, the associated system of first order ODEs would first need to be added to the model utilities class.

