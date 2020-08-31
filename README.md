# im-srg_tensorflow

#### PURPOSE:
The purpose of this code is to write the IM-SRG flow solution for the pairing model into the TensorFlow architecture. The motivation for this project is that the IM-SRG flow equation terms can be written as operations on tensors built from N-body interactions in the nuclear Hamiltonian. The TensorFlow library offers optimized and intuitive tools for performing tensor operations. We want to know if these methods can be used to efficiently solve the flow equations; if so, we would like to apply those methods to solve for three-body interactions.

We solve the IMSRG flow in two architectures; 1) pure TensorFlow and 2) TensorNetwork, a library that uses TensorFlow as a backend.

#### HOW TO IMPORT PACKAGE:
For practical use, include these lines at the top of your Python code:

`sys.path.append('path/to/im-srg_tensorflow/')`
`from main import main`

#### RUN AN IM-SRG(2) FLOW:
The main() method implements the IM-SRG(2). To compute an IM-SRG(2) flow on the Pairing model with half-filling, with g=0.5, pb=0.0, 8 single particle states, standard reference state [1,1,1,1,0,0,0,0], and Wegner generator,
you only have to include this line:

`main(4,4)`

Consult the documentation for information on the keyword parameters provided by main().

#### FILES:
* `testing_tensorflow_v2.ipynb`: main notebook for TensorFlow implementation
* `IMSRG_tensornetwork.ipynb`: main notebook for TensorNetwork implementation
* `oop_imsrg/`: directory for object-oriented IM-SRG
  * hamiltonian: module defines pairing Hamiltonian
  * occupation_tensors: module defines the common occupation tensors used in the flow
  * generator: module defines the generator used in the flow
  * flow: module defines the IM-SRG flow structure (uses TensorNetwork architecture)
  * main: runs the IM-SRG flow for various parameters
  * plot_data: module defines plotting function used in main
* `TODO.md`: information regarding future tasks or strategies to implement
* `README.md`: overview of project (this file)

#### SUMMARY OF `oop_imsrg/`:

This directory contains all files for solving the IM-SRG using an object-oriented approach. The purpose of writing an object-oriented code is so that updates can be designed and implemented without major changes to the base IM-SRG(2) code that already works. The OOP code is "plug-and-play" in the sense that any class that inherits from Hamiltonian, Generator, and Flow can be used to run the code. In this way, algorithms that solve the IM-SRG flow in different ways (e.g. three-body truncation, different generator, etc.) can be encapsulated in their own classes without changes to the main method.

Documentation about each class and how they are used is found in `oop_imsrg/docs`.

#### SUMMARY OF `testing_tensorflow_v2.ipynb`:

__Cell 1:__
Import libraries and print TensorFlow version. This code was written in the API r1.13 (latest stable version at the time of writing).

__Cell 2:__
`def build_hamiltonian()` Build the one-body (1B) and two-body (2B) pieces of the Hamiltonian (H). The `H1B` is a rank 2 tensor (two indices), while the `H2B` is a rank 4 tensor (four indices). The pairing Hamiltonian is defined by the equation

![equation](https://latex.codecogs.com/gif.latex?%24%24H%20%3D%20%5Csum_%7BP%5Csigma%7D%5Cdelta%28P-1%29a%5E%7B%5Cdagger%7D_%7BP%5Csigma%7D%20a_%7BP%5Csigma%7D%20-%20%5Csum_%7Bpq%7D%5Cfrac%7Bg%7D%7B2%7Da%5E%7B%5Cdagger%7D_%7Bp%2C&plus;1%7Da%5E%7B%5Cdagger%7D_%7Bp%2C-1%7D%20a_%7Bq%2C-1%7Da_%7Bq%2C&plus;1%7D.%24%24).

Here we also build some useful tensors that represent occupation in the reference state. _We define our reference state as a Slater determinant with equal hole and particle states_. `occA` is a rank 4 tensor given by n_a - n_b (two dummy indices). `occB` is a rank 4 tensor given by 1 - n_a - n_b (two dummy indices). `occC` is a rank 6 tensor given by n_a n_b + (1 - n_a - n_b)n_c (three dummy indices). `occD` is a rank 4 tensor given by n_a n_b (1 - n_c - n_d) + n_a n_b n_c n_d (no dummy indices). n is a rank 1 tensor where n_a = 1 if _a_ is hole index, and n_a = 0 if _a_ is particle index.

__Cell 3:__
`def normal_order()` Normal-order the Hamiltonian and return numpy arrays that correspond to graph evaluations. The normal-ordered Hamiltonian consists of zero-, one-, and two-body pieces. `E` is a rank 0 tensor. `f` is rank 2 tensor. `G` is rank 4 tensor.

__Cell 4:__
`def white()` Disabled code for White's generator. Still in development.

__Cell 5:__
`def wegner()` Graph that calculates Wegner's generator and returns the results as numpy arrays. Find the equations in _An Advanced Course in Computational Nuclear Physics_ (Ch. 10, 2017).

__Cell 6:__
`def flow()` Graph that calculates the IM-SRG flow equations for `E`, `f`, and `G`. Returns the results as numpy arrays. Find the equations in _An Advanced Course in Computational Nuclear Physics_ (Ch. 10, 2017).

__Cell 7:__
`def derivative()` Defines the derivative function to pass into scipy.integrate.ode solver. The arguments _must_ be a dummy variable `t`, followed by a 1-D list `y` of all data to pass into the derivative and iterate. We need to shape `E`, `f`, `G` into a 1-D before passing into derivative, and then reconstruct them in order to pass into `wegner()` and `flow()`. Finally, we need to return another 0-D array. The proceeding cell defines functions for this "unraveling" and "raveling" process.

__Cell 8:__
`def unravel()` Reshape `E`, `f`, and `G` into 1-D lists and stick them together. Return the result.

`def ravel()` Reshape `y` into its original components `E`, `f`, and `G`.

__Cell 9:__
Main procedure that initializes variables and runs the scipy.integrate.ode solver. First, we build the Hamiltonian and retrieve all necessary components (1B and 2B tensors, reference state, holes, particles, and single-particle basis). Then, we construct the occupation tensors using information in the reference state. We normal order the Hamiltonian and pass the components into the scipy.integrate.ode solver, including any necessary additional arguments required by `derivative()`. The solver iterates until our scale parameter _s_ = 2, at which point the ground state energy has reached convergence.

#### SUMMARY OF `IMSRG2_tensornetwork.ipynb`:

__Cell 1:__
Import libraries. We use the ncon interface, provided in TensorNetwork, for calculating tensor contractions. Print out the TensorFlow version and TensorNetwork version.

__Cell 2:__
`def build_hamiltonian()`: Here we build the pairing Hamiltonian following the same logic as in `testing_tensorflow_v2`. The difference is that we have included delta functions `delta2B()` and `deltaPB()` that decide the value of the matrix element defined by the input single particle states. The `deltaPB()` function gives us matrix elements defined by a new pair-breaking/-creating term we have introduced to couple off-diagonal blocks of the Hamiltonian (e.g. 0p-0h couples to 1p-1h, 1p-1h couples to 2p-2h). This term is written as

![equation](https://latex.codecogs.com/gif.latex?f_\text{p-break}&space;=&space;\frac{f}{2}\sum_{p'\ne&space;p,q}\left(a^\dag_{p',&plus;1}&space;a^\dag_{p,-1}a_{q,-1}a_{q,&plus;1}&space;&plus;&space;a^\dag_{q,&plus;1}a^\dag_{q,-1}a_{p,-1}a_{p',&plus;1}&space;\right)).

__Cell 3:__
Functions that define occupation tensors in the same way as `testing_tensorflow_v2`. We provide the option to treat each occupation tensor as a rank equal to either the same number of indices that appear in the equation within the text, or a rank twice that number of indices.

__Cell 4:__
`def normal_order()`: We provide a function to normal order the pairing Hamiltonian that is generated from `build_hamiltonian()`. The normal ordering procedure is written in the canonical TensorNetwork architecture, rather than the ncon interface. This function could be rewritten with ncon, but this would not likely provide any discernable change in performance.

__Cell 5:__
`def wegner_tn()`: Wegner generator written in canonical TensorNetwork architecture. This function is still in development.

__Cell 6:__
`def wegner_ncon()`: Wegner generator written in ncon interface provided by TensorNetwork. This function calculates the tensor contractions, term-by-term, encoded in the Wegner's generator (see _An Advanced Course in Computational Nuclear Physics_, Ch. 10).

__Cell 7:__
`def flow()`: IM-SRG(2) flow written in ncon interface provided by TensorNetwork. Performs the same calculations as Cell 6 in `testing_tensorflow_v2`.

__Cell 8:__
`def derivative()`: Derivative to pass into scipy.integrate.ode.

__Cell 9:__
`def unravel()`: Transform E, f, and G tensors into 1D array so that data can be passed into scipy.integrate.ode.

`def ravel()`: Undo operation performed in `unravel()`, so that data can be stored and analyzed.

__Cell 10:__
`def main()`: Solves the IM-SRG(2) flow using scipy.ode.integrate. The flow equation is a sum of differential equations that describe how E, f, and G change with the scale parameter _s_. We choose convergence criterion as no change in energy within 10^-8 precision. We choose divergence criterion as a change in energy greater than 1.0; since change in energy should always decrease toward convergence, we can safely assume the flow has diverged in the event of such a large change in energy.

__Cell 11:__
This routine runs the flow for varying values of _g_ and _pb_. The energy spacing _d_ is kept fixed as the maximum value of _g_ and _pb_. Outputs from `main()` are stored in data arrays and dumped into binary files that can be directly loaded using the standard Python library pickle.

__Cell 12:__
This routine loads the binary files stored by Cell 11 and creates a plot of _g_ versus _pb_.

__Cell 13:__
Performance testing. Measures the time for one iteration of the flow.

__Cell 14:__
Performance testing. Measures the time for flow convergence with _g=0.5_ and _pb=0.0_.
