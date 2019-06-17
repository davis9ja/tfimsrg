### TODO (experimental branch)

* reconfigure code logic
  * X encapsulate normal ordered Hamiltonian and remove global variables
  * change variable names to make more sense when reading
  * X create method for generating particle/hole basis and defining reference state (this is for an easy way to change number of particles in G.S., etc.)
  * look into normal ordering procedure (could be written differently)
  * try object-oriented approach?
  * X pass particle/hole tensors to generator
  * configure ability to flow with n_holes =/= n_particles
* X optimize code execution (should it be this slow?) -- optimized for 4h/4p
* X update README with useful information about what the code does
* do some scaling tests; benchmark with reference code

* write IMSRG flow in TensorNetwork architecture