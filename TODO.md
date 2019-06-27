### TODO (experimental branch)

* reconfigure code logic
  * [x] encapsulate normal ordered Hamiltonian and remove global variables
  * [ ] change variable names to make more sense when reading
  * [x] create method for generating particle/hole basis and defining reference state (this is for an easy way to change number of particles in G.S., etc.)
  * [ ] look into normal ordering procedure (could be written differently)
  * [ ] try object-oriented approach?
  * [x] pass particle/hole tensors to generator
  * [ ] configure ability to flow with n_holes =/= n_particles
* [x] optimize code execution (should it be this slow?) -- optimized for 4h/4p
* [x] update README with useful information about what the code does
* [x] do some scaling tests; benchmark with reference code

* [x] write IMSRG flow in TensorNetwork architecture
  * [ ] refactor TN code for intuitive user control (e.g. let user control coupling strength and level spacing)
  * [ ] add pair breaking/creating term to Hamiltonian and include interactions in 1 pair and 0 pair blocks
  * [ ] using TN to implement the IMSRG(3)