### TODO (experimental branch)

* reconfigure code logic
  * [x] encapsulate normal ordered Hamiltonian and remove global variables
  * [x] change variable names to make more sense when reading
  * [x] create method for generating particle/hole basis and defining reference state (this is for an easy way to change number of particles in G.S., etc.)
  * [x] look into normal ordering procedure (could be written differently)
  * [x] try object-oriented approach?
  * [x] pass particle/hole tensors to generator
  * [x] configure ability to flow with n_holes =/= n_particles
* [x] optimize code execution (should it be this slow?) -- optimized for 4h/4p
* [x] update README with useful information about what the code does
* [x] do some scaling tests; benchmark with reference code

* [x] write IMSRG flow in TensorNetwork architecture
  * [x] refactor TN code for intuitive user control (e.g. let user control coupling strength and level spacing)
  * [x] add pair breaking/creating term to Hamiltonian and include interactions in 1 pair and 0 pair blocks
    * [x] investigate where level crossing may occur for values of pb and g
      * [x] use exact diagonalization results to verify
    * [x] scan reference state configurations to find ground state for values of pb and g
  * [x] using TN to implement the IMSRG(3)

* [X] data for committee meeting
  * [X] get scaling data for various IMSRG(2) implementations
  * [X] benchmark IMSRG(2) results against full CI and Heiko's python code (check energy convergence discrepancy)

* [ ] test IMSRG(3)
  * [ ] TT decompose IMSRG(3)-relevant occupation tensors (should speed things up)
  * [ ] compare energy divergence to IMSRG(2)

# [ ] implement batch processing
  * [ ] modulate the flow equations (need more control over memory allocation)
  * [ ] might move away from NCON framework in favor of TensorNetwork syntax
  * [ ] need function to isolate batches, run data through flow equations, and bring them back together
