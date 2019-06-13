# im-srg_tensorflow

#### PURPOSE:
The purpose of this code is to write the IM-SRG flow solution for the pairing model into the TensorFlow architecture. The motivation for this project is that the IM-SRG flow equation terms can be written as operations on tensors built from N-body interactions in the nuclear Hamiltonian. The TensorFlow library offers optimized and intuitive tools for performing tensor operations. We want to know if these methods can be used to efficiently solve the flow equations; if so, we would like to apply those methods to solve for three-body interactions.

#### FILES:
`testing_tensorflow_v2`: main notebook
`TODO.md`: information regarding future tasks or strategies to implement
`README.md`: overview of project (this file)

#### SUMMARY OF CODE:

__Cell 1:__ 
Import libraries and print TensorFlow version. This code was written in the API r1.13 (latest stable version at the time of writing).

__Cell 2:__ 
Build the one-body (1B) and two-body (2B) pieces of the Hamiltonian (H). The `H1B` is a rank 2 tensor (two indices), while the `H2B` is a rank 4 tensor (four indices). The pairing Hamiltonian is defined by the equation

$$H = \sum_{P\sigma}\delta(P-1)a^{\dagger}_{P\sigma} a_{P\sigma} - \sum_{pq}\frac{g}{2}a^{\dagger}_{p,+1}a^{\dagger}_{p,-1} a_{q,-1}a_{q,+1}.$$

Here we also build some useful tensors that represent occupation in the reference state. _We define our reference state as a Slater determinant with equal hole and particle states_. `occA` is a rank 4 tensor given by $n_a - n_b$ (two dummy indices). `occB` is a rank 4 tensor given by $1 - n_a - n_b$ (two dummy indices). `occC` is a rank 6 tensor given by $n_a n_b + (1 - n_a - n_b)n_c$ (three dummy indices). `occD` is a rank 4 tensor given by $n_a n_b (1 - n_c - n_d) + n_a n_b n_c n_d$ (no dummy indices). $n$ is a rank 1 tensor where $n_a = 1$ if $a$ is hole index, and $n_a = 0$ if $a$ is particle index.