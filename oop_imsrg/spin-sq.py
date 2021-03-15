import numpy as np
#import tensorflow as tf
# tf.enable_v2_behavior()
import tensornetwork as tn
tn.set_default_backend("numpy") 

class TSpinSq(object):
    """Generate the vacuum S^2 operator."""

    def __init__(self, n_hole_states, n_particle_states, ref=[], s=0.5):
        """Class constructor. Instantiate PairingHamiltonian2B object.

        Arguments:

        n_hole_states -- number of holes states in the single particle basis
        n_particle_states -- number of particles states in the single particle basis

        Keyword arguments:

        ref -- the reference state. must match dimensions imposed by arugments (default: [1,1,1,1,0,0,0,0])
        d -- the energy level spacing (default: 1.0)
        g -- the pairing strength (default: 0.5)
        pb -- strength of the pair-breaking term (operates in double particle basis) (default: 0.0)"""

        self._s = s
        if ref == []:
            self._reference = np.append(np.ones(n_hole_states,dtype=np.float32),
                                        np.zeros(n_particle_states,dtype=np.float32))
        else:
            self._reference = np.asarray(ref,dtype=np.float32)

        self._holes = np.arange(n_hole_states, dtype=np.int32)
        self._n_sp_states = n_hole_states + n_particle_states
        self._particles = np.arange(n_hole_states,self.n_sp_states, dtype=np.int32)
        self._sp_basis = np.append(self.holes, self.particles)

        self._SS1B, self._SS2B = self.construct()
        self._E, self._f, self._G = self.normal_order()

    @property
    def s(self):
        """Returns:

        s -- particle spin."""
        return self._s


    @property
    def reference(self):
        """Returns:

        reference -- reference state (ground state)."""
        return self._reference

    @property
    def holes(self):
        """Returns:

        holes -- indices of hole states in single particle basis."""
        return self._holes

    @property
    def particles(self):
        """Returns:

        particles -- indices of particle states in single particle basis."""
        return self._particles

    @property
    def sp_basis(self):
        """Returns:

        sp_basis -- indices of full single particle basis."""
        return self._sp_basis

    @property
    def n_sp_states(self):
        """Returns:

        n_sp_states -- size of single-particle basis."""
        return self._n_sp_states

    @property
    def SS1B(self):
        """Returns:

        SS1B -- one-body (rank 2) tensor defined by construct()."""
        return self._SS1B

    @property
    def SS2B(self):
        """Returns:

        SS2B -- two-body (rank 4) tensor defined by construct()."""
        return self._H2B

    @property
    def E(self):
        """Returns:

        E -- zero-body (rank 0) tensor defined by normal_order()."""
        return self._E

    @property
    def f(self):
        """Returns:

        f -- one-body (rank 2) tensor defined by normal_order()."""
        return self._f

    @property
    def G(self):
        """Returns:

        G -- two-body (rank 4) tensor defined by normal_order()."""
        return self._G


    def delta1B(self, p,q):
        """Delta part of 1B in S^2

        Arguments:

        p,q -- indices in single particle basis
        """
        pp = np.floor_divide(p,2)
        qp = np.floor_divide(q,2)
        ps = 1 if p%2==0 else -1
        qs = 1 if q%2==0 else -1

        if pp == qp and ps == qs:
            return 1
        else:
            return 0

    def me2B(self, p,q,r,s):
        """Matrix element of 2B operator (i.e. S_i*S_j for i!=j) in S^2

        Arguments:

        p,q,r,s -- indices in single-particle basis"""

        pp = np.floor_divide(p,2)
        qp = np.floor_divide(q,2)
        rp = np.floor_divide(r,2)
        sp = np.floor_divide(s,2)

        ps = 1 if p%2==0 else -1
        qs = 1 if q%2==0 else -1
        rs = 1 if r%2==0 else -1
        ss = 1 if s%2==0 else -1

        spin = self.s
        sisj_matrix = np.array([[spin**2,   0,    0,     0],
                                [0,  -spin**2,  spin,     0],
                                [0,    spin, spin**2,     0],
                                [0,      0,    0, -spin**2]])

        sym1 = sisj_matrix[int(ps==qs),int(rs==ss)]
        sym2 = sisj_matrix[int(qs==ps),int(ss==rs)]
        asym1 = sisj_matrix[int(ps==qs),int(ss==rs)]
        asym2 = sisj_matrix[int(qs==ps),int(rs==ss)]

        me = 0.5*(int(pp==rp)*int(qp==sp)*(sym1+sym2) - int(pp==sp)*int(qp==rp)*(asym1+asym2))


        return me

    def construct(self):
        """Constructs the one- and two-body pieces of the pairing
        Hamiltonian.

        Returns:

        (H1B, -- one-body tensor elements (defined by one-body operator)
         H2B) -- two-body tensor elements (defined by two-body operator)"""


        bas1B = self.sp_basis # get the single particle basis

        # one body part of Hamiltonian is floor-division of basis index
        # matrix elements are (P-1) where P is energy level
        for p in bas1B:
            for q in bas1B:
                SS1B = self.s*(self.s+1)*self.delta1B(p,q)

        # two body part of Hamiltonian constructed from four indices
        # with non-zero elements defined by pairing term
        SS2B = np.zeros(np.ones(4, dtype=np.int32)*self.n_sp_states,dtype=np.float32)
        for p in bas1B:
            for q in bas1B:
                for r in bas1B:
                    for s in bas1B:
                        SS2B[p,q,r,s] += self.me2B(p,q,r,s)
                        SS2B[p,q,r,s] += self.me2B(p,q,r,s)

        return (SS1B, SS2B)

    def normal_order(self):
        """Normal-orders the pairing Hamiltonian.

        Returns:

        (E, -- zero-body piece
         f, -- one-body piece
         G) -- two-body piece"""

        bas1B = self.sp_basis # get the single particle basis
        H1B_t = self.SS1B.astype(np.float32)   # get the 1B tensor
        H2B_t = self.SS2B.astype(np.float32)   # get the 2B tensor
        holes = self.holes         # get hole states
        particles = self.particles # get particle states

        #net = TensorNetwork()

        # - Calculate 0B piece
        H1B_holes = H1B_t[np.ix_(holes,holes)]
        H2B_holes = H2B_t[np.ix_(holes,holes,holes,holes)]
        
        ob_node0b = tn.Node(H1B_holes)
        tb_node0b = tn.Node(H2B_holes)

        ob_node0b[0] ^ ob_node0b[1]
        ob_contract = ob_node0b @ ob_node0b  # optimized contraction
        
        tb_node0b[0] ^ tb_node0b[2]
        tb_node0b[1] ^ tb_node0b[3]
        tb_contract = tb_node0b @ tb_node0b

        # ob_ii = tn.connect(ob_node0b[0],ob_node0b[1])
        # tb_ijij1 = tn.connect(tb_node0b[0], tb_node0b[2])
        # tb_ijij2 = tn.connect(tb_node0b[1], tb_node0b[3])

        # flatten = tn.flatten_edges([tb_ijij1, tb_ijij2])
        # ob_contract = tn.contract(ob_ii).tensor#.tensor.numpy()
        # tb_contract = 0.5*tn.contract(flatten).tensor#.tensor.numpy()

        E = ob_contract.tensor + 0.5*tb_contract.tensor
        #E = E.astype(np.float32)

        # - Calculate 1B piece
        ob_node1b = tn.Node(H1B_t)
        tb_node1b = tn.Node(H2B_t[np.ix_(bas1B,holes,bas1B,holes)])

        tb_node1b[1] ^ tb_node1b[3]
        tb_contract = tb_node1b @ tb_node1b

        # tb_ihjh = tn.connect(tb_node1b[1], tb_node1b[3])
        # tb_contract = tn.contract(tb_ihjh)

        #f = ob_node1b.tensor.numpy() + tb_contract.tensor.numpy()
        f = ob_node1b.tensor + tb_contract.tensor
        #f = f.astype(np.float32)

        # - Calculate 2B piece
        G = H2B_t


        return (E, f, G)
