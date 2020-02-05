import numpy as np
#import tensorflow as tf
# tf.enable_v2_behavior()
import tensornetwork as tn
tn.set_default_backend("numpy") 

class Hamiltonian(object):
    """Parent class for organization purposes. Ideally, all Hamiltonian
    classes should inherit from this class. In this way, AssertionErrors
    can be handled in a general way."""

    def construct():
        print("Function that constructs the Hamiltonian")

    def normal_order():
        print("Function that normal-orders the Hamiltonian")

class PairingHamiltonian2B(Hamiltonian):
    """Generate the two-body pairing Hamiltonian. Inherits from Hamiltonian."""

    def __init__(self, n_hole_states, n_particle_states, ref=None, d=1.0, g=0.5, pb=0.0):
        """Class constructor. Instantiate PairingHamiltonian2B object.

        Arguments:

        n_hole_states -- number of holes states in the single particle basis
        n_particle_states -- number of particles states in the single particle basis

        Keyword arguments:

        ref -- the reference state. must match dimensions imposed by arugments (default: [1,1,1,1,0,0,0,0])
        d -- the energy level spacing (default: 1.0)
        g -- the pairing strength (default: 0.5)
        pb -- strength of the pair-breaking term (operates in double particle basis) (default: 0.0)"""

        self._d = d
        self._g = g
        self._pb = pb
        if ref == None:
            self._reference = np.append(np.ones(n_hole_states,dtype=np.float32),
                                        np.zeros(n_particle_states,dtype=np.float32))
        else:
            self._reference = np.asarray(ref,dtype=np.float32)

        self._holes = np.arange(n_hole_states, dtype=np.int32)
        self._n_sp_states = n_hole_states + n_particle_states
        self._particles = np.arange(n_hole_states,self.n_sp_states, dtype=np.int32)
        self._sp_basis = np.append(self.holes, self.particles)

        self._H1B, self._H2B = self.construct()
        self._E, self._f, self._G = self.normal_order()

    @property
    def d(self):
        """Returns:

        d -- energy level spacing."""
        return self._d

    @property
    def g(self):
        """Returns:

        g -- pairing strength."""
        return self._g

    @property
    def pb(self):
        """Returns:

        pb -- pair-breaking strength."""
        return self._pb

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
    def H1B(self):
        """Returns:

        H1B -- one-body (rank 2) tensor defined by construct()."""
        return self._H1B

    @property
    def H2B(self):
        """Returns:

        H2B -- two-body (rank 4) tensor defined by construct()."""
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


    def delta2B(self, p,q,r,s):
        """Determines if a two-body tensor elements should be zero,
        positive, or negative. This behavior is dictated by the pairing
        term in pairing Hamiltonian.

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

        if pp != qp or rp != sp:
            return 0
        if ps == qs or rs == ss:
            return 0
        if ps == rs and qs == ss:
            return -1
        if ps == ss and qs == rs:
            return 1

        return 0

    def deltaPB(self, p,q,r,s):
        """Determines if a two-body tensor elements should be zero,
        positive, or negative. This behavior is dictated by the pair-
        breaking term in pairing Hamiltonian.

        Arguments:

        p,q,r,s -- indices in single particle basis"""

        pp = np.floor_divide(p,2)
        qp = np.floor_divide(q,2)
        rp = np.floor_divide(r,2)
        sp = np.floor_divide(s,2)

        ps = 1 if p%2==0 else -1
        qs = 1 if q%2==0 else -1
        rs = 1 if r%2==0 else -1
        ss = 1 if s%2==0 else -1

        if (pp != qp and rp == sp) or (pp == qp and rp != sp):
            if ps == qs or rs == ss:
                return 0
            if ps == rs and qs == ss:
                return 1
            if ps == ss and qs == rs:
                return -1

        return 0

    def construct(self):
        """Constructs the one- and two-body pieces of the pairing
        Hamiltonian.

        Returns:

        (H1B, -- one-body tensor elements (defined by one-body operator)
         H2B) -- two-body tensor elements (defined by two-body operator)"""


        bas1B = self.sp_basis # get the single particle basis

        # one body part of Hamiltonian is floor-division of basis index
        # matrix elements are (P-1) where P is energy level
        H1B = np.diag(self.d*np.floor_divide(bas1B,2))

        # two body part of Hamiltonian constructed from four indices
        # with non-zero elements defined by pairing term
        H2B = np.zeros(np.ones(4, dtype=np.int32)*self.n_sp_states,dtype=np.float32)
        for p in bas1B:
            for q in bas1B:
                for r in bas1B:
                    for s in bas1B:
                        H2B[p,q,r,s] += self.g*0.5*self.delta2B(p,q,r,s)
                        H2B[p,q,r,s] += self.pb*0.5*self.deltaPB(p,q,r,s)

        return (H1B, H2B)

    def normal_order(self):
        """Normal-orders the pairing Hamiltonian.

        Returns:

        (E, -- zero-body piece
         f, -- one-body piece
         G) -- two-body piece"""

        bas1B = self.sp_basis # get the single particle basis
        H1B_t = self.H1B.astype(np.float32)   # get the 1B tensor
        H2B_t = self.H2B.astype(np.float32)   # get the 2B tensor
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

        E = ob_contract + tn.Node(0.5)*tb_contract
        #E = E.astype(np.float32)

        # - Calculate 1B piece
        ob_node1b = tn.Node(H1B_t)
        tb_node1b = tn.Node(H2B_t[np.ix_(bas1B,holes,bas1B,holes)])

        tb_node1b[1] ^ tb_node1b[3]
        tb_contract = tb_node1b @ tb_node1b

        # tb_ihjh = tn.connect(tb_node1b[1], tb_node1b[3])
        # tb_contract = tn.contract(tb_ihjh)

        #f = ob_node1b.tensor.numpy() + tb_contract.tensor.numpy()
        f = (ob_node1b + tb_contract)
        #f = f.astype(np.float32)

        # - Calculate 2B piece
        G = tn.Node(H2B_t)


        return (E, f, G)
