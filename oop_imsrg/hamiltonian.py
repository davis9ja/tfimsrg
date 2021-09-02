import numpy as np
import time
#import tensorflow as tf
# tf.enable_v2_behavior()
import tensornetwork as tn
tn.set_default_backend("numpy") 

from pyci.density_matrix.density_matrix import density_1b, density_2b, density_3b
#import pyci.imsrg_ci.pyci_p3h as pyci


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

    def __init__(self, n_hole_states, n_particle_states, ref=None, d=1.0, g=0.5, pb=0.0, dens_weights=None):
        """Class constructor. Instantiate PairingHamiltonian2B object.

        Arguments:

        n_hole_states -- number of holes states in the single particle basis
        n_particle_states -- number of particles states in the single particle basis

        Keyword arguments:

        ref -- the reference state. must match dimensions imposed by arugments (default: [1,1,1,1,0,0,0,0])
        d -- the energy level spacing (default: 1.0)
        g -- the pairing strength (default: 0.5)
        pb -- strength of the pair-breaking term (operates in double particle basis) (default: 0.0)"""

        print('Start Hamiltonian')
        self._d = d
        self._g = g
        self._pb = pb
        if ref is None:
            self._reference = np.append(np.ones(n_hole_states,dtype=np.float32),
                                        np.zeros(n_particle_states,dtype=np.float32))
        else:
            self._reference = np.asarray(ref,dtype=np.float32)

        where_h = np.where(self._reference >= 0.5)
        where_p = np.where(self._reference < 0.5)
        self._holes, self._particles = where_h[0], where_p[0]
        #print(self._holes, self._particles)
        #self._holes = np.arange(n_hole_states, dtype=np.int32)
        self._n_sp_states = n_hole_states + n_particle_states
        #self._particles = np.arange(n_hole_states,self.n_sp_states, dtype=np.int32)
        self._sp_basis = np.arange(n_hole_states+n_particle_states,dtype=np.int32)#np.append(self.holes, self.particles)

        self._H1B, self._H2B = self.construct()

        self._rho1b = None
        self._rho2b = None
        self._rho3b = None

        if dens_weights is None:
            self._E, self._f, self._G = self.normal_order()
        else:
            self._rho1b, self._rho2b, self._rho3b, self._lambda2b, self._lambda3b = self.make_densities(dens_weights)

            self._E, self._f, self._G = self.normal_order_slow(self._rho1b,self._rho2b)
            
        self._dens_weights = dens_weights

        # E, f, G = self.normal_order()
        # rho1b, rho2b, rho3b, lambda2b, lambda3b = self.make_densities(dens_weights)
        # E1, f1, G1 = self.normal_order_slow(rho1b,rho2b)

        # print(np.array_equal(f, f1), np.array_equal(G, G1), E, E1)

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

    def make_densities(self, dens_weights):

        holes = self.holes
        particles = self.particles
        bas1B = self.sp_basis

        ti = time.time()
        rho1b = density_1b(len(holes), len(particles), weights=dens_weights)
        tf = time.time()
        print('generated 1b density in {: .4f} seconds'.format(tf-ti))

        ti = time.time()
        rho2b = density_2b(len(holes), len(particles), weights=dens_weights)
        tf = time.time()
        print('generated 2b density in {: .4f} seconds'.format(tf-ti))

        ti = time.time()
        rho3b = density_3b(len(holes), len(particles), weights=dens_weights)
        tf = time.time()
        print('generated 3b density in {: .4f} seconds'.format(tf-ti))

        

        ti = time.time()
        lambda2b = np.zeros_like(rho2b)
        for i in bas1B:
            for j in bas1B:
                for k in bas1B:
                    for l in bas1B:
                        lambda2b[i,j,k,l] += rho2b[i,j,k,l] - rho1b[i,k]*rho1b[j,l] + rho1b[i,l]*rho1b[j,k]
        tf = time.time()
        print('generated lambda2b in {: .4f} seconds'.format(tf-ti))
        
        ti = time.time()
        lambda3b = np.zeros_like(rho3b)
        for i in bas1B:
            for j in bas1B:
                for k in bas1B:
                    for l in bas1B:
                        for m in bas1B:
                            for n in bas1B:
                                lambda3b[i,j,k,l,m,n] += rho3b[i,j,k,l,m,n] - rho1b[i,l]*lambda2b[j,k,m,n] -\
                                                         rho1b[j,m]*lambda2b[i,k,l,n] - rho1b[k,n]*lambda2b[i,j,l,m] +\
                                                         rho1b[i,m]*lambda2b[j,k,l,n] + rho1b[i,n]*lambda2b[j,k,m,l] +\
                                                         rho1b[j,l]*lambda2b[i,k,m,n] + rho1b[j,n]*lambda2b[i,k,l,m] +\
                                                         rho1b[k,l]*lambda2b[i,j,n,m] + rho1b[k,m]*lambda2b[i,j,l,n] -\
                                                         rho1b[i,l]*rho1b[j,m]*rho1b[k,n] - rho1b[i,m]*rho1b[j,n]*rho1b[k,l] -\
                                                         rho1b[i,n]*rho1b[j,l]*rho1b[k,m] + rho1b[i,l]*rho1b[j,n]*rho1b[k,m] +\
                                                         rho1b[j,m]*rho1b[i,n]*rho1b[k,l] + rho1b[i,m]*rho1b[j,l]*rho1b[k,n]
        tf = time.time()
        print('generated lambda3b density in {: .4f} seconds'.format(tf-ti))
                
        

        return (rho1b, rho2b, rho3b, lambda2b, lambda3b)
        

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
                return -1
            if ps == ss and qs == rs:
                return 1

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
        G = np.copy(H2B_t)


        return (E, f, G)

    def normal_order_slow(self, rho1b, rho2b):

        bas1B = self.sp_basis # get the single particle basis
        H1B_t = self.H1B.astype(np.float32)   # get the 1B tensor
        H2B_t = self.H2B.astype(np.float32)   # get the 2B tensor
        holes = self.holes         # get hole states
        particles = self.particles # get particle states
        n_states = len(bas1B)
        
        d = self._d
        g = self._g
        pb = self._pb

        # hme = pyci.matrix(len(holes), len(particles), 0.0, H1B_t, H2B_t, H2B_t, imsrg=True)
        # w,v = np.linalg.eigh(hme)
        # v0 = v[:, 0]


        contract_1b = np.einsum('ij,ij->', rho1b, H1B_t)

        # rho_reshape_2b = np.reshape(rho2b, (n_states**2,n_states**2))
        # h2b_reshape_2b = np.reshape(H2B_t, (n_states**2,n_states**2))
        # contract_2b = np.einsum('ij,ij->', rho_reshape_2b, h2b_reshape_2b)
        contract_2b = np.einsum('ijkl,ijkl->', rho2b, H2B_t)
            
        E = contract_1b + 0.25*contract_2b
        
        f = H1B_t + np.einsum('piqj,ij->pq', H2B_t, rho1b) #0.5*(np.einsum('piqj,ij->pq', H2B_t, rho1b) - np.einsum('pijq,ij->pq', H2B_t, rho1b))

        G = H2B_t

        #print("NOH contract_1b, contract_2b, piqj", contract_1b, contract_2b, np.einsum('piqj,ij->pq', H2B_t, rho1b))

        return (E, f, G)
