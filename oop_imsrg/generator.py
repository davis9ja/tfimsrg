#import tensorflow as tf
# tf.enable_v2_behavior()
#import tensornetwork as tn
import numpy as np
from tfimsrg.oop_imsrg.hamiltonian import *
from tfimsrg.oop_imsrg.occupation_tensors import *
#tn.set_default_backend("tensorflow") 

from numba import jit

class Generator(object):
    """Parent class for organization purposes. Ideally, all Generator
    classes should inherit from this class. In this way, AssertionErrors
    can be handled in a general way."""
    
    def calc_eta():
        print("Function that calculates the generator")

class WegnerGenerator(Generator):
    """Calculate Wegner's generator for a normal ordered Hamiltonian.
       Truncate at two-body interactions."""

    def __init__(self, h, occ_t):
        """Class constructor. Instantiate WegnerGenerator object.

        Arguments:

        h -- Hamiltonian object (must be normal-ordered)
        occ_t -- OccupationTensor object"""

#        assert isinstance(h, Hamiltonian), "Arg 0 must be Hamiltonian object"
        assert isinstance(occ_t, OccupationTensors), "Arg 1 must be OccupationTensors object"

        self.f = h.f
        self.G = h.G


        self._holes = h.holes
        self._particles = h.particles

        self._occA = occ_t.occA
        self._occA4 = occ_t.occA4
        self._occB = occ_t.occB
        self._occB4 = occ_t.occB4
        self._occC = occ_t.occC
        self._occD = occ_t.occD

        ref = h.reference
        n = len(self._holes)+len(self._particles)
        Ga = tn.Node(np.transpose(np.append(ref[np.newaxis,:], np.zeros((1,n)), axis=0).astype(float)))
        Gb = tn.Node(np.append(ref[::-1][np.newaxis,:],np.zeros((1,n)), axis=0).astype(float))
        
        self._occRef1 = tn.ncon([Ga,Gb], [(-1,1),(1,-2)])                                                                    # n_a(1-n_b)
        self._occRef2 = tn.ncon([tn.Node(np.transpose(Gb.tensor)), tn.Node(np.transpose(Ga.tensor))], [(-1,1),(1,-2)])       # (1-n_a)n_b

        self._eta1B = np.zeros_like(self.f)
        self._eta2B = np.zeros_like(self.G)

    @property
    def eta1B(self):
        """Returns:

        eta1B -- one-body generator"""
        return self._eta1B

    @property
    def eta2B(self):
        """Returns:

        eta2B -- one-body generator"""
        return self._eta2B


    @property
    def f(self):
        """Returns:

        f -- one-body tensor elements (initialized by Hamiltonian object)"""
        return self._f

    @property
    def G(self):
        """Returns:

        f -- two-body tensor elements (initialized by Hamiltonian object)"""
        return self._G

    @f.setter
    def f(self, f):
        """Sets the one-body tensor."""
        self._f = f

    @G.setter
    def G(self, G):
        """Sets the two-body tensor."""
        self._G = G

    # @classmethod
    def decouple_OD(self):
        """Decouple the off-/diagonal elements from each other in
        the one- and two-body tensors. This procedure is outlined in
        An Advanced Course in Computation Nuclear Physics, Ch.10.

        Returns:

        (fd, -- diagonal part of f
         fod, -- off-diagonal part of f
         Gd, -- diagonal part of G
         God) -- off-diagonal part of G"""

        f = self.f
        G = self.G
        holes = self._holes
        particles = self._particles

        occRef1 = self._occRef1
        occRef2 = self._occRef2

        occD = self._occD
        occDt = tn.Node(np.transpose(occD.tensor,[2,3,0,1]))

        # - Decouple off-diagonal 1B and 2B pieces
 #       fod1 = tn.ncon([])
        fod = np.zeros(f.shape, dtype=np.float32)
        fod += np.multiply(occRef2.tensor, f)
        fod += np.multiply(occRef1.tensor, f)
        # fod[np.ix_(particles, holes)] += f[np.ix_(particles, holes)]
        # fod[np.ix_(holes, particles)] += f[np.ix_(holes, particles)]
        fd = f - fod

        God = np.zeros(G.shape, dtype=np.float32)
        God += np.multiply(occDt.tensor, G)
        God += np.multiply(occD.tensor, G)
        # God[np.ix_(particles, particles, holes, holes)] += G[np.ix_(particles, particles, holes, holes)]
        # God[np.ix_(holes, holes, particles, particles)] += G[np.ix_(holes, holes, particles, particles)]
        Gd = G - God

        return (fd, fod, Gd, God)

    # @classmethod
    def calc_eta(self):
        """Calculate the generator. The terms are defined in An
        Advanced Course in Computation Nuclear Physics, Ch.10.
        See also dx.doi.org/10.1016/j.physrep.2015.12.007

        Returns:

        (eta1B, -- one-body generator
         eta2B) -- two-body generator"""

        partition = self.decouple_OD()
        fd = partition[0].astype(np.float32)
        fod = partition[1].astype(np.float32)
        Gd = partition[2].astype(np.float32)
        God = partition[3].astype(np.float32)

        holes = self._holes
        particles = self._particles

        occA = self._occA
        occA4 = self._occA4
        occB = self._occB
        occB4 = self._occB4
        occC = self._occC


        # - Calculate 1B generator
        # first term
        sum1_1b_1 = tn.ncon([fd, fod], [(-1, 1), (1, -2)])#.numpy()
        sum1_1b_2 = np.transpose(sum1_1b_1)
        sum1_1b = sum1_1b_1 - sum1_1b_2

        # second term
        # sum2_1b_1 = tn.ncon([fd, occA, God], [(3,4), (3,4,1,2), (2,-1,1,-2)])#.numpy()
        # sum2_1b_2 = tn.ncon([fod, occA, Gd], [(3,4), (3,4,1,2), (2,-1,1,-2)])#.numpy()
        # sum2_1b = sum2_1b_1 - sum2_1b_2

        fdPrime = np.multiply(occA.tensor, fd)
        fodPrime = np.multiply(occA.tensor, fod)
        sum2_1b_1 = tn.ncon([fdPrime, God], [(1,2), (2,-1,1,-2)])
        sum2_1b_2 = tn.ncon([fodPrime, Gd], [(1,2), (2,-1,1,-2)])
        sum2_1b = sum2_1b_1 - sum2_1b_2
        
        # sum2_1b_1 = tn.ncon([fd, God], [(0, 1), (1, -1, 0, -2)])#.numpy()
        # sum2_1b_2 = tn.ncon([fod, Gd], [(0, 1), (1, -1, 0, -2)])#.numpy()
        # sum2_1b_3 = sum2_1b_1 - sum2_1b_2
        # sum2_1b = tn.ncon([occA, sum2_1b_3],[(-1, -2, 0, 1), (0,1)])#.numpy()

        # third term
        #sum3_1b_1 = tn.ncon([Gd, occC, God], [(6,-1,4,5), (4,5,6,1,2,3), (1,2,3,-2)])#.numpy()
        #sum3_1b = sum3_1b_1 - np.transpose(sum3_1b_1)
        
        sum3_1b_1 = np.multiply(occC.tensor, God)#np.multiply(tn.outer_product(tn.Node(occC), tn.Node(np.ones(8))).tensor, God)
        sum3_1b_2 = tn.ncon([Gd, God], [(3,-1,1,2),(1,2,3,-2)])
        sum3_1b = sum3_1b_2 - np.transpose(sum3_1b_2)

        # sum3_1b_1 = tn.ncon([occC, God], [(-1, -2, -3, 0, 1, 2), (0, 1, 2, -4)])#.numpy()
        # sum3_1b_2 = tn.ncon([Gd, sum3_1b_1], [(2, -1, 0, 1), (0, 1, 2, -2)])#.numpy()
        # sum3_1b_3 = np.transpose(sum3_1b_2)
        # sum3_1b = sum3_1b_2 - sum3_1b_3

        eta1B = sum1_1b + sum2_1b + 0.5*sum3_1b

        # - Calculate 2B generator
        # first term (P_ij piece)
        sum1_2b_1 = tn.ncon([fd, God], [(-1, 1), (1, -2, -3, -4)])#.numpy()
        sum1_2b_2 = tn.ncon([fod, Gd], [(-1, 1), (1, -2, -3, -4)])#.numpy()
        sum1_2b_3 = sum1_2b_1 - sum1_2b_2
        sum1_2b_4 = np.transpose(sum1_2b_3, [1, 0, 2, 3])
        sum1_2b_5 = sum1_2b_3 - sum1_2b_4

        # first term (P_kl piece)
        sum1_2b_6 = tn.ncon([fd, God], [(1, -3), (-1, -2, 1, -4)])#.numpy()
        sum1_2b_7 = tn.ncon([fod, Gd], [(1, -3), (-1, -2, 1, -4)])#.numpy()
        sum1_2b_8 = sum1_2b_6 - sum1_2b_7
        sum1_2b_9 = np.transpose(sum1_2b_8, [0, 1, 3, 2])
        sum1_2b_10 = sum1_2b_8 - sum1_2b_9

        sum1_2b = sum1_2b_5 - sum1_2b_10

        # second term
        #sum2_2b_1 = tn.ncon([Gd, occB, God], [(-1,-2,3,4), (3,4,1,2), (1,2,-3,-4)])#.numpy()
        #sum2_2b_2 = tn.ncon([God, occB, Gd], [(-1,-2,3,4), (3,4,1,2), (1,2,-3,-4)])#.numpy()

        
        GodPrime = np.multiply(occB4.tensor, God)
        GdPrime = np.multiply(occB4.tensor, Gd)
        sum2_2b_1 = tn.ncon([Gd, GodPrime], [(-1,-2,1,2), (1,2,-3,-4)])
        sum2_2b_2 = tn.ncon([God, GdPrime], [(-1,-2,1,2), (1,2,-3,-4)])

#        sum2_2b_1 = tn.ncon([Gd, occB[:,:,None],occB[None,:,:], God], [(-1,-2,4,5), (4,5,1), (1,2,3), (2,3,-3,-4)])#.numpy()
#        sum2_2b_2 = tn.ncon([God, occB[:,:,None],occB[None,:,:], Gd], [(-1,-2,4,5), (4,5,1), (1,2,3), (2,3,-3,-4)])#.numpy()

        sum2_2b = sum2_2b_1 - sum2_2b_2

        # sum2_2b_1 = tn.ncon([occB, God], [(-1, -2, 0, 1), (0, 1, -3, -4)])#.numpy()
        # sum2_2b_2 = tn.ncon([occB,  Gd], [(-1, -2, 0, 1), (0, 1, -3, -4)])#.numpy()
        # sum2_2b_3 = tn.ncon([Gd,  sum2_2b_1], [(-1, -2, 0, 1), (0, 1, -3, -4)])#.numpy()
        # sum2_2b_4 = tn.ncon([God, sum2_2b_2], [(-1, -2, 0, 1), (0, 1, -3, -4)])#.numpy()
        # sum2_2b = sum2_2b_3 - sum2_2b_4

        # third term
        # sum3_2b_1 = tn.ncon([Gd, occA, God], [(3,-1,4,-3), (3,4,1,2), (2,-2,1,-4)])#.numpy()
        # sum3_2b_2 = sum3_2b_1 - np.transpose(sum3_2b_1, [0,1,3,2])
        # sum3_2b = sum3_2b_2 - np.transpose(sum3_2b_2, [1,0,2,3])

        GodPrime = np.multiply(np.transpose(occA4.tensor, [0,2,1,3]), God)
        sum3_2b_1 = tn.ncon([Gd, GodPrime], [(2,-2,1,-4), (1,-1,2,-3)])
        sum3_2b_2 = sum3_2b_1 - np.transpose(sum3_2b_1, [0,1,3,2])
        sum3_2b = sum3_2b_2 - np.transpose(sum3_2b_2, [1,0,2,3])

        # sum3_2b_1 = tn.ncon([Gd, God], [(0, -1, 1, -3), (1, -2, 0, -4)])#.numpy()
        # sum3_2b_2 = np.transpose(sum3_2b_1, [1, 0, 2, 3])
        # sum3_2b_3 = np.transpose(sum3_2b_1, [0, 1, 3, 2])
        # sum3_2b_4 = np.transpose(sum3_2b_1, [1, 0, 3, 2])
        # sum3_2b_5 = sum3_2b_1 - sum3_2b_2 - sum3_2b_3 + sum3_2b_4
        # sum3_2b = tn.ncon([occA, sum3_2b_5], [(0, 1, -1, -2), (0, 1, -3, -4)])#.numpy()

        eta2B = sum1_2b + 0.5*sum2_2b - sum3_2b


        self._eta1B = eta1B
        self._eta2B = eta2B

        return (eta1B, eta2B)

class WegnerGenerator3B(WegnerGenerator):
    """Calculate Wegner's generator for a normal ordered Hamiltonian.
       Truncate at three-body interactions. Inherits from WegnerGenerator."""

    def __init__(self, h, occ_t):
        """Class constructor. Instantiate WegnerGenerator object.

        Arguments:

        h -- Hamiltonian object (must be normal-ordered)
        occ_t -- OccupationTensor object"""

        assert isinstance(h, Hamiltonian), "Arg 0 must be Hamiltonian object"
        assert isinstance(occ_t, OccupationTensors), "Arg 1 must be OccupationTensors object"

        self.f = h.f
        self.G = h.G
        self.W = np.zeros(h.n_sp_states*np.ones(6,dtype=np.int32),dtype=np.float32)

        self._holes = h.holes
        self._particles = h.particles

        self._occA = occ_t.occA
        self._occA4 = occ_t.occA4
        self._occB = occ_t.occB
        self._occB4 = occ_t.occB4
        self._occC = occ_t.occC
        self._occC6 = occ_t.occC6
        self._occD = occ_t.occD
        self._occDv2 = occ_t.occDv2
        self._occE = occ_t.occE
        self._occF = occ_t.occF
        self._occG = occ_t.occG
        self._occH = occ_t.occH
        self._occI = occ_t.occI
        self._occJ = occ_t.occJ

        ref = h.reference
        n = len(self._holes)+len(self._particles)
        Ga = tn.Node(np.transpose(np.append(ref[np.newaxis,:], np.zeros((1,n)), axis=0).astype(float)))
        Gb = tn.Node(np.append(ref[::-1][np.newaxis,:],np.zeros((1,n)), axis=0).astype(float))
        
        self._occRef1 = tn.ncon([Ga,Gb], [(-1,1),(1,-2)])                                                                    # n_a(1-n_b)
        self._occRef2 = tn.ncon([tn.Node(np.transpose(Gb.tensor)), tn.Node(np.transpose(Ga.tensor))], [(-1,1),(1,-2)])       # (1-n_a)n_b

        self._eta1B = np.zeros_like(self.f)
        self._eta2B = np.zeros_like(self.G)
        self._eta3B = np.zeros_like(self.W)


    @property
    def eta1B(self):
        """Returns:

        eta1B -- one-body generator"""
        return self._eta1B

    @property
    def eta2B(self):
        """Returns:

        eta2B -- one-body generator"""
        return self._eta2B

    @property
    def eta3B(self):
        """Returns:

        eta3B -- one-body generator"""
        return self._eta3B

    @property
    def W(self):
        """Returns:

        W -- three-body tensor elements (initialized by Hamiltonian object)"""
        return self._W

    @W.setter
    def W(self, W):
        """Sets the three-body tensor."""
        self._W = W

    # @classmethod
    def decouple_OD(self):
        """Inherits from WegnerGenerator.

        Decouple the off-/diagonal elements from each other in
        the one- and two-body tensors. This procedure is outlined in
        An Advanced Course in Computation Nuclear Physics, Ch.10.

        Returns:

        (fd, -- diagonal part of f
         fod, -- off-diagonal part of f
         Gd, -- diagonal part of G
         God, -- off-diagonal part of G
         Wd, -- diagonal part of W
         Wod) -- off-diagonal part of W"""

        partition = super().decouple_OD()
        fd = partition[0]
        fod = partition[1]
        Gd = partition[2]
        God = partition[3]
        
        W = self.W
        
        holes = self._holes
        particles = self._particles

        Wod = np.zeros(W.shape, dtype=np.float32)
        Wod[np.ix_(particles, particles, particles, holes, holes, holes)] += W[np.ix_(particles, particles, particles, holes, holes, holes)]
        Wod[np.ix_(holes, holes, holes, particles, particles, particles)] += W[np.ix_(holes, holes, holes, particles, particles, particles)]
        Wd = W - Wod

        return(fd, fod, Gd, God, Wd, Wod)

    # @classmethod
    def calc_eta(self):
        """Inherits from WegnerGenerator.

        Calculate the generator. See dx.doi.org/10.1016/j.physrep.2015.12.007,
        Appendix B, for three-body flow equations.

        Returns:

        (eta1B, -- one-body generator
         eta2B, -- two-body generator
         eta3B) -- three-body generator"""

        partition = self.decouple_OD()
        fd = partition[0]
        fod = partition[1]
        Gd = partition[2]
        God = partition[3]
        Wd = partition[4]
        Wod = partition[5]
        
        eta1B, eta2B = super().calc_eta()

        holes = self._holes
        particles = self._particles

        occA4 = self._occA4
        occA = self._occA
        #occA2 = self._occA2
        occB4 = self._occB4
        occC = self._occC
        occC6 = self._occC6
        occD = self._occD
        occDv2 = self._occDv2
        occF = self._occF
        occG = self._occG
        occH = self._occH
        occI = self._occI
        occJ = self._occJ


        # Calculate 1B generator
        # fourth term

        #occD_transpose = np.transpose(occDv2.tensor, [2,3,0,1])
        sum4_1b_1 = np.multiply(occDv2.tensor, Gd)
        sum4_1b_2 = np.multiply(occDv2.tensor, God)
        sum4_1b_3 = tn.ncon([Wod,  sum4_1b_1], [(3,4,-1,1,2,-2),(1,2,3,4)])#.numpy()
        sum4_1b_4 = tn.ncon([Wd,   sum4_1b_2], [(3,4,-1,1,2,-2),(1,2,3,4)])#.numpy()
        sum4_1b = sum4_1b_3 - sum4_1b_4

        # fifth term

        sum5_1b_1 = np.multiply(np.transpose(occF.tensor,[0,1,5,2,3,4]), Wd)
        sum5_1b_2 = np.multiply(np.transpose(occF.tensor,[0,1,5,2,3,4]), Wod)
        sum5_1b_3 = tn.ncon([sum5_1b_1, Wod],[(1,2,-1,3,4,5),(3,4,5,1,2,-2)])
        sum5_1b_4 = tn.ncon([sum5_1b_2, Wd],[(1,2,-1,3,4,5),(3,4,5,1,2,-2)])

        sum5_1b = sum5_1b_3 - sum5_1b_4


        # sum5_1b_1 = tn.ncon([Wd, occF, Wod], [(6,7,-1,8,9,10),
        #                                       (6,7,8,9,10,1,2,3,4,5),
        #                                       (3,4,5,1,2,-2)])#.numpy()
        # sum5_1b_2 = tn.ncon([Wod, occF, Wd], [(6,7,-1,8,9,10),
        #                                       (6,7,8,9,10,1,2,3,4,5),
        #                                       (3,4,5,1,2,-2)])#.numpy()
        # sum5_1b = sum5_1b_1 - sum5_1b_2

        # sum5_1b_1 = tn.ncon([occF, Wd.astype(np.float32)],
        #                  [(-1,-3,-4,-5,-6,0,1,2,3,4), (0,1,-2,2,3,4)])#.numpy()
        # sum5_1b_2 = tn.ncon([occF, Wod.astype(np.float32)],
        #                  [(-1,-3,-4,-5,-6,0,1,2,3,4), (0,1,-2,2,3,4)])#.numpy()
        # sum5_1b_3 = tn.ncon([sum5_1b_1, Wod.astype(np.float32)],
        #                  [(0,1,-1,2,3,4), (2,3,4,0,1,-2)])#.numpy()
        # sum5_1b_4 = tn.ncon([sum5_1b_2, Wd.astype(np.float32)],
        #                  [(0,1,-1,2,3,4), (2,3,4,0,1,-2)])#.numpy()
        # sum5_1b = sum5_1b_3 - sum5_1b_4

        eta1B += 0.25*sum4_1b + (1/12)*sum5_1b

        # Calculate 2B generator
        # fourth term
        sum4_2b_1 = np.matmul(-1.0*np.transpose(occA.tensor), fod)
        sum4_2b_2 = np.matmul(-1.0*np.transpose(occA.tensor),  fd)
        sum4_2b_3 = tn.ncon([Wd,  sum4_2b_1], [(1,-1,-2,2,-3,-4), (2,1)])#.numpy()
        sum4_2b_4 = tn.ncon([Wod, sum4_2b_2], [(1,-1,-2,2,-3,-4), (2,1)])#.numpy()
        sum4_2b = sum4_2b_3 - sum4_2b_4

        #fifth term

        sum5_2b_1 = np.multiply(np.transpose(occG.tensor, [1,2,0,3]), God)
        sum5_2b_2 = np.multiply(np.transpose(occG.tensor, [1,2,0,3]), Gd)

        sum5_2b_3 = tn.ncon([Wd, sum5_2b_1],[(1,-1,-2,2,3,-4),(2,3,1,-3)])
        sum5_2b_4 = tn.ncon([Wod, sum5_2b_2],[(1,-1,-2,2,3,-4),(2,3,1,-3)])

        sum5_2b_5 = sum5_2b_3 - sum5_2b_4

        sum5_2b = sum5_2b_5 - np.transpose(sum5_2b_5, [2,3,1,0]) - \
                    np.transpose(sum5_2b_5, [0,1,3,2]) + \
                    np.transpose(sum5_2b_5, [2,3,0,1])


        # sum5_2b_1 = tn.ncon([Wd, occG, God], [(4,-1,-2,5,6,-4),
        #                                       (4,5,6,1,2,3),
        #                                       (2,3,1,-3)])#.numpy()
        # sum5_2b_2 = tn.ncon([Wod, occG, Gd], [(4,-1,-2,5,6,-4),
        #                                       (4,5,6,1,2,3),
        #                                       (2,3,1,-3)])#.numpy()
        # sum5_2b = sum5_2b_2 - np.transpose(sum5_2b_2, [3,2,0,1]) - \
        #             np.transpose(sum5_2b_2, [0,1,3,2]) + \
        #             np.transpose(sum5_2b_2, [2,3,0,1])

        # sum5_2b_1 = tn.ncon([occG, God], [(-1,-2,-4,0,1,2), (1,2,0,-3)])#.numpy()
        # sum5_2b_2 = tn.ncon([occG,  Gd], [(-1,-2,-4,0,1,2), (1,2,0,-3)])#.numpy()
        # sum5_2b_3 = tn.ncon([Wd,  sum5_2b_1], [(0,-1,-2,1,2,-4), (1,2,0,-3)])#.numpy()
        # sum5_2b_4 = tn.ncon([Wod, sum5_2b_2], [(0,-1,-2,1,2,-4), (1,2,0,-3)])#.numpy()
        # sum5_2b_5 = sum5_2b_3 - sum5_2b_4
        # sum5_2b = sum5_2b_5 - np.transpose(sum5_2b_5, [3,2,0,1]) - \
        #             np.transpose(sum5_2b_5, [0,1,3,2]) + \
        #             np.transpose(sum5_2b_5, [2,3,0,1])

        #sixth term

        sum5_2b_1 = np.multiply(np.transpose(occH.tensor,[0,4,5,1,2,3]), Wd)
        sum5_2b_2 = np.multiply(np.transpose(occH.tensor,[0,4,5,1,2,3]), Wod)
        sum5_2b_3 = tn.ncon([sum5_2b_1, Wod], [(1,-1,-2,2,3,4),(2,3,4,1,-3,-4)])
        sum5_2b_4 = tn.ncon([Wd, sum5_2b_2], [(1,-1,-2,2,3,4),(2,3,4,1,-3,-4)])
        sum6_2b = sum5_2b_3 - sum5_2b_4

        # sum6_2b_1 = tn.ncon([Wd, occH, Wod], [(5,-1,-2,6,7,8),
        #                                       (5,6,7,8,1,2,3,4),
        #                                       (2,3,4,1,-3,-4)])#.numpy()
        # sum6_2b_2 = tn.ncon([Wd, occH, Wod], [(6,7,8,5,-3,-4),
        #                                       (5,6,7,8,1,2,3,4),
        #                                       (1,-1,-2,2,3,4)])#.numpy()
        # sum6_2b = sum6_2b_1 - sum6_2b_2

        # sum6_2b_1 = tn.ncon([occH, Wod], [(-1,-2,-3,-4,0,1,2,3),(1,2,3,0,-5,-6)])#.numpy()
        # sum6_2b_2 = tn.ncon([occH, Wod], [(-3,-4,-5,-6,0,1,2,3),(0,-1,-2,1,2,3)])#.numpy()
        # sum6_2b_3 = tn.ncon([Wd, sum6_2b_1], [(0,-1,-2,1,2,3), (1,2,3,0,-3,-4)])#.numpy()
        # sum6_2b_4 = tn.ncon([Wd, sum6_2b_2], [(1,2,3,0,-3,-4), (0,-1,-2,1,2,3)])#.numpy()
        # sum6_2b = sum6_2b_3 - sum6_2b_4

        #seventh term
        sum7_2b_1 = np.multiply(np.transpose(occI.tensor,[0,1,4,2,3,5]),Wd)
        sum7_2b_2 = tn.ncon([sum7_2b_1, Wod], [(1,2,-1,3,4,-4),(3,4,-2,1,2,-3)])
        sum7_2b = sum7_2b_2 - np.transpose(sum7_2b_2,[1,0,2,3]) - \
                              np.transpose(sum7_2b_2,[0,1,3,2]) + \
                              np.transpose(sum7_2b_2,[1,0,3,2])        


        # sum7_2b_1 = tn.ncon([Wd, occI, Wod], [(5,6,-1,7,8,-4),
        #                                       (5,6,7,8,1,2,3,4),
        #                                       (3,4,-2,1,2,-3)])#.numpy()
        # sum7_2b = sum7_2b_1 - np.transpose(sum7_2b_1,[1,0,2,3]) - \
        #                       np.transpose(sum7_2b_1,[0,1,3,2]) + \
        #                       np.transpose(sum7_2b_1,[1,0,3,2])

        # sum7_2b_1 = tn.ncon([occI, Wod], [(-1,-2,-3,-4,0,1,2,3), (2,3,-5,0,1,-6)])#.numpy()
        # sum7_2b_2 = tn.ncon([Wd, sum7_2b_1], [(0,1,-1,2,3,-4),(2,3,-2,0,1,-3)])#.numpy()
        # sum7_2b = sum7_2b_2 - np.transpose(sum7_2b_2,[1,0,2,3]) - \
        #                       np.transpose(sum7_2b_2,[0,1,3,2]) + \
        #                       np.transpose(sum7_2b_2,[1,0,3,2])

        eta2B += sum4_2b + 0.5*sum5_2b + (1/6)*sum6_2b + 0.25*sum7_2b

        # Calculate 3B generator
        #first, second, and third terms (one index contraction)

        #terms with P(i/jk) -- line 1 and 2
        sum1_3b_1 = tn.ncon([fd, Wod], [(-1,1), (1,-2,-3,-4,-5,-6)])#.numpy()
        sum1_3b_2 = tn.ncon([fod, Wd], [(-1,1), (1,-2,-3,-4,-5,-6)])#.numpy()
        sum1_3b_3 = sum1_3b_1 - sum1_3b_2
        sum1_3b_4 = sum1_3b_3 - np.transpose(sum1_3b_3, [1,0,2,3,4,5]) - \
                                np.transpose(sum1_3b_3, [2,1,0,3,4,5])

        #terms with P(l/mn) -- line 1 and 2
        sum1_3b_5 = tn.ncon([fd, Wod], [(1,-4), (-1,-2,-3,1,-5,-6)])#.numpy()
        sum1_3b_6 = tn.ncon([fod, Wd], [(1,-4), (-1,-2,-3,1,-5,-6)])#.numpy()
        sum1_3b_7 = sum1_3b_6 - sum1_3b_5
        sum1_3b_8 = sum1_3b_7 - np.transpose(sum1_3b_7, [0,1,2,4,3,5]) - \
                                np.transpose(sum1_3b_7, [0,1,2,5,4,3])

        #terms with P(ij/k)P(l/mn) -- line 3
        sum1_3b_9  = tn.ncon([Gd, God], [(-1,-2,-4,1),(1,-3,-5,-6)])#.numpy()
        sum1_3b_10 = tn.ncon([God, Gd], [(-1,-2,-4,1),(1,-3,-5,-6)])#.numpy()
        sum1_3b_11 = sum1_3b_9 - sum1_3b_10
        sum1_3b_12 = sum1_3b_11 - np.transpose(sum1_3b_11, [0,1,2,4,3,5]) - \
                                  np.transpose(sum1_3b_11, [0,1,2,5,4,3])
        sum1_3b_13 = sum1_3b_12 - np.transpose(sum1_3b_12, [2,1,0,3,4,5]) - \
                                  np.transpose(sum1_3b_12, [0,2,1,3,4,5])

        sum1_3b = sum1_3b_4 + sum1_3b_8 + sum1_3b_13

        #fourth term
        sum4_3b_1 = np.multiply(np.transpose(occB4.tensor,[2,3,0,1]), Gd)
        sum4_3b_2 = np.multiply(np.transpose(occB4.tensor,[2,3,0,1]), God)
        sum4_3b_3 = tn.ncon([sum4_3b_1, Wod], [(-1,-2,1,2),(1,2,-3,-4,-5,-6)])
        sum4_3b_4 = tn.ncon([sum4_3b_2, Wd], [(-1,-2,1,2),(1,2,-3,-4,-5,-6)])

        sum4_3b_5 = sum4_3b_3 - sum4_3b_4
        sum4_3b = sum4_3b_5 - np.transpose(sum4_3b_5, [1,0,2,3,4,5]) - \
                              np.transpose(sum4_3b_5, [2,1,0,3,4,5])

        # sum4_3b_1 = tn.ncon([Gd, occB4, Wod], [(-1,-2,3,4),(3,4,1,2),(1,2,-3,-4,-5,-6)])#.numpy()
        # sum4_3b_2 = tn.ncon([God, occB4, Wd], [(-1,-2,3,4),(3,4,1,2),(1,2,-3,-4,-5,-6)])#.numpy()
        # sum4_3b_3 = sum4_3b_1 - sum4_3b_2
        # sum4_3b = sum4_3b_3 - np.transpose(sum4_3b_3, [1,0,2,3,4,5]) - \
        #                       np.transpose(sum4_3b_3, [2,1,0,3,4,5])

        #fifth term
        sum5_3b_1 = np.multiply(occB4.tensor, Gd)
        sum5_3b_2 = np.multiply(occB4.tensor, God)
        sum5_3b_3 = tn.ncon([sum5_3b_1, Wod], [(1,2,-4,-5),(-1,-2,-3,1,2,-6)])
        sum5_3b_4 = tn.ncon([sum5_3b_2, Wd], [(1,2,-4,-5),(-1,-2,-3,1,2,-6)])
        sum5_3b_5 = sum5_3b_3 - sum5_3b_4
        sum5_3b = sum5_3b_5 - np.transpose(sum5_3b_5, [0,1,2,5,4,3]) - \
                              np.transpose(sum5_3b_5, [0,1,2,3,5,4])


        # sum5_3b_1 = tn.ncon([Gd, occB4, Wod], [(3,4,-4,-5),(3,4,1,2),(-1,-2,-3,1,2,-6)])#.numpy()
        # sum5_3b_2 = tn.ncon([God, occB4, Wd], [(3,4,-4,-5),(3,4,1,2),(-1,-2,-3,1,2,-6)])#.numpy()
        # sum5_3b_3 = sum5_3b_1 - sum5_3b_2
        # sum5_3b = sum5_3b_3 - np.transpose(sum5_3b_3, [0,1,2,5,4,3]) - \
        #                       np.transpose(sum5_3b_3, [0,1,2,3,5,4])

        #sixth term
        sum6_3b_1 = np.multiply(np.transpose(occA4.tensor,[1,2,0,3]), Gd)
        sum6_3b_2 = np.multiply(np.transpose(occA4.tensor,[1,2,0,3]), God)
        sum6_3b_3 = tn.ncon([sum6_3b_1,Wod], [(2,-1,1,-4),(1,-2,-3,2,-5,-6)])
        sum6_3b_4 = tn.ncon([sum6_3b_2, Wd], [(2,-1,1,-4),(1,-2,-3,2,-5,-6)])
        sum6_3b_5 = sum6_3b_3 - sum6_3b_4
        sum6_3b_6 = sum6_3b_5 - np.transpose(sum6_3b_5, [0,1,2,4,3,5]) - \
                                np.transpose(sum6_3b_5, [0,1,2,5,4,3])
        sum6_3b = sum6_3b_6 - np.transpose(sum6_3b_6, [1,0,2,3,4,5]) - \
                              np.transpose(sum6_3b_6, [2,1,0,3,4,5])


        # sum6_3b_1 = tn.ncon([Gd, occA4, Wod], [(4,-1,3,-4),(3,4,1,2),(1,-2,-3,2,-5,-6)])#.numpy()
        # sum6_3b_2 = tn.ncon([God, occA4, Wd], [(4,-1,3,-4),(3,4,1,2),(1,-2,-3,2,-5,-6)])#.numpy()
        # sum6_3b_3 = sum6_3b_1 - sum6_3b_2
        # sum6_3b_4 = sum6_3b_3 - np.transpose(sum6_3b_3, [0,1,2,4,3,5]) - \
        #                         np.transpose(sum6_3b_3, [0,1,2,5,4,3])
        # sum6_3b = sum6_3b_4 - np.transpose(sum6_3b_4, [1,0,2,3,4,5]) - \
        #                       np.transpose(sum6_3b_4, [2,1,0,3,4,5])

        #seventh term
        sum7_3b_1 = np.multiply(np.transpose(occJ.tensor,[3,4,5,0,1,2]), Wd)
        sum7_3b_2 = np.multiply(np.transpose(occJ.tensor,[3,4,5,0,1,2]), Wod)
        sum7_3b_3 = tn.ncon([sum7_3b_1,Wod], [(-1,-2,-3,1,2,3),(1,2,3,-4,-5,-6)])
        sum7_3b_4 = tn.ncon([sum7_3b_2, Wd], [(-1,-2,-3,1,2,3),(1,2,3,-4,-5,-6)])
        sum7_3b = sum7_3b_3 - sum7_3b_4


        # sum7_3b_1 = tn.ncon([Wd, occJ, Wod], [(-1,-2,-3,4,5,6), (4,5,6,1,2,3), (1,2,3,-4,-5,-6)])#.numpy()
        # sum7_3b_2 = tn.ncon([Wod, occJ, Wd], [(-1,-2,-3,4,5,6), (4,5,6,1,2,3), (1,2,3,-4,-5,-6)])#.numpy()
        # sum7_3b = sum7_3b_1 - sum7_3b_2

        #eighth term
        sum8_3b_1 = np.multiply(np.transpose(occC6.tensor,[0,1,3,2,4,5]), Wd)
        sum8_3b_2 = np.multiply(np.transpose(occC6.tensor,[2,3,4,0,1,5]), Wd)
        sum8_3b_3 = tn.ncon([sum8_3b_1, Wod], [(1,2,-3,3,-5,-6),(3,-1,-2,1,2,-4)])
        sum8_3b_4 = tn.ncon([sum8_3b_2, Wod], [(3,-2,-3,1,2,-6),(-1,1,2,-4,-5,3)])        
        sum8_3b_5 = sum8_3b_3 - sum8_3b_4
        sum8_3b_6 = sum8_3b_5 - np.transpose(sum8_3b_5, [0,1,2,4,3,5]) - \
                                np.transpose(sum8_3b_5, [0,1,2,5,4,3])
        sum8_3b = sum8_3b_6 - np.transpose(sum8_3b_6, [2,1,0,3,4,5]) - \
                              np.transpose(sum8_3b_6, [0,2,1,3,4,5])


        # sum8_3b_1 = tn.ncon([Wd, occC6, Wod], [(4,5,-3,6,-5,-6), (4,5,6,1,2,3), (3,-1,-2,1,2,-4)])#.numpy()
        # sum8_3b_2 = tn.ncon([Wd, occC6, Wod], [(6,-2,-3,4,5,-6), (4,5,6,1,2,3), (-1,1,2,-4,-5,3)])#.numpy()
        # sum8_3b_3 = sum8_3b_1 - sum8_3b_2
        # sum8_3b_4 = sum8_3b_3 - np.transpose(sum8_3b_3, [0,1,2,4,3,5]) - \
        #                         np.transpose(sum8_3b_3, [0,1,2,5,4,3])
        # sum8_3b = sum8_3b_4 - np.transpose(sum8_3b_4, [2,1,0,3,4,5]) - \
        #                       np.transpose(sum8_3b_4, [0,2,1,3,4,5])

        eta3B = sum1_3b + 0.5*sum4_3b + (-0.5)*sum5_3b + (-1.0)*sum6_3b + (1/6)*sum7_3b + 0.5*sum8_3b

        self._eta1B = eta1B
        self._eta2B = eta2B
        self._eta3B = eta3B

        return (eta1B, eta2B, eta3B)


class WhiteGenerator(Generator):
    """Calculate White's generator for a normal ordered Hamiltonian.
       This standard implemenation uses Epstein-Nesbet denominators."""

    
    def __init__(self, h):

#        assert isinstance(h, Hamiltonian), "Arg 0 must be Hamiltonian object"
        
        self._my_f = h.f
        self._my_G = h.G
        
        self._holes = h.holes
        self._particles = h.particles
        self._sp_basis = h.sp_basis

        self._reference = h.reference

        self._eta1B = np.zeros_like(self.f)
        self._eta2B = np.zeros_like(self.G)

    @property
    def eta1B(self):
        """Returns:

        eta1B -- one-body generator"""
        return self._eta1B

    @property
    def eta2B(self):
        """Returns:

        eta2B -- one-body generator"""
        return self._eta2B


    @property
    def f(self):
        """Returns:

        f -- one-body tensor elements (initialized by Hamiltonian object)"""
        return self._my_f

    @property
    def G(self):
        """Returns:

        f -- two-body tensor elements (initialized by Hamiltonian object)"""
        return self._my_G

    @f.setter
    def f(self, f):
        """Sets the one-body tensor."""
        self._my_f = f

    @G.setter
    def G(self, G):
        """Sets the two-body tensor."""
        self._my_G = G


    def calc_eta(self):

        bas1B = self._sp_basis
        holes = self._holes
        particles = self._particles
        ref = self._reference

        f = self.f
        G = self.G

        eta1B, eta2B = self._wrapper_calc_eta_generalized_ph(bas1B, holes, particles, f, G, ref)
        #eta1B, eta2B = self._wrapper_calc_eta(bas1B, holes, particles, f, G, ref)
        #print(np.array_equal(eta1B1, eta1B), np.array_equal(eta2B1, eta2B))

        self._eta1B = eta1B
        self._eta2B = eta2B

        # for i in bas1B:
        #     for j in bas1B:
        #         if eta1B[i,j] != eta1B1[i,j]:
        #             print(i,j, eta1B[i,j], eta1B1[i,j])
        #         for k in bas1B:
        #             for l in bas1B:
        #                 if eta2B[i,j,k,l] != eta2B1[i,j,k,l]:
        #                     print(i,j,k,l, eta2B[i,j,k,l],eta2B1[i,j,k,l])

        # assert np.array_equal(eta1B, -1*np.transpose(eta1B)), "1B not anti-sym"
        # assert np.array_equal(eta2B, -1*np.transpose(eta2B, [0,1,3,2])), "2B not anti-sym"

        return (eta1B, eta2B)

    @staticmethod
    @jit(nopython=True)
    def _wrapper_calc_eta(bas1B, holes, particles, f, G, n):

        # bas1B = self._sp_basis
        # holes = self._holes
        # particles = self._particles

        # f = self.f
        # G = self.G
        
        eta1B = np.zeros_like(f)
        eta2B = np.zeros_like(G)
        
        for a in particles:
            for i in holes:
                denom = f[a,a] - f[i,i] + G[a,i,a,i]

                if abs(denom)<1.0e-10:
                    result = 0.25 * np.pi * np.sign(f[a,i]) * np.sign(denom)
                else:
                    result = f[a,i] / denom

                eta1B[a,i] += result
                eta1B[i,a] += -result
                
                # if denom < 1:
                #     print('one body {}{},'.format(a, i), denom)
                # if a == 4 and i == 3:
                #     print(G[a,i,a,i])

        for a in particles:
            for b in particles:
                for i in holes:
                    for j in holes:
                        denom = (
                            f[a,a] + f[b,b] - f[i,i] - f[j,j]
                            + G[a,b,a,b] + G[i,j,i,j] - G[a,i,a,i]
                            - G[b,j,b,j] - G[a,j,a,j] - G[b,i,b,i]
                        )

                        if abs(denom)<1.0e-10:
                            result = 0.25 * np.pi * np.sign(G[a,b,i,j]) * np.sign(denom)
                        else:
                            result = G[a,b,i,j] / denom


                        
                        eta2B[a,b,i,j] += result
                        eta2B[i,j,a,b] += -result

                        # if denom < 1:
                        #    print('two body {}{}{}{},'.format(a,b,i,j), denom)
                        # if a == 5 and b == 4 and i == 3 and j ==2:
                        #     print(denom)
                        # print(eta2B[5,4,3,2])
                    
                    
        return (eta1B, eta2B)


    @staticmethod
    @jit(nopython=True)
    def _wrapper_calc_eta_generalized_ph(bas1B, holes, particles, f, G, n):

        # bas1B = self._sp_basis
        # holes = self._holes
        # particles = self._particles

        # f = self.f
        # G = self.G
        
        eta1B = np.zeros_like(f)
        eta2B = np.zeros_like(G)
        
        for a in bas1B:
            for i in bas1B:
                nbar_a = 1-n[a]
                n_i = n[i]
                denom = f[a,a]*nbar_a**2 - f[i,i]*n_i**2 + G[a,i,a,i]*nbar_a**2*n_i**2

                if abs(denom)<1.0e-10:
                    result = 0.25 * np.pi * np.sign(f[a,i]*nbar_a*n_i) * np.sign(denom)
                else:
                    result = f[a,i]*nbar_a*n_i / denom


                eta1B[a,i] += result
                eta1B[i,a] += -result
                
                # if abs(denom) < 0.001:
                #     print('one body',a,i, denom)
                # if a == 4 and i == 3:
                #     print(G[a,i,a,i])

        for a in bas1B:
            for b in bas1B:
                for i in bas1B:
                    for j in bas1B:
                        nbar_a = 1 - n[a]
                        nbar_b = 1 - n[b]
                        n_i = n[i]
                        n_j = n[j]

                        denom = (
                            f[a,a]*nbar_a**2 + f[b,b]*nbar_b**2 - f[i,i]*n_i**2 - f[j,j]*n_j**2
                            + G[a,b,a,b]*nbar_a**2*nbar_b**2 + G[i,j,i,j]*n_i**2*n_j**2 - G[a,i,a,i]*nbar_a**2*n_i**2
                            - G[b,j,b,j]*nbar_b**2*n_j**2 - G[a,j,a,j]*nbar_a**2*n_j**2 - G[b,i,b,i]*nbar_b**2*n_i**2
                        )

                        if abs(denom)<1.0e-10:
                            result = 0.25 * np.pi * np.sign(G[a,b,i,j]*nbar_a*nbar_b*n_i*n_j) * np.sign(denom)
                        else:
                            result = G[a,b,i,j]*nbar_a*nbar_b*n_i*n_j / denom


                        
                        eta2B[a,b,i,j] += result
                        eta2B[i,j,a,b] += -result

                        # if abs(denom) < 0.001:
                        #    print('two body ', a,b,i,j, denom)
                        # if a == 5 and b == 4 and i == 3 and j ==2:
                        #     print(denom)
                        # print(eta2B[5,4,3,2])
                    
                    
        return (eta1B, eta2B)

class WhiteGeneratorAtan(Generator):
    """Calculate White's generator for a normal ordered Hamiltonian.
       This implemenation uses Epstein-Nesbet denominator and regulates
       via the arctan."""

    
    def __init__(self, h):

#        assert isinstance(h, Hamiltonian), "Arg 0 must be Hamiltonian object"
        
        self._my_f = h.f
        self._my_G = h.G
        
        self._holes = h.holes
        self._particles = h.particles
        self._sp_basis = h.sp_basis

        self._reference = h.reference

        self._eta1B = np.zeros_like(self.f)
        self._eta2B = np.zeros_like(self.G)

    @property
    def eta1B(self):
        """Returns:

        eta1B -- one-body generator"""
        return self._eta1B

    @property
    def eta2B(self):
        """Returns:

        eta2B -- one-body generator"""
        return self._eta2B


    @property
    def f(self):
        """Returns:

        f -- one-body tensor elements (initialized by Hamiltonian object)"""
        return self._my_f

    @property
    def G(self):
        """Returns:

        f -- two-body tensor elements (initialized by Hamiltonian object)"""
        return self._my_G

    @f.setter
    def f(self, f):
        """Sets the one-body tensor."""
        self._my_f = f

    @G.setter
    def G(self, G):
        """Sets the two-body tensor."""
        self._my_G = G


    def calc_eta(self):

        bas1B = self._sp_basis
        holes = self._holes
        particles = self._particles
        ref = self._reference

        f = self.f
        G = self.G

        eta1B, eta2B = self._wrapper_calc_eta_generalized_ph(bas1B, holes, particles, f, G, ref)
        #eta1B, eta2B = self._wrapper_calc_eta(bas1B, holes, particles, f, G, ref)
        #print(np.array_equal(eta1B1, eta1B), np.array_equal(eta2B1, eta2B))

        self._eta1B = eta1B
        self._eta2B = eta2B

        # for i in bas1B:
        #     for j in bas1B:
        #         if eta1B[i,j] != eta1B1[i,j]:
        #             print(i,j, eta1B[i,j], eta1B1[i,j])
        #         for k in bas1B:
        #             for l in bas1B:
        #                 if eta2B[i,j,k,l] != eta2B1[i,j,k,l]:
        #                     print(i,j,k,l, eta2B[i,j,k,l],eta2B1[i,j,k,l])

        # assert np.array_equal(eta1B, -1*np.transpose(eta1B)), "1B not anti-sym"
        # assert np.array_equal(eta2B, -1*np.transpose(eta2B, [0,1,3,2])), "2B not anti-sym"

        return (eta1B, eta2B)

    @staticmethod
    @jit(nopython=True)
    def _wrapper_calc_eta_generalized_ph(bas1B, holes, particles, f, G, n):

        # bas1B = self._sp_basis
        # holes = self._holes
        # particles = self._particles

        # f = self.f
        # G = self.G
        
        eta1B = np.zeros_like(f)
        eta2B = np.zeros_like(G)
        
        for a in bas1B:
            for i in range(a):
                nbar_a = 1-n[a]
                n_i = n[i]
                denom = f[a,a]*nbar_a**2 - f[i,i]*n_i**2 + G[a,i,a,i]*nbar_a**2*n_i**2

                if denom != 0.0:
                    result = 0.5*np.arctan(2*f[a,i]*nbar_a*n_i / denom)
                else:
                    result = 0.0

                eta1B[a,i] += result
                eta1B[i,a] += -result
                
                # if abs(denom) < 0.001:
                #     print('one body',a,i, denom)
                # if a == 4 and i == 3:
                #     print(G[a,i,a,i])

        for b in bas1B:
            for a in bas1B:
                for j in bas1B:
                    for i in bas1B:
                        nbar_a = 1 - n[a]
                        nbar_b = 1 - n[b]
                        n_i = n[i]
                        n_j = n[j]

                        denom = (
                            f[a,a]*nbar_a**2 + f[b,b]*nbar_b**2 - f[i,i]*n_i**2 - f[j,j]*n_j**2
                            + G[a,b,a,b]*nbar_a**2*nbar_b**2 + G[i,j,i,j]*n_i**2*n_j**2 - G[a,i,a,i]*nbar_a**2*n_i**2
                            - G[b,j,b,j]*nbar_b**2*n_j**2 - G[a,j,a,j]*nbar_a**2*n_j**2 - G[b,i,b,i]*nbar_b**2*n_i**2
                        )

                        if denom != 0.0:
                            result = (1/8)*np.arctan(2*G[a,b,i,j]*nbar_a*nbar_b*n_i*n_j / denom)
                        else:
                            result = 0.0
                        
                        eta2B[a,b,i,j] += result
                        eta2B[i,j,a,b] += -result

                        # if abs(denom) < 0.001:
                        #    print('two body ', a,b,i,j, denom)
                        # if a == 5 and b == 4 and i == 3 and j ==2:
                        #     print(denom)
                        # print(eta2B[5,4,3,2])
                    
                    
        return (eta1B, eta2B)

        
class WhiteGeneratorMP3B(Generator):
    """Calculate White's generator for a normal ordered Hamiltonian.
       This "standard" implemenation uses Moller-Plesset denominators."""


    def __init__(self, h):

        assert isinstance(h, Hamiltonian), "Arg 0 must be Hamiltonian object"
        
        self.f = h.f
        self.G = h.G
        self.W = np.zeros(h.n_sp_states*np.ones(6,dtype=np.int32),dtype=np.float32)
        print('here',np.zeros(h.n_sp_states*np.ones(6,dtype=np.int32),dtype=np.float32).shape)
        self._holes = h.holes
        self._particles = h.particles
        self._sp_basis = h.sp_basis
        self._eta1B = np.zeros_like(self.f)
        self._eta2B = np.zeros_like(self.G)
        self._eta3B = np.zeros_like(self.W)

    @property
    def eta1B(self):
        """Returns:

        eta1B -- one-body generator"""
        return self._eta1B

    @property
    def eta2B(self):
        """Returns:

        eta2B -- one-body generator"""
        return self._eta2B

    @property
    def eta3B(self):
        """Returns:

        eta2B -- three-body generator"""
        return self._eta3B


    @property
    def f(self):
        """Returns:

        f -- one-body tensor elements (initialized by Hamiltonian object)"""
        return self._f

    @property
    def G(self):
        """Returns:

        G -- two-body tensor elements (initialized by Hamiltonian object)"""
        return self._G

    @property
    def W(self):
        """Returns:

        W -- three-body tensor elements (initialized by Hamiltonian object)"""
        return self._W

    @f.setter
    def f(self, f):
        """Sets the one-body tensor."""
        self._f = f

    @G.setter
    def G(self, G):
        """Sets the two-body tensor."""
        self._G = G

    @W.setter
    def W(self, W):
        """Sets the two-body tensor."""
        self._W = W


    def calc_eta(self):

        bas1B = self._sp_basis
        holes = self._holes
        particles = self._particles

        f = self.f
        G = self.G
        W = self.W
        
        eta1B, eta2B, eta3B = self._wrapper_calc_eta(bas1B, holes, particles, f, G, W)
        self._eta1B = eta1B
        self._eta2B = eta2B
        self._eta3B = eta3B
        
        return (eta1B, eta2B, eta3B)

    @staticmethod
    @jit(nopython=True)
    def _wrapper_calc_eta(bas1B, holes, particles, f, G, W):

        # bas1B = self._sp_basis
        # holes = self._holes
        # particles = self._particles

        # f = self.f
        # G = self.G
        
        eta1B = np.zeros_like(f)
        eta2B = np.zeros_like(G)
        eta3B = np.zeros_like(W)

        for a in particles:
            for i in holes:
                denom = f[a,a] - f[i,i]
                result = f[a,i] / denom

                eta1B[a,i] += result
                eta1B[i,a] += -result
                
                # if denom < 1:
                #     print('one body {}{},'.format(a, i), denom)

        for a in particles:
            for b in particles:
                for i in holes:
                    for j in holes:
                        denom = (
                            f[a,a] + f[b,b] - f[i,i] - f[j,j] 
                        )
                        result = G[a,b,i,j] / denom
                        
                        eta2B[a,b,i,j] += result
                        eta2B[i,j,a,b] += -result

        for a in particles:
            for b in particles:
                for c in holes:
                    for i in holes:
                        for j in holes:
                            for k in holes:
                                denom = (
                                    f[a,a] + f[b,b] + f[c,c] - f[i,i] - f[j,j] - f[k,k]
                                )
                                result = W[a,b,c,i,j,k] / denom

                                eta3B[a,b,c,i,j,k] += result
                                eta3B[i,j,k,a,b,c] += -result

        return (eta1B, eta2B, eta3B)

class WhiteGeneratorMP(Generator):
    """Calculate White's generator for a normal ordered Hamiltonian.
       This "standard" implemenation uses Moller-Plesset denominators."""


    def __init__(self, h):

        assert isinstance(h, Hamiltonian), "Arg 0 must be Hamiltonian object"
        
        self.f = h.f
        self.G = h.G


        self._holes = h.holes
        self._particles = h.particles
        self._sp_basis = h.sp_basis

        self._eta1B = np.zeros_like(self.f)
        self._eta2B = np.zeros_like(self.G)

    @property
    def eta1B(self):
        """Returns:

        eta1B -- one-body generator"""
        return self._eta1B

    @property
    def eta2B(self):
        """Returns:

        eta2B -- one-body generator"""
        return self._eta2B

    @property
    def f(self):
        """Returns:

        f -- one-body tensor elements (initialized by Hamiltonian object)"""
        return self._f

    @property
    def G(self):
        """Returns:

        f -- two-body tensor elements (initialized by Hamiltonian object)"""
        return self._G

    @f.setter
    def f(self, f):
        """Sets the one-body tensor."""
        self._f = f

    @G.setter
    def G(self, G):
        """Sets the two-body tensor."""
        self._G = G

    def calc_eta(self):

        bas1B = self._sp_basis
        holes = self._holes
        particles = self._particles

        f = self.f
        G = self.G
        
        eta1B = np.zeros_like(f)
        eta2B = np.zeros_like(G)
        
        for a in particles:
            for i in holes:
                denom = f[a,a] - f[i,i]
                result = f[a,i] / denom

                eta1B[a,i] += result
                eta1B[i,a] += -result
                
                # if denom < 1:
                #     print('one body {}{},'.format(a, i), denom)

        for a in particles:
            for b in particles:
                for i in holes:
                    for j in holes:
                        denom = (
                            f[a,a] + f[b,b] - f[i,i] - f[j,j] 
                        )
                        result = G[a,b,i,j] / denom
                        
                        eta2B[a,b,i,j] += result
                        eta2B[i,j,a,b] += -result

                        # if denom < 1:
                        #     print('two body {}{}{}{},'.format(a,b,i,j), denom)
                        #print(eta2B[5,4,3,2])


        return (eta1B, eta2B)


class BrillouinGenerator(Generator):
    """Calculate Brillouin generator for a normal ordered Hamiltonian."""


    def __init__(self, h):

        assert isinstance(h, Hamiltonian), "Arg 0 must be Hamiltonian object"
        
        self.f = h.f
        self.G = h.G

        self._holes = h.holes
        self._particles = h.particles
        self._sp_basis = h.sp_basis

    @property
    def f(self):
        """Returns:

        f -- one-body tensor elements (initialized by Hamiltonian object)"""
        return self._f

    @property
    def G(self):
        """Returns:

        f -- two-body tensor elements (initialized by Hamiltonian object)"""
        return self._G

    @f.setter
    def f(self, f):
        """Sets the one-body tensor."""
        self._f = f

    @G.setter
    def G(self, G):
        """Sets the two-body tensor."""
        self._G = G

    def calc_eta(self):

        bas1B = self._sp_basis
        holes = self._holes
        particles = self._particles

        f = self.f
        G = self.G
        
        eta1B = np.zeros_like(f)
        eta2B = np.zeros_like(G)
        

        for a in particles:
            for i in holes:
                # (1-n_a)n_i - n_a(1-n_i) = n_i - n_a
                eta1B[a, i] +=  f[a,i]
                eta1B[i, a] += -f[a,i]



        for a in particles:
            for b in particles:
                for i in holes:
                    for j in holes:
                        val = G[a,b,i,j]

                        eta2B[a,b,i,j] += val
                        eta2B[i,j,a,b] += -val


        return (eta1B, eta2B)

class BrillouinGeneratorMR(Generator):
    """Calculate multi reference Brillouin generator for a normal ordered Hamiltonian."""


    def __init__(self, h, occ):

        print('Start BrilluoinMR generator')
        assert isinstance(h, Hamiltonian), "Arg 0 must be Hamiltonian object"
        
        self.f = h.f
        self.G = h.G

        self._holes = h.holes
        self._particles = h.particles
        self._sp_basis = h.sp_basis

        self._reference = h.reference

        self._eta1B = np.zeros_like(self.f)
        self._eta2B = np.zeros_like(self.G)
        
        self._occA = occ.occA#.tensor
        self._occB = occ.occB#.tensor
        self._occI = occ.occI#.tensor
        self._occA4 = occ.occA4#.tensor
        self._occB4 = occ.occB4#.tensor
        self._occI4 = occ.occI4#.tensor
        
        rho1b = h._rho1b
        rho2b = h._rho2b
        rho3b = h._rho3b
        self._lambda2b = h._lambda2b
        self._lambda3b = h._lambda3b

        # rho1b = density_1b(len(h.holes), len(h.particles), weights=h._dens_weights)
        # rho2b = density_2b(len(h.holes), len(h.particles), weights=h._dens_weights)
        
        # ti = time.time()
        # rho3b = density_3b(len(holes), len(particles), weights=h._dens_weights)
        # tf = time.time()
        # print('generated 3b density in {: .4f} seconds'.format(tf-ti))

        # lambda2b = np.zeros_like(rho2b)
        # for i in h.sp_basis:
        #     for j in h.sp_basis:
        #         for k in h.sp_basis:
        #             for l in h.sp_basis:
        #                 lambda2b[i,j,k,l] += rho2b[i,j,k,l] - rho1b[i,k]*rho1b[j,l] + rho1b[i,l]*rho1b[j,k]
        
        # lambda3b = np.zeros_like(rho3b)
        # for i in h.sp_basis:
        #     for j in h.sp_basis:
        #         for k in h.sp_basis:
        #             for l in h.sp_basis:
        #                 for m in h.sp_basis:
        #                     for n in h.sp_basis:
        #                         lambda3b[i,j,k,l,m,n] += rho3b[i,j,k,l,m,n] - rho1b[i,l]*lambda2b[j,k,m,n] -\
        #                                                  rho1b[j,m]*lambda2b[i,k,l,n] - rho1b[k,n]*lambda2b[i,j,l,m] +\
        #                                                  rho1b[i,m]*lambda2b[j,k,l,n] + rho1b[i,n]*lambda2b[j,k,m,l] +\
        #                                                  rho1b[j,l]*lambda2b[i,k,m,n] + rho1b[j,n]*lambda2b[i,k,l,m] +\
        #                                                  rho1b[k,l]*lambda2b[i,j,n,m] + rho1b[k,m]*lambda2b[i,j,l,n] -\
        #                                                  rho1b[i,l]*rho1b[j,m]*rho1b[k,n] - rho1b[i,m]*rho1b[j,n]*rho1b[k,l] -\
        #                                                  rho1b[i,n]*rho1b[j,l]*rho1b[k,m] + rho1b[i,l]*rho1b[j,n]*rho1b[k,m] +\
        #                                                  rho1b[j,m]*rho1b[i,n]*rho1b[k,l] + rho1b[i,m]*rho1b[j,l]*rho1b[k,n]

        # self._eta1B = np.zeros_like(self.f)
        # self._eta2B = np.zeros_like(self.G)

    @property
    def eta1B(self):
        """Returns:

        eta1B -- one-body generator"""
        return self._eta1B

    @property
    def eta2B(self):
        """Returns:

        eta2B -- one-body generator"""
        return self._eta2B


    @property
    def f(self):
        """Returns:

        f -- one-body tensor elements (initialized by Hamiltonian object)"""
        return self._f

    @property
    def G(self):
        """Returns:

        f -- two-body tensor elements (initialized by Hamiltonian object)"""
        return self._G

    @f.setter
    def f(self, f):
        """Sets the one-body tensor."""
        self._f = f

    @G.setter
    def G(self, G):
        """Sets the two-body tensor."""
        self._G = G

    @staticmethod
    @jit(nopython=True)
    def _wrapper_calc_eta(numsp, f, Gamma, lambdaG, Glambda, lambda2b, lambda3b, occA, occB, occI):
        
        # 1b generator
        eta1b = np.zeros_like(f)
        for i in range(numsp):
            for j in range(numsp):
        
                Gammalambda = np.zeros_like(f)
                for a in range(numsp):
                    for b in range(numsp):
                        for c in range(numsp):
                            Gammalambda[i,j] += Gamma[j,a,b,c]*lambda2b[i,a,b,c] - Gamma[a,b,i,c]*lambda2b[a,b,j,c]
                
                eta1b[i,j] += occA[j,i]*f[j,i] - 0.5*Gammalambda[i,j]
        
        # 2b generator
        eta2b = np.zeros_like(Gamma)
        for i in range(numsp):
            for j in range(numsp):
                for k in range(numsp):
                    for l in range(numsp):
                        
                        sum_2b_1 = np.zeros_like(Gamma)
                        sum_2b_2 = np.zeros_like(Gamma)
                        sum_2b_3 = np.zeros_like(Gamma)
                        for a in range(numsp):
                            sum_2b_1[i,j,k,l] += f[a,i]*lambda2b[a,j,k,l] - f[a,j]*lambda2b[a,i,k,l] - f[k,a]*lambda2b[i,j,a,l] + f[l,a]*lambda2b[i,j,a,k]

                        for a in range(numsp):
                            for c in range(numsp):
                                sum_2b_2[i,j,k,l] += occA[j,k]*Gamma[a,k,c,j]*lambda2b[a,i,c,l] - \
                                                     occA[i,k]*Gamma[a,k,c,i]*lambda2b[a,j,c,l] - \
                                                     occA[j,l]*Gamma[a,l,c,j]*lambda2b[a,i,c,k] + \
                                                     occA[i,l]*Gamma[a,l,c,i]*lambda2b[a,j,c,k]
                        for a in range(numsp):
                            for b in range(numsp):
                                for c in range(numsp):
                                    sum_2b_3[i,j,k,l] += Gamma[k,a,b,c]*lambda3b[a,i,j,b,c,l] - Gamma[l,a,b,c]*lambda3b[a,i,j,b,c,k] -\
                                                         Gamma[a,b,i,c]*lambda3b[a,b,j,c,k,l] + Gamma[a,b,j,c]*lambda3b[a,b,i,c,k,l]

                        result = Gamma[k,l,i,j]*occI[i,j,k,l] \
                                 + sum_2b_1[i,j,k,l] \
                                 + 0.5*(lambdaG[k,l,i,j]*occB[i,j] - Glambda[k,l,i,j]*occB[k,l]) \
                                 + sum_2b_2[i,j,k,l] \
                                 + 0.5*sum_2b_3[i,j,k,l]

                        eta2b[i,j,k,l] += result
                        eta2b[k,l,i,j] += -result

        return (eta1b, eta2b)

    def calc_eta(self):

        bas1B = self._sp_basis
        holes = self._holes
        particles = self._particles
        
        f = self.f
        Gamma = self.G

        lambda2b = (self._lambda2b)
        lambda3b = (self._lambda3b)

        occA = self._occA
        occB = self._occB
        occI = self._occI4

        numStates = len(bas1B)
        dimMat = (numStates**2, numStates**2)
        dimTens = (numStates, numStates, numStates, numStates)
        lambda2b_rs = np.reshape(lambda2b, dimMat)
        G_rs = np.reshape(Gamma, dimMat)
        lambdaG = np.matmul(lambda2b_rs, G_rs)
        Glambda = np.matmul(G_rs, lambda2b_rs)
        lambdaG = np.reshape(lambdaG, dimTens)
        Glambda = np.reshape(Glambda, dimTens)

        #print(type(lambdaG), type(Glambda), type(occB), type(occI))

        eta1B, eta2B = self._wrapper_calc_eta(len(bas1B), f, Gamma, lambdaG, Glambda, lambda2b, lambda3b, occA, occB, occI)

        # update the class members
        self._eta1B = eta1B
        self._eta2B = eta2B

        return (eta1B, eta2B)

    # def calc_eta(self):

    #     bas1B = self._sp_basis
    #     holes = self._holes
    #     particles = self._particles
        
    #     f = self.f
    #     G = self.G

    #     lambda2b = self._lambda2b
    #     lambda3b = (self._lambda3b)

    #     occA = self._occA
    #     occA4 = self._occA4
    #     occB4 = self._occB4
    #     occI4 = self._occI4
        
    #     # 1B generator
    #     sum1_1b = np.multiply(np.transpose(occA), np.transpose(f))
    #     #sum1_1b = np.transpose(sum1_1b)
        
    #     sum2_1b_1 = tn.ncon([G, lambda2b], [(-2,1,2,3),(-1,1,2,3)])
    #     sum2_1b_2 = tn.ncon([G, lambda2b], [(1,2,-1,3),(1,2,-2,3)])        
    #     sum2_1b = sum2_1b_1 - sum2_1b_2
        
    #     #sum2_1b = 0

    #     eta1B = sum1_1b - 0.5*sum2_1b
    #     eta1B = -1*eta1B.conj().T

    #     #print(np.linalg.norm(sum1_1b + sum1_1b.conj().T), np.linalg.norm(sum2_1b + sum2_1b.conj().T), np.linalg.norm(eta1B + eta1B.conj().T))

    #     #eta1B = np.zeros_like(eta1B)
    #     #print(np.linalg.norm(G), np.linalg.norm(lambda2b))
        
    #     # 2B generator
    #     sum1_2b = np.multiply(G,np.transpose(occI4,[2,3,0,1]))
    #     sum1_2b = np.transpose(sum1_2b, [2,3,0,1])
        
    #     sum2_2b_1 = tn.ncon([f, lambda2b], [(1,-1),(1,-2,-3,-4)])
    #     sum2_2b_2 = tn.ncon([f, lambda2b], [(-3,1),(-1,-2,1,-4)])
    #     sum2_2b_3 = tn.ncon([f, lambda2b], [(1,-2),(1,-1,-3,-4)])
    #     sum2_2b_4 = tn.ncon([f, lambda2b], [(-4,1),(-1,-2,1,-3)])
    #     sum2_2b = sum2_2b_1 - sum2_2b_3 - sum2_2b_2 + sum2_2b_4
    #     # sum2_2b_3 = sum2_2b_1 - np.transpose(sum2_2b_1,[1,0,2,3])
    #     # sum2_2b_4 = sum2_2b_2 - np.transpose(sum2_2b_2,[0,1,3,2])
    #     # sum2_2b = sum2_2b_3 - sum2_2b_4

    #     numStates = len(bas1B)
    #     dimMat = (numStates**2, numStates**2)
    #     dimTens = (numStates, numStates, numStates, numStates)
    #     lambda2b_rs = np.reshape(lambda2b, dimMat)
    #     G_rs = np.reshape(G, dimMat)
    #     lambdaG = np.matmul(lambda2b_rs, G_rs)
    #     Glambda = np.matmul(G_rs, lambda2b_rs)
    #     lambdaG = np.reshape(lambdaG, dimTens)
    #     Glambda = np.reshape(Glambda, dimTens)

    #     sum3_2b_1 = np.multiply(np.transpose(lambdaG,[2,3,0,1]), np.transpose(occB4, [2,3,0,1]))
    #     sum3_2b_2 = np.multiply(np.transpose(Glambda,[2,3,0,1]), occB4)
    #     sum3_2b = sum3_2b_1 - sum3_2b_2

    #     occA_transpose = np.transpose(occA4, [3,1,2,0])
    #     sum4_2b_1 = np.multiply(occA_transpose, G)
    #     sum4_2b_2 = tn.ncon([sum4_2b_1, lambda2b], [(1,-3,2,-2), (1,-1,2,-4)])
    #     sum4_2b_3 = tn.ncon([sum4_2b_1, lambda2b], [(1,-3,2,-1), (1,-2,2,-4)])
    #     sum4_2b_4 = tn.ncon([sum4_2b_1, lambda2b], [(1,-4,2,-2), (1,-1,2,-3)])
    #     sum4_2b_5 = tn.ncon([sum4_2b_1, lambda2b], [(1,-4,2,-1), (1,-2,2,-3)])
    #     sum4_2b = sum4_2b_2 - sum4_2b_3 - sum4_2b_4 + sum4_2b_5
    #     # sum4_2b = sum4_2b_2 - np.transpose(sum4_2b_2,[1,0,2,3]) - \
    #     #           - np.transpose(sum4_2b_2,[0,1,3,2]) + np.transpose(sum4_2b_2,[1,0,3,2])

    #     sum5_2b_1 = tn.ncon([G, lambda3b], [(-3,1,2,3),(1,-1,-2,2,3,-4)])
    #     sum5_2b_2 = tn.ncon([G, lambda3b], [(1,2,-1,3),(1,2,-2,3,-3,-4)])
    #     sum5_2b_3 = tn.ncon([G, lambda3b], [(-4,1,2,3),(1,-1,-2,2,3,-3)])
    #     sum5_2b_4 = tn.ncon([G, lambda3b], [(1,2,-2,3),(1,2,-1,3,-3,-4)])
    #     # sum5_2b_3 = np.transpose(sum5_2b_1, [0,1,3,2])
    #     # sum5_2b_4 = np.transpose(sum5_2b_2, [1,0,2,3])

    #     sum5_2b = sum5_2b_1 - sum5_2b_3 - sum5_2b_2 + sum5_2b_4

    #     #sum4_2b = 0
        
    #     eta2B = 0.25*(sum1_2b + sum2_2b + 0.5*sum3_2b + sum4_2b + 0.5*sum5_2b)
        
    #     eta2b_rs = np.reshape(eta2B, (len(bas1B)**2, len(bas1B)**2))
    #     sum4_2b_rs = np.reshape(sum4_2b, (len(bas1B)**2, len(bas1B)**2))

    #     eta2b_rs = -1*eta2b_rs.conj().T
    #     eta2B = np.reshape(eta2b_rs, (len(bas1B),len(bas1B),len(bas1B),len(bas1B)))

    #     #eta2b_rs = np.reshape(eta2B, (len(bas1B)**2, len(bas1B)**2))

    #     #print(np.linalg.norm(eta2b_rs + eta2b_rs.conj().T), np.linalg.norm(G_rs - G_rs.conj().T), np.linalg.norm(sum1_2b_rs - sum1_2b_rs.conj().T))
    #     #print(np.linalg.norm(sum4_2b_rs + sum4_2b_rs.conj().T));
                              

    #     #eta2B = np.zeros_like(self._eta2B)
    #     #norm = np.linalg.norm
        
    #     # update the class members
    #     self._eta1B = eta1B
    #     self._eta2B = eta2B


    #     return (eta1B, eta2B)


class ImTimeGenerator(Generator):
    """Calculate Imaginary time generator for a normal ordered Hamiltonian."""


    def __init__(self, h):

        assert isinstance(h, Hamiltonian), "Arg 0 must be Hamiltonian object"
        
        self.f = h.f
        self.G = h.G

        self._holes = h.holes
        self._particles = h.particles
        self._sp_basis = h.sp_basis
        self._eta1B = np.zeros_like(self.f)
        self._eta2B = np.zeros_like(self.G)

    @property
    def eta1B(self):
        """Returns:

        eta1B -- one-body generator"""
        return self._eta1B

    @property
    def eta2B(self):
        """Returns:

        eta2B -- one-body generator"""
        return self._eta2B

    @property
    def f(self):
        """Returns:

        f -- one-body tensor elements (initialized by Hamiltonian object)"""
        return self._f

    @property
    def G(self):
        """Returns:

        f -- two-body tensor elements (initialized by Hamiltonian object)"""
        return self._G

    @f.setter
    def f(self, f):
        """Sets the one-body tensor."""
        self._f = f

    @G.setter
    def G(self, G):
        """Sets the two-body tensor."""
        self._G = G

    def calc_eta(self):

        bas1B = self._sp_basis
        holes = self._holes
        particles = self._particles

        f = self.f
        G = self.G
        
        eta1B = np.zeros_like(f)
        eta2B = np.zeros_like(G)
        

        for a in particles:
            for i in holes:
                dE = f[a,a] - f[i,i] + G[a,i,a,i]
                val = np.sign(dE)*f[a,i]
                eta1B[a, i] =  val
                eta1B[i, a] = -val 


        for a in particles:
            for b in particles:
                for i in holes:
                    for j in holes:
                        dE = ( 
                            f[a,a] + f[b,b] - f[i,i] - f[j,j]  
                            + G[a,b,a,b] 
                            + G[i,j,i,j]
                            - G[a,i,a,i] 
                            - G[a,j,a,j] 
                            - G[b,i,b,i] 
                            - G[b,j,b,j] 
                        )

                        val = np.sign(dE)*G[a,b,i,j]

                        eta2B[a,b,i,j] = val
                        eta2B[i,j,a,b] = -val


        return (eta1B, eta2B)

