import tensorflow as tf
# tf.enable_v2_behavior()
from tensornetwork import *
import numpy as np
from oop_imsrg.hamiltonian import *
from oop_imsrg.occupation_tensors import *

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

        assert isinstance(h, Hamiltonian), "Arg 0 must be Hamiltonian object"
        assert isinstance(occ_t, OccupationTensors), "Arg 1 must be OccupationTensors object"

        self.f = h.f
        self.G = h.G


        self._holes = h.holes
        self._particles = h.particles

        self._occA = occ_t.occA
        self._occB = occ_t.occB
        self._occC = occ_t.occC
        self._occD = occ_t.occD

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

        # - Decouple off-diagonal 1B and 2B pieces
        fod = np.zeros(f.shape, dtype=np.float32)
        fod[np.ix_(particles, holes)] += f[np.ix_(particles, holes)]
        fod[np.ix_(holes, particles)] += f[np.ix_(holes, particles)]
        fd = f - fod

        God = np.zeros(G.shape, dtype=np.float32)
        God[np.ix_(particles, particles, holes, holes)] += G[np.ix_(particles, particles, holes, holes)]
        God[np.ix_(holes, holes, particles, particles)] += G[np.ix_(holes, holes, particles, particles)]
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
        fd = partition[0]
        fod = partition[1]
        Gd = partition[2]
        God = partition[3]

        holes = self._holes
        particles = self._particles

        occA = self._occA
        occB = self._occB
        occC = self._occC


        # - Calculate 1B generator
        # first term
        sum1_1b_1 = ncon([fd, fod], [(-1, 0), (0, -2)]).numpy()
        sum1_1b_2 = np.transpose(sum1_1b_1)
        sum1_1b = sum1_1b_1 - sum1_1b_2

        # second term
        sum2_1b_1 = ncon([fd, occA, God], [(2,3), (2,3,0,1), (1,-1,0,-2)]).numpy()
        sum2_1b_2 = ncon([fod, occA, Gd], [(2,3), (2,3,0,1), (1,-1,0,-2)]).numpy()
        sum2_1b = sum2_1b_1 - sum2_1b_2
        # sum2_1b_1 = ncon([fd, God], [(0, 1), (1, -1, 0, -2)]).numpy()
        # sum2_1b_2 = ncon([fod, Gd], [(0, 1), (1, -1, 0, -2)]).numpy()
        # sum2_1b_3 = sum2_1b_1 - sum2_1b_2
        # sum2_1b = ncon([occA, sum2_1b_3],[(-1, -2, 0, 1), (0,1)]).numpy()

        # third term
        sum3_1b_1 = ncon([Gd, occC, God], [(5,-1,3,4), (3,4,5,0,1,2), (0,1,2,-2)]).numpy()
        sum3_1b = sum3_1b_1 - np.transpose(sum3_1b_1)
        # sum3_1b_1 = ncon([occC, God], [(-1, -2, -3, 0, 1, 2), (0, 1, 2, -4)]).numpy()
        # sum3_1b_2 = ncon([Gd, sum3_1b_1], [(2, -1, 0, 1), (0, 1, 2, -2)]).numpy()
        # sum3_1b_3 = np.transpose(sum3_1b_2)
        # sum3_1b = sum3_1b_2 - sum3_1b_3

        eta1B = sum1_1b + sum2_1b + 0.5*sum3_1b

        # - Calculate 2B generator
        # first term (P_ij piece)
        sum1_2b_1 = ncon([fd, God], [(-1, 0), (0, -2, -3, -4)]).numpy()
        sum1_2b_2 = ncon([fod, Gd], [(-1, 0), (0, -2, -3, -4)]).numpy()
        sum1_2b_3 = sum1_2b_1 - sum1_2b_2
        sum1_2b_4 = np.transpose(sum1_2b_3, [1, 0, 2, 3])
        sum1_2b_5 = sum1_2b_3 - sum1_2b_4

        # first term (P_kl piece)
        sum1_2b_6 = ncon([fd, God], [(0, -3), (-1, -2, 0, -4)]).numpy()
        sum1_2b_7 = ncon([fod, Gd], [(0, -3), (-1, -2, 0, -4)]).numpy()
        sum1_2b_8 = sum1_2b_6 - sum1_2b_7
        sum1_2b_9 = np.transpose(sum1_2b_8, [0, 1, 3, 2])
        sum1_2b_10 = sum1_2b_8 - sum1_2b_9

        sum1_2b = sum1_2b_5 - sum1_2b_10

        # second term
        sum2_2b_1 = ncon([Gd, occB, God], [(-1,-2,2,3), (2,3,0,1), (0,1,-3,-4)]).numpy()
        sum2_2b_2 = ncon([God, occB, Gd], [(-1,-2,2,3), (2,3,0,1), (0,1,-3,-4)]).numpy()
        sum2_2b = sum2_2b_1 - sum2_2b_2
        # sum2_2b_1 = ncon([occB, God], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
        # sum2_2b_2 = ncon([occB,  Gd], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
        # sum2_2b_3 = ncon([Gd,  sum2_2b_1], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
        # sum2_2b_4 = ncon([God, sum2_2b_2], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
        # sum2_2b = sum2_2b_3 - sum2_2b_4

        # third term
        sum3_2b_1 = ncon([Gd, occA, God], [(2,-1,3,-3), (2,3,0,1), (1,-2,0,-4)]).numpy()
        sum3_2b_2 = sum3_2b_1 - np.transpose(sum3_2b_1, [0,1,3,2])
        sum3_2b = sum3_2b_2 - np.transpose(sum3_2b_2, [1,0,2,3])
        # sum3_2b_1 = ncon([Gd, God], [(0, -1, 1, -3), (1, -2, 0, -4)]).numpy()
        # sum3_2b_2 = np.transpose(sum3_2b_1, [1, 0, 2, 3])
        # sum3_2b_3 = np.transpose(sum3_2b_1, [0, 1, 3, 2])
        # sum3_2b_4 = np.transpose(sum3_2b_1, [1, 0, 3, 2])
        # sum3_2b_5 = sum3_2b_1 - sum3_2b_2 - sum3_2b_3 + sum3_2b_4
        # sum3_2b = ncon([occA, sum3_2b_5], [(0, 1, -1, -2), (0, 1, -3, -4)]).numpy()

        eta2B = sum1_2b + 0.5*sum2_2b + sum3_2b

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
        self._occA2 = occ_t.occA2
        self._occB = occ_t.occB
        self._occC = occ_t.occC
        self._occD = occ_t.occD
        self._occE = occ_t.occE
        self._occF = occ_t.occF
        self._occG = occ_t.occG
        self._occH = occ_t.occH
        self._occI = occ_t.occI
        self._occJ = occ_t.occJ

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

        occA = self._occA
        occA2 = self._occA2
        occB = self._occB
        occC = self._occC
        occD = self._occD
        occF = self._occF
        occG = self._occG
        occH = self._occH
        occI = self._occI
        occJ = self._occJ


        # Calculate 1B generator
        # fourth term
        sum4_1b_1 = np.matmul(np.transpose(occD,[2,3,0,1]), God)
        sum4_1b_2 = np.matmul(np.transpose(occD,[2,3,0,1]), Gd)
        sum4_1b_3 = ncon([Wd,  sum4_1b_1], [(0,1,-1,2,3,-2),(2,3,0,1)]).numpy()
        sum4_1b_4 = ncon([Wod, sum4_1b_2], [(0,1,-1,2,3,-2),(2,3,0,1)]).numpy()
        sum4_1b = sum4_1b_3 - sum4_1b_4

        # fifth term
        sum5_1b_1 = ncon([occF, Wd.astype(np.float32)],
                         [(-1,-3,-4,-5,-6,0,1,2,3,4), (0,1,-2,2,3,4)]).numpy()
        sum5_1b_2 = ncon([occF, Wod.astype(np.float32)],
                         [(-1,-3,-4,-5,-6,0,1,2,3,4), (0,1,-2,2,3,4)]).numpy()
        sum5_1b_3 = ncon([sum5_1b_1, Wod.astype(np.float32)],
                         [(0,1,-1,2,3,4), (2,3,4,0,1,-2)]).numpy()
        sum5_1b_4 = ncon([sum5_1b_2, Wd.astype(np.float32)],
                         [(0,1,-1,2,3,4), (2,3,4,0,1,-2)]).numpy()
        sum5_1b = sum5_1b_3 - sum5_1b_4

        eta1B += 0.25*sum4_1b + (1/12)*sum5_1b

        # Calculate 2B generator
        # fourth term
        sum4_2b_1 = np.matmul(-1.0*np.transpose(occA2), fod)
        sum4_2b_2 = np.matmul(-1.0*np.transpose(occA2),  fd)
        sum4_2b_3 = ncon([Wd,  sum4_2b_1], [(0,-1,-2,1,-3,-4), (1,0)]).numpy()
        sum4_2b_4 = ncon([Wod, sum4_2b_2], [(0,-1,-2,1,-3,-4), (1,0)]).numpy()
        sum4_2b = sum4_2b_3 - sum4_2b_4

        #fifth term
        sum5_2b_1 = ncon([occG, God], [(-1,-2,-4,0,1,2), (1,2,0,-3)]).numpy()
        sum5_2b_2 = ncon([occG,  Gd], [(-1,-2,-4,0,1,2), (1,2,0,-3)]).numpy()
        sum5_2b_3 = ncon([Wd,  sum5_2b_1], [(0,-1,-2,1,2,-4), (1,2,0,-3)]).numpy()
        sum5_2b_4 = ncon([Wod, sum5_2b_2], [(0,-1,-2,1,2,-4), (1,2,0,-3)]).numpy()
        sum5_2b_5 = sum5_2b_3 - sum5_2b_4
        sum5_2b = sum5_2b_5 - np.transpose(sum5_2b_5, [3,2,0,1]) - \
                    np.transpose(sum5_2b_5, [0,1,3,2]) + \
                    np.transpose(sum5_2b_5, [2,3,0,1])

        #sixth term
        sum6_2b_1 = ncon([occH, Wod], [(-1,-2,-3,-4,0,1,2,3),(1,2,3,0,-5,-6)]).numpy()
        sum6_2b_2 = ncon([occH, Wod], [(-3,-4,-5,-6,0,1,2,3),(0,-1,-2,1,2,3)]).numpy()
        sum6_2b_3 = ncon([Wd, sum6_2b_1], [(0,-1,-2,1,2,3), (1,2,3,0,-3,-4)]).numpy()
        sum6_2b_4 = ncon([Wd, sum6_2b_2], [(1,2,3,0,-3,-4), (0,-1,-2,1,2,3)]).numpy()
        sum6_2b = sum6_2b_3 - sum6_2b_4

        #seventh term
        sum7_2b_1 = ncon([occI, Wod], [(-1,-2,-3,-4,0,1,2,3), (2,3,-5,0,1,-6)]).numpy()
        sum7_2b_2 = ncon([Wd, sum7_2b_1], [(0,1,-1,2,3,-4),(2,3,-2,0,1,-3)]).numpy()
        sum7_2b = sum7_2b_2 - np.transpose(sum7_2b_2,[1,0,2,3]) - \
                              np.transpose(sum7_2b_2,[0,1,3,2]) + \
                              np.transpose(sum7_2b_2,[1,0,3,2])

        eta2B += sum4_2b + 0.5*sum5_2b + (1/6)*sum6_2b + 0.25*sum7_2b

        # Calculate 3B generator
        #first, second, and third terms (one index contraction)

        #terms with P(i/jk) -- line 1 and 2
        sum1_3b_1 = ncon([fd, Wod], [(-1,0), (0,-2,-3,-4,-5,-6)]).numpy()
        sum1_3b_2 = ncon([fod, Wd], [(-1,0), (0,-2,-3,-4,-5,-6)]).numpy()
        sum1_3b_3 = sum1_3b_1 - sum1_3b_2
        sum1_3b_4 = sum1_3b_3 - np.transpose(sum1_3b_3, [1,0,2,3,4,5]) - \
                                np.transpose(sum1_3b_3, [2,1,0,3,4,5])

        #terms with P(l/mn) -- line 1 and 2
        sum1_3b_5 = ncon([fd, Wod], [(0,-4), (-1,-2,-3,0,-5,-6)]).numpy()
        sum1_3b_6 = ncon([fod, Wd], [(0,-4), (-1,-2,-3,0,-5,-6)]).numpy()
        sum1_3b_7 = sum1_3b_6 - sum1_3b_5
        sum1_3b_8 = sum1_3b_7 - np.transpose(sum1_3b_7, [0,1,2,4,3,5]) - \
                                np.transpose(sum1_3b_7, [0,1,2,5,4,3])

        #terms with P(ij/k)P(l/mn) -- line 3
        sum1_3b_9  = ncon([Gd, God], [(-1,-2,-4,0),(0,-3,-5,-6)]).numpy()
        sum1_3b_10 = ncon([God, Gd], [(-1,-2,-4,0),(0,-3,-5,-6)]).numpy()
        sum1_3b_11 = sum1_3b_9 - sum1_3b_10
        sum1_3b_12 = sum1_3b_11 - np.transpose(sum1_3b_11, [0,1,2,4,3,5]) - \
                                  np.transpose(sum1_3b_11, [0,1,2,5,4,3])
        sum1_3b_13 = sum1_3b_12 - np.transpose(sum1_3b_12, [2,1,0,3,4,5]) - \
                                  np.transpose(sum1_3b_12, [0,2,1,3,4,5])

        sum1_3b = sum1_3b_4 + sum1_3b_8 + sum1_3b_13

        #fourth term
        sum4_3b_1 = ncon([Gd, occB, Wod], [(-1,-2,2,3),(2,3,0,1),(0,1,-3,-4,-5,-6)]).numpy()
        sum4_3b_2 = ncon([God, occB, Wd], [(-1,-2,2,3),(2,3,0,1),(0,1,-3,-4,-5,-6)]).numpy()
        sum4_3b_3 = sum4_3b_1 - sum4_3b_2
        sum4_3b = sum4_3b_3 - np.transpose(sum4_3b_3, [1,0,2,3,4,5]) - \
                              np.transpose(sum4_3b_3, [2,1,0,3,4,5])

        #fifth term
        sum5_3b_1 = ncon([Gd, occB, Wod], [(2,3,-4,-5),(2,3,0,1),(-1,-2,-3,0,1,-6)]).numpy()
        sum5_3b_2 = ncon([God, occB, Wd], [(2,3,-4,-5),(2,3,0,1),(-1,-2,-3,0,1,-6)]).numpy()
        sum5_3b_3 = sum5_3b_1 - sum5_3b_2
        sum5_3b = sum5_3b_3 - np.transpose(sum5_3b_3, [0,1,2,5,4,3]) - \
                              np.transpose(sum5_3b_3, [0,1,2,3,5,4])

        #sixth term
        sum6_3b_1 = ncon([Gd, occA, Wod], [(3,-1,2,-4),(2,3,0,1),(0,-2,-3,1,-5,-6)]).numpy()
        sum6_3b_2 = ncon([God, occA, Wd], [(3,-1,2,-4),(2,3,0,1),(0,-2,-3,1,-5,-6)]).numpy()
        sum6_3b_3 = sum6_3b_1 - sum6_3b_2
        sum6_3b_4 = sum6_3b_3 - np.transpose(sum6_3b_3, [0,1,2,4,3,5]) - \
                                np.transpose(sum6_3b_3, [0,1,2,5,4,3])
        sum6_3b = sum6_3b_4 - np.transpose(sum6_3b_4, [1,0,2,3,4,5]) - \
                              np.transpose(sum6_3b_4, [2,1,0,3,4,5])

        #seventh term
        sum7_3b_1 = ncon([Wd, occJ, Wod], [(-1,-2,-3,3,4,5), (3,4,5,0,1,2), (0,1,2,-4,-5,-6)]).numpy()
        sum7_3b_2 = ncon([Wod, occJ, Wd], [(-1,-2,-3,3,4,5), (3,4,5,0,1,2), (0,1,2,-4,-5,-6)]).numpy()
        sum7_3b = sum7_3b_1 - sum7_3b_2

        #eighth term
        sum8_3b_1 = ncon([Wd, occC, Wod], [(3,4,-3,5,-5,-6), (3,4,5,0,1,2), (2,-1,-2,0,1,-4)]).numpy()
        sum8_3b_2 = ncon([Wd, occC, Wod], [(5,-2,-3,3,4,-6), (3,4,5,0,1,2), (-1,0,1,-4,-5,2)]).numpy()
        sum8_3b_3 = sum8_3b_1 - sum8_3b_2
        sum8_3b_4 = sum8_3b_3 - np.transpose(sum8_3b_3, [0,1,2,4,3,5]) - \
                                np.transpose(sum8_3b_3, [0,1,2,5,4,3])
        sum8_3b = sum8_3b_4 - np.transpose(sum8_3b_4, [2,1,0,3,4,5]) - \
                              np.transpose(sum8_3b_4, [0,2,1,3,4,5])

        eta3B = sum1_3b + 0.5*sum4_3b + (-0.5)*sum5_3b + (-1.0)*sum6_3b + (1/6)*sum7_3b + 0.5*sum8_3b

        return (eta1B, eta2B, eta3B)
