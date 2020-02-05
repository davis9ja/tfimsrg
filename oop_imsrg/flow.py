#import tensorflow as tf
# tf.enable_v2_behavior()
#import tensornetwork as tn
#import numpy as np
from oop_imsrg.hamiltonian import *
from oop_imsrg.occupation_tensors import *
from oop_imsrg.generator import *
#tn.set_default_backend("tensorflow") 

class Flow(object):
    """Parent class for organization purposes. Ideally, all Flow
    classes should inherit from this class. In this way, AssertionErrors
    can be handled in a general way."""

    def flow():
        print("Function that iterates flow equation once")

class Flow_IMSRG2(Flow):
    """Calculates the flow equations for the IMSRG(2)."""

    def __init__(self, h, occ_t):
        """Class constructor. Instantiates Flow_IMSRG2 object.

        Arguments:

        h -- Hamiltonian object
        occ_t -- OccupationTensors object"""

        assert isinstance(h, Hamiltonian), "Arg 0 must be PairingHamiltonian object"
        assert isinstance(occ_t, OccupationTensors), "Arg 1 must be OccupationTensors object"

        # self.f = h.f
        # self.G = h.G

        self._holes = h.holes
        self._particles = h.particles

        self._occA = occ_t.occA
        self._occB = occ_t.occB
        self._occC = occ_t.occC
        self._occD = occ_t.occD

    # @property
    # def f(self):
    #     return self._f

    # @property
    # def G(self):
    #     return self._G

    # @f.setter
    # def f(self, f):
    #     self._f = f

    # @G.setter
    # def G(self, G):
    #     self._G = G

    def flow(self, gen):
        """Iterates the IMSRG2 flow equations once.

        Arugments:

        gen -- Generator object; generator produces the flow

        Returns:

        (dE, -- zero-body tensor
         df, -- one-body tensor
         dG) -- two-body tensor"""

        assert isinstance(gen, Generator), "Arg 0 must be WegnerGenerator object"

        # f = self.f
        # G = self.G
        f = gen.f.astype(np.float32)
        G = gen.G.astype(np.float32)

        partition = gen.calc_eta()
        eta1B = partition[0]#.astype(np.float32)
        eta2B = partition[1]#.astype(np.float32)

        # occA = occ_t.occA
        # occB = occ_t.occB
        # occC = occ_t.occC
        # occD = occ_t.occD

        occA = self._occA
        occB = self._occB
        occC = self._occC
        occD = self._occD

        # - Calculate dE/ds
        # first term
        sum1_0b = tn.ncon([eta1B, occA, f], [(3,4), (3,4,1,2), (2,1)])#.numpy()
        # sum1_0b_1 = tn.ncon([occA, eta1B], [(0, 1, -1, -2), (0, 1)])#.numpy()
        # sum1_0b = tn.ncon([sum1_0b_1, f], [(0, 1), (1, 0)])#.numpy()

        # second term
        #print(occD.shape)
        #sum2_0b = tn.ncon([eta2B, occD, G], [(5,6,7,8), (5,6,7,8,1,2,3,4), (3,4,1,2)])#.numpy()
        sum2_0b_1 = np.multiply(eta2B, occD.tensor)
        #sum2_0b_1 = tn.ncon([eta2B, occD],[(-1,-2,1,2),(-3,-4,1,2)])
        sum2_0b = tn.ncon([sum2_0b_1, G], [(1, 2, 3, 4), (3, 4, 1, 2)])#.numpy()

        dE = sum1_0b + 0.5*sum2_0b


        # - Calculate df/ds
        # first term
        sum1_1b_1 = tn.ncon([eta1B, f], [(-1, 1), (1, -2)])#.numpy()
        sum1_1b_2 = np.transpose(sum1_1b_1)
        sum1_1b = sum1_1b_1 + sum1_1b_2

        # second term (might need to fix)
        sum2_1b_1 = tn.ncon([eta1B, occA, G], [(3,4), (3,4,1,2), (2,-1,1,-2)])#.numpy()
        sum2_1b_2 = tn.ncon([f, occA, eta2B], [(3,4), (3,4,1,2), (2,-1,1,-2)])#.numpy()
        sum2_1b = sum2_1b_1 - sum2_1b_2
        # sum2_1b_1 = tn.ncon([eta1B, G], [(0, 1), (1, -1, 0, -2)])#.numpy()
        # sum2_1b_2 = tn.ncon([f, eta2B], [(0, 1), (1, -1, 0, -2)])#.numpy()
        # sum2_1b_3 = sum2_1b_1 - sum2_1b_2
        # sum2_1b = tn.ncon([occA, sum2_1b_3],[(-1, -2, 0, 1), (0,1)])#.numpy()

        # third term
        sum3_1b_1 = tn.ncon([eta2B, occC, G], [(6,-1,4,5), (4,5,6,1,2,3), (1,2,3,-2)])#.numpy()
        sum3_1b = sum3_1b_1 + np.transpose(sum3_1b_1)
        # sum3_1b_1 = tn.ncon([occC, G], [(-1, -2, -3, 0, 1, 2), (0, 1, 2, -4)])#.numpy()
        # sum3_1b_2 = tn.ncon([eta2B, sum3_1b_1], [(2, -1, 0, 1), (0, 1, 2, -2)])#.numpy()
        # sum3_1b_3 = np.transpose(sum3_1b_2)
        # sum3_1b = sum3_1b_2 + sum3_1b_3

        df = sum1_1b + sum2_1b + 0.5*sum3_1b


        # - Calculate dG/ds
        # first term (P_ij piece)
        sum1_2b_1 = tn.ncon([eta1B, G], [(-1, 1), (1, -2, -3, -4)])#.numpy()
        sum1_2b_2 = tn.ncon([f, eta2B], [(-1, 1), (1, -2, -3, -4)])#.numpy()
        sum1_2b_3 = sum1_2b_1 - sum1_2b_2
        sum1_2b_4 = np.transpose(sum1_2b_3, [1, 0, 2, 3])
        sum1_2b_5 = sum1_2b_3 - sum1_2b_4

        # first term (P_kl piece)
        sum1_2b_6 = tn.ncon([eta1B, G], [(1, -3), (-1, -2, 1, -4)])#.numpy()
        sum1_2b_7 = tn.ncon([f, eta2B], [(1, -3), (-1, -2, 1, -4)])#.numpy()
        sum1_2b_8 = sum1_2b_6 - sum1_2b_7
        sum1_2b_9 = np.transpose(sum1_2b_8, [0, 1, 3, 2])
        sum1_2b_10 = sum1_2b_8 - sum1_2b_9

        sum1_2b = sum1_2b_5 - sum1_2b_10

        # second term
        # node1 = tn.Node(eta2B)
        # node2 = tn.Node(occB[:,:,None])
        # node2cp = tn.Node(occB[None,:,:])
        # node3 = tn.Node(G)
        # node1[2] ^ node2[0]
        # node1[3] ^ node2[1]
        # node2[2] ^ node2cp[0]
        # node2cp[1] ^ node3[0]
        # node2cp[2] ^ node3[1]
        # result = node1 @ node2 @ node2cp @ node3
        
        sum2_2b_1 = tn.ncon([eta2B, occB, G], [(-1,-2,3,4), (3,4,1,2), (1,2,-3,-4)])#.numpy()
        sum2_2b_2 = tn.ncon([G, occB, eta2B], [(-1,-2,3,4), (3,4,1,2), (1,2,-3,-4)])#.numpy()

#        sum2_2b_1 = tn.ncon([eta2B, occB, occB, G], [(-1,-2,3,4), (3,4),(1,2), (1,2,-3,-4)])#.numpy()
#        sum2_2b_2 = tn.ncon([G, occB, occB, eta2B], [(-1,-2,3,4), (3,4),(1,2), (1,2,-3,-4)])#.numpy()

        sum2_2b = sum2_2b_1 - sum2_2b_2
        # sum2_2b_1 = tn.ncon([occB,     G], [(-1, -2, 0, 1), (0, 1, -3, -4)])#.numpy()
        # sum2_2b_2 = tn.ncon([occB, eta2B], [(-1, -2, 0, 1), (0, 1, -3, -4)])#.numpy()
        # sum2_2b_3 = tn.ncon([eta2B,  sum2_2b_1], [(-1, -2, 0, 1), (0, 1, -3, -4)])#.numpy()
        # sum2_2b_4 = tn.ncon([G,      sum2_2b_2], [(-1, -2, 0, 1), (0, 1, -3, -4)])#.numpy()
        # sum2_2b = sum2_2b_3 - sum2_2b_4

        # third term
        sum3_2b_1 = tn.ncon([eta2B, occA, G], [(3,-1,4,-3), (3,4,1,2), (2,-2,1,-4)])#.numpy()
        sum3_2b_2 = sum3_2b_1 - np.transpose(sum3_2b_1, [0,1,3,2])
        sum3_2b = sum3_2b_2 - np.transpose(sum3_2b_2, [1,0,2,3])
        # sum3_2b_1 = tn.ncon([eta2B, G], [(0, -1, 1, -3), (1, -2, 0, -4)])#.numpy()
        # sum3_2b_2 = np.transpose(sum3_2b_1, [1, 0, 2, 3])
        # sum3_2b_3 = np.transpose(sum3_2b_1, [0, 1, 3, 2])
        # sum3_2b_4 = np.transpose(sum3_2b_1, [1, 0, 3, 2])
        # sum3_2b_5 = sum3_2b_1 - sum3_2b_2 - sum3_2b_3 + sum3_2b_4
        # sum3_2b = tn.ncon([occA, sum3_2b_5], [(0, 1, -1, -2), (0, 1, -3, -4)])#.numpy()

        dG = sum1_2b + 0.5*sum2_2b + sum3_2b

        return (dE, df, dG)

class Flow_IMSRG3(Flow_IMSRG2):
    """Calculates the flow equations for the IMSRG(3). Inherits from Flow_IMSRG2."""

    def __init__(self, h, occ_t):
        """Class constructor. Instantiates Flow_IMSRG3 object.

        Arguments:

        h -- Hamiltonian object
        occ_t -- OccupationTensors object"""

        assert isinstance(h, Hamiltonian), "Arg 0 must be PairingHamiltonian object"
        assert isinstance(occ_t, OccupationTensors), "Arg 1 must be OccupationTensors object"

        # self.f = h.f
        # self.G = h.G

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

    def flow(self, gen):
        """Iterates the IMSRG3 flow equations once. Extends IMSRG2 flow function.

        Arugments:

        gen -- Generator object; generator produces the flow

        Returns:

        (dE, -- zero-body tensor
         df, -- one-body tensor
         dG, -- two-body tensor
         dW) -- three-body tensor"""

        assert isinstance(gen, Generator), "Arg 0 must be WegnerGenerator object"

        # f = self.f
        # G = self.G
        f = gen.f
        G = gen.G
        W = gen.W

        partition = super().flow(gen)
        dE = partition[0]
        df = partition[1]
        dG = partition[2]

        partition = gen.calc_eta()
        eta1B = partition[0]
        eta2B = partition[1]
        eta3B = partition[2]

        occA = self._occA
        occA2 = self._occA2
        occB = self._occB
        occC = self._occC
        occD = self._occD
        occE = self._occE
        occF = self._occF
        occG = self._occG
        occH = self._occH
        occI = self._occI
        occJ = self._occJ

        # Calculate 0B flow equation
        sum3_0b_1 = np.matmul(eta3B, occE)
        sum3_0b = tn.ncon([sum3_0b_1, W], [(1,2,3,4,5,6), (4,5,6,1,2,3)])#.numpy()
        dE += (1/18)*sum3_0b

        # Calculate 1B flow equation
        # fourth term
        sum4_1b_1 = np.matmul(np.transpose(occD,[2,3,0,1]), G)
        sum4_1b_2 = np.matmul(np.transpose(occD,[2,3,0,1]), eta2B)
        sum4_1b_3 = tn.ncon([eta3B,  sum4_1b_1], [(1,2,-1,3,4,-2),(3,4,1,2)])#.numpy()
        sum4_1b_4 = tn.ncon([W, sum4_1b_2], [(1,2,-1,3,4,-2),(3,4,1,2)])#.numpy()
        sum4_1b = sum4_1b_3 - sum4_1b_4

        # fifth term
        sum5_1b_1 = tn.ncon([eta3B, occF, W], [(6,7,-1,8,9,10),
                                               (6,7,8,9,10,1,2,3,4,5),
                                               (3,4,5,1,2,-2)])#.numpy()
        sum5_1b_2 = tn.ncon([W, occF, eta3B], [(6,7,-1,8,9,10),
                                               (6,7,8,9,10,1,2,3,4,5),
                                               (3,4,5,1,2,-2)])#.numpy()
        sum5_1b = sum5_1b_1 - sum5_1b_2
        
        # sum5_1b_1 = tn.ncon([occF, eta3B.astype(np.float32)],
        #                  [(-1,-3,-4,-5,-6,0,1,2,3,4), (0,1,-2,2,3,4)])#.numpy()
        # sum5_1b_2 = tn.ncon([occF, W.astype(np.float32)],
        #                  [(-1,-3,-4,-5,-6,0,1,2,3,4), (0,1,-2,2,3,4)])#.numpy()
        # sum5_1b_3 = tn.ncon([sum5_1b_1, W.astype(np.float32)],
        #                  [(0,1,-1,2,3,4), (2,3,4,0,1,-2)])#.numpy()
        # sum5_1b_4 = tn.ncon([sum5_1b_2, eta3B.astype(np.float32)],
        #                  [(0,1,-1,2,3,4), (2,3,4,0,1,-2)])#.numpy()
        # sum5_1b = sum5_1b_3 - sum5_1b_4

        df += 0.25*sum4_1b + (1/12)*sum5_1b

        # Calculate 2B flow equation
        # fourth term
        sum4_2b_1 = np.matmul(-1.0*np.transpose(occA2), f)
        sum4_2b_2 = np.matmul(-1.0*np.transpose(occA2),  eta1B)
        sum4_2b_3 = tn.ncon([eta3B,  sum4_2b_1], [(1,-1,-2,2,-3,-4), (2,1)])#.numpy()
        sum4_2b_4 = tn.ncon([W, sum4_2b_2], [(1,-1,-2,2,-3,-4), (2,1)])#.numpy()
        sum4_2b = sum4_2b_3 - sum4_2b_4

        #fifth term
        sum5_2b_1 = tn.ncon([eta3B, occG, G], [(4,-1,-2,5,6,-4),
                                               (4,5,6,1,2,3),
                                               (2,3,1,-3)])#.numpy()
        sum5_2b_2 = tn.ncon([W, occG, eta2B], [(4,-1,-2,5,6,-4),
                                               (4,5,6,1,2,3),
                                               (2,3,1,-3)])#.numpy()

        sum5_2b = sum5_2b_2 - np.transpose(sum5_2b_2, [3,2,0,1]) - \
                    np.transpose(sum5_2b_2, [0,1,3,2]) + \
                    np.transpose(sum5_2b_2, [2,3,0,1])
        
        # sum5_2b_1 = tn.ncon([occG, G], [(-1,-2,-4,0,1,2), (1,2,0,-3)])#.numpy()
        # sum5_2b_2 = tn.ncon([occG,  eta2B], [(-1,-2,-4,0,1,2), (1,2,0,-3)])#.numpy()
        # sum5_2b_3 = tn.ncon([eta3B,  sum5_2b_1], [(0,-1,-2,1,2,-4), (1,2,0,-3)])#.numpy()
        # sum5_2b_4 = tn.ncon([W, sum5_2b_2], [(0,-1,-2,1,2,-4), (1,2,0,-3)])#.numpy()
        # sum5_2b_5 = sum5_2b_3 - sum5_2b_4
        # sum5_2b = sum5_2b_5 - np.transpose(sum5_2b_5, [3,2,0,1]) - \
        #             np.transpose(sum5_2b_5, [0,1,3,2]) + \
        #             np.transpose(sum5_2b_5, [2,3,0,1])

        #sixth term
        sum6_2b_1 = tn.ncon([eta3B, occH, W], [(5,-1,-2,6,7,8),
                                               (5,6,7,8,1,2,3,4),
                                               (2,3,4,1,-3,-4)])#.numpy()
        sum6_2b_2 = tn.ncon([eta3B, occH, W], [(6,7,8,5,-3,-4),
                                               (5,6,7,8,1,2,3,4),
                                               (1,-1,-2,2,3,4)])#.numpy()
        sum6_2b = sum6_2b_1 - sum6_2b_2

        # sum6_2b_1 = tn.ncon([occH, W], [(-1,-2,-3,-4,0,1,2,3),(1,2,3,0,-5,-6)])#.numpy()
        # sum6_2b_2 = tn.ncon([occH, W], [(-3,-4,-5,-6,0,1,2,3),(0,-1,-2,1,2,3)])#.numpy()
        # sum6_2b_3 = tn.ncon([eta3B, sum6_2b_1], [(0,-1,-2,1,2,3), (1,2,3,0,-3,-4)])#.numpy()
        # sum6_2b_4 = tn.ncon([eta3B, sum6_2b_2], [(1,2,3,0,-3,-4), (0,-1,-2,1,2,3)])#.numpy()
        # sum6_2b = sum6_2b_3 - sum6_2b_4

        #seventh term
        sum7_2b_1 = tn.ncon([eta3B, occI, W], [(5,6,-1,7,8,-4),
                                               (5,6,7,8,1,2,3,4),
                                               (3,4,-2,1,2,-3)])#.numpy()
        sum7_2b = sum7_2b_1 - np.transpose(sum7_2b_1,[1,0,2,3]) - \
                              np.transpose(sum7_2b_1,[0,1,3,2]) + \
                              np.transpose(sum7_2b_1,[1,0,3,2])
        
        # sum7_2b_1 = tn.ncon([occI, W], [(-1,-2,-3,-4,0,1,2,3), (2,3,-5,0,1,-6)])#.numpy()
        # sum7_2b_2 = tn.ncon([eta3B, sum7_2b_1], [(0,1,-1,2,3,-4),(2,3,-2,0,1,-3)])#.numpy()
        # sum7_2b = sum7_2b_2 - np.transpose(sum7_2b_2,[1,0,2,3]) - \
        #                       np.transpose(sum7_2b_2,[0,1,3,2]) + \
        #                       np.transpose(sum7_2b_2,[1,0,3,2])

        dG += sum4_2b + 0.5*sum5_2b + (1/6)*sum6_2b + 0.25*sum7_2b

        # Calculate 3B flow equation
        #first, second, and third terms (one index contraction)

        #terms with P(i/jk) -- line 1 and 2
        sum1_3b_1 = tn.ncon([eta1B, W], [(-1,1), (1,-2,-3,-4,-5,-6)])#.numpy()
        sum1_3b_2 = tn.ncon([f, eta3B], [(-1,1), (1,-2,-3,-4,-5,-6)])#.numpy()
        sum1_3b_3 = sum1_3b_1 - sum1_3b_2
        sum1_3b_4 = sum1_3b_3 - np.transpose(sum1_3b_3, [1,0,2,3,4,5]) - \
                                np.transpose(sum1_3b_3, [2,1,0,3,4,5])

        #terms with P(l/mn) -- line 1 and 2
        sum1_3b_5 = tn.ncon([eta1B, W], [(1,-4), (-1,-2,-3,1,-5,-6)])#.numpy()
        sum1_3b_6 = tn.ncon([f, eta3B], [(1,-4), (-1,-2,-3,1,-5,-6)])#.numpy()
        sum1_3b_7 = sum1_3b_6 - sum1_3b_5
        sum1_3b_8 = sum1_3b_7 - np.transpose(sum1_3b_7, [0,1,2,4,3,5]) - \
                                np.transpose(sum1_3b_7, [0,1,2,5,4,3])

        #terms with P(ij/k)P(l/mn) -- line 3
        sum1_3b_9  = tn.ncon([eta2B, G], [(-1,-2,-4,1),(1,-3,-5,-6)])
        sum1_3b_10 = tn.ncon([G, eta2B], [(-1,-2,-4,1),(1,-3,-5,-6)])
        sum1_3b_11 = sum1_3b_9 - sum1_3b_10
        sum1_3b_12 = sum1_3b_11 - np.transpose(sum1_3b_11, [0,1,2,4,3,5]) - \
                                  np.transpose(sum1_3b_11, [0,1,2,5,4,3])
        sum1_3b_13 = sum1_3b_12 - np.transpose(sum1_3b_12, [2,1,0,3,4,5]) - \
                                  np.transpose(sum1_3b_12, [0,2,1,3,4,5])

        sum1_3b = sum1_3b_4 + sum1_3b_8 + sum1_3b_13

        #fourth term
        sum4_3b_1 = tn.ncon([eta2B, occB, W], [(-1,-2,3,4),(3,4,1,2),(1,2,-3,-4,-5,-6)])#.numpy()
        sum4_3b_2 = tn.ncon([G, occB, eta3B], [(-1,-2,3,4),(3,4,1,2),(1,2,-3,-4,-5,-6)])#.numpy()
        sum4_3b_3 = sum4_3b_1 - sum4_3b_2
        sum4_3b = sum4_3b_3 - np.transpose(sum4_3b_3, [1,0,2,3,4,5]) - \
                              np.transpose(sum4_3b_3, [2,1,0,3,4,5])

        #fifth term
        sum5_3b_1 = tn.ncon([eta2B, occB, W], [(3,4,-4,-5),(3,4,1,2),(-1,-2,-3,1,2,-6)])#.numpy()
        sum5_3b_2 = tn.ncon([G, occB, eta3B], [(3,4,-4,-5),(3,4,1,2),(-1,-2,-3,1,2,-6)])#.numpy()
        sum5_3b_3 = sum5_3b_1 - sum5_3b_2
        sum5_3b = sum5_3b_3 - np.transpose(sum5_3b_3, [0,1,2,5,4,3]) - \
                              np.transpose(sum5_3b_3, [0,1,2,3,5,4])

        #sixth term
        sum6_3b_1 = tn.ncon([eta2B, occA, W], [(4,-1,3,-4),(3,4,1,2),(1,-2,-3,2,-5,-6)])#.numpy()
        sum6_3b_2 = tn.ncon([G, occA, eta3B], [(4,-1,3,-4),(3,4,1,2),(1,-2,-3,2,-5,-6)])#.numpy()
        sum6_3b_3 = sum6_3b_1 - sum6_3b_2
        sum6_3b_4 = sum6_3b_3 - np.transpose(sum6_3b_3, [0,1,2,4,3,5]) - \
                                np.transpose(sum6_3b_3, [0,1,2,5,4,3])
        sum6_3b = sum6_3b_4 - np.transpose(sum6_3b_4, [1,0,2,3,4,5]) - \
                              np.transpose(sum6_3b_4, [2,1,0,3,4,5])

        #seventh term
        sum7_3b_1 = tn.ncon([eta3B, occJ, W], [(-1,-2,-3,4,5,6), (4,5,6,1,2,3), (1,2,3,-4,-5,-6)])#.numpy()
        sum7_3b_2 = tn.ncon([W, occJ, eta3B], [(-1,-2,-3,4,5,6), (4,5,6,1,2,3), (1,2,3,-4,-5,-6)])#.numpy()
        sum7_3b = sum7_3b_1 - sum7_3b_2

        #eighth term
        sum8_3b_1 = tn.ncon([eta3B, occC, W], [(4,5,-3,6,-5,-6), (4,5,6,1,2,3), (3,-1,-2,1,2,-4)])#.numpy()
        sum8_3b_2 = tn.ncon([eta3B, occC, W], [(6,-2,-3,4,5,-6), (4,5,6,1,2,3), (-1,1,2,-4,-5,3)])#.numpy()
        sum8_3b_3 = sum8_3b_1 - sum8_3b_2
        sum8_3b_4 = sum8_3b_3 - np.transpose(sum8_3b_3, [0,1,2,4,3,5]) - \
                                np.transpose(sum8_3b_3, [0,1,2,5,4,3])
        sum8_3b = sum8_3b_4 - np.transpose(sum8_3b_4, [2,1,0,3,4,5]) - \
                              np.transpose(sum8_3b_4, [0,2,1,3,4,5])

        dW = sum1_3b + 0.5*sum4_3b + (-0.5)*sum5_3b + (-1.0)*sum6_3b + (1/6)*sum7_3b + 0.5*sum8_3b

        return (dE, df, dG, dW)
