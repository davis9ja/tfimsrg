#import tensorflow as tf
# tf.enable_v2_behavior()
#import tensornetwork as tn
import numpy as np
from tfimsrg.oop_imsrg.hamiltonian import *
from tfimsrg.oop_imsrg.occupation_tensors import *
from tfimsrg.oop_imsrg.generator import *
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

        #assert isinstance(h, Hamiltonian), "Arg 0 must be PairingHamiltonian object"
        assert isinstance(occ_t, OccupationTensors), "Arg 1 must be OccupationTensors object"

        # self.f = h.f
        # self.G = h.G

        self._holes = h.holes
        self._particles = h.particles

        self._occA = occ_t.occA
        self._occA4 = occ_t.occA4
        self._occB = occ_t.occB
        self._occB4 = occ_t.occB4
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

    def flow(self, E, f, G, gen):
        """Iterates the IMSRG2 flow equations once.

        Arugments:

        gen -- Generator object; generator produces the flow

        Returns:

        (dE, -- zero-body tensor
         df, -- one-body tensor
         dG) -- two-body tensor"""

        assert isinstance(gen, Generator), "Arg 0 must be Generator object"

        # f = self.f
        # G = self.G
        # f = gen.f.astype(np.float32)
        # G = gen.G.astype(np.float32)

        eta1B, eta2B = gen.calc_eta()
        # eta1B = partition[0]#.astype(np.float32)
        # eta2B = partition[1]#.astype(np.float32)

        # occA = occ_t.occA
        # occB = occ_t.occB
        # occC = occ_t.occC
        # occD = occ_t.occD

        occA = self._occA
        occA4 = self._occA4
        occB = self._occB
        occB4 = self._occB4
        occC = self._occC
        occD = self._occD

        # - Calculate dE/ds
        # first term
        #sum1_0b = tn.ncon([eta1B, occA, f], [(3,4), (3,4,1,2), (2,1)])#.numpy()
        
        sum1_0b_1 = np.multiply(occA.tensor, eta1B)
        sum1_0b = tn.ncon([sum1_0b_1, f], [(1,2),(2,1)])

        # sum1_0b_1 = tn.ncon([occA, eta1B], [(0, 1, -1, -2), (0, 1)])#.numpy()
        # sum1_0b = tn.ncon([sum1_0b_1, f], [(0, 1), (1, 0)])#.numpy()

        # second term
        #print(occD.shape)
        #sum2_0b = tn.ncon([eta2B, occD, G], [(5,6,7,8), (5,6,7,8,1,2,3,4), (3,4,1,2)])#.numpy()

        #sum2_0b_1 = tn.ncon([eta2B, occD],[(-1,-2,1,2),(-3,-4,1,2)])
        sum2_0b_1 = np.multiply(eta2B, occD.tensor)
        sum2_0b = tn.ncon([sum2_0b_1, G], [(1, 2, 3, 4), (3, 4, 1, 2)])#.numpy()

        dE = sum1_0b + 0.5*sum2_0b


        # - Calculate df/ds
        # first term
        sum1_1b_1 = tn.ncon([eta1B, f], [(-1, 1), (1, -2)])#.numpy()
        sum1_1b_2 = np.transpose(sum1_1b_1)
        sum1_1b = sum1_1b_1 + sum1_1b_2

        # second term (might need to fix)
        #sum2_1b_1 = tn.ncon([eta1B, occA, G], [(3,4), (3,4,1,2), (2,-1,1,-2)])#.numpy()
        #sum2_1b_2 = tn.ncon([f, occA, eta2B], [(3,4), (3,4,1,2), (2,-1,1,-2)])#.numpy()

        eta1BPrime = np.multiply(occA.tensor, eta1B)
        fPrime = np.multiply(occA.tensor, f)
        sum2_1b_1 = tn.ncon([eta1BPrime, G], [(1,2), (2,-1,1,-2)])
        sum2_1b_2 = tn.ncon([fPrime, eta2B], [(1,2), (2,-1,1,-2)])
        sum2_1b = sum2_1b_1 - sum2_1b_2

        # sum2_1b_1 = tn.ncon([eta1B, G], [(0, 1), (1, -1, 0, -2)])#.numpy()
        # sum2_1b_2 = tn.ncon([f, eta2B], [(0, 1), (1, -1, 0, -2)])#.numpy()
        # sum2_1b_3 = sum2_1b_1 - sum2_1b_2
        # sum2_1b = tn.ncon([occA, sum2_1b_3],[(-1, -2, 0, 1), (0,1)])#.numpy()

        # third term
        #sum3_1b_1 = tn.ncon([eta2B, occC, G], [(6,-1,4,5), (4,5,6,1,2,3), (1,2,3,-2)])#.numpy()
        #sum3_1b = sum3_1b_1 + np.transpose(sum3_1b_1)

        sum3_1b_1 = np.multiply(occC.tensor, G) #np.multiply(tn.outer_product(tn.Node(occC), tn.Node(np.ones(8))).tensor, G)
        sum3_1b_2 = tn.ncon([eta2B, G], [(3,-1,1,2),(1,2,3,-2)])
        sum3_1b = sum3_1b_2 + np.transpose(sum3_1b_2)

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
        # sum2_2b_1 = tn.ncon([eta2B, occB, G], [(-1,-2,3,4), (3,4,1,2), (1,2,-3,-4)])#.numpy()
        # sum2_2b_2 = tn.ncon([G, occB, eta2B], [(-1,-2,3,4), (3,4,1,2), (1,2,-3,-4)])#.numpy()

        GPrime = np.multiply(occB4.tensor, G)
        eta2BPrime = np.multiply(occB4.tensor, eta2B)
        sum2_2b_1 = tn.ncon([eta2B, GPrime], [(-1,-2,1,2), (1,2,-3,-4)])
        sum2_2b_2 = tn.ncon([G, eta2BPrime], [(-1,-2,1,2), (1,2,-3,-4)])


#        sum2_2b_1 = tn.ncon([eta2B, occB, occB, G], [(-1,-2,3,4), (3,4),(1,2), (1,2,-3,-4)])#.numpy()
#        sum2_2b_2 = tn.ncon([G, occB, occB, eta2B], [(-1,-2,3,4), (3,4),(1,2), (1,2,-3,-4)])#.numpy()

        sum2_2b = sum2_2b_1 - sum2_2b_2

        # sum2_2b_1 = tn.ncon([occB,     G], [(-1, -2, 0, 1), (0, 1, -3, -4)])#.numpy()
        # sum2_2b_2 = tn.ncon([occB, eta2B], [(-1, -2, 0, 1), (0, 1, -3, -4)])#.numpy()
        # sum2_2b_3 = tn.ncon([eta2B,  sum2_2b_1], [(-1, -2, 0, 1), (0, 1, -3, -4)])#.numpy()
        # sum2_2b_4 = tn.ncon([G,      sum2_2b_2], [(-1, -2, 0, 1), (0, 1, -3, -4)])#.numpy()
        # sum2_2b = sum2_2b_3 - sum2_2b_4

        # third term
        # sum3_2b_1 = tn.ncon([eta2B, occA, G], [(3,-1,4,-3), (3,4,1,2), (2,-2,1,-4)])#.numpy()
        # sum3_2b_2 = sum3_2b_1 - np.transpose(sum3_2b_1, [0,1,3,2])
        # sum3_2b = sum3_2b_2 - np.transpose(sum3_2b_2, [1,0,2,3])

        GPrime = np.multiply(np.transpose(occA4.tensor, [0,2,1,3]), G)
        sum3_2b_1 = tn.ncon([eta2B, GPrime], [(2,-2,1,-4), (1,-1,2,-3)])
        sum3_2b_2 = sum3_2b_1 - np.transpose(sum3_2b_1, [0,1,3,2])
        sum3_2b = sum3_2b_2 - np.transpose(sum3_2b_2, [1,0,2,3])


        # sum3_2b_1 = tn.ncon([eta2B, G], [(0, -1, 1, -3), (1, -2, 0, -4)])#.numpy()
        # sum3_2b_2 = np.transpose(sum3_2b_1, [1, 0, 2, 3])
        # sum3_2b_3 = np.transpose(sum3_2b_1, [0, 1, 3, 2])
        # sum3_2b_4 = np.transpose(sum3_2b_1, [1, 0, 3, 2])
        # sum3_2b_5 = sum3_2b_1 - sum3_2b_2 - sum3_2b_3 + sum3_2b_4
        # sum3_2b = tn.ncon([occA, sum3_2b_5], [(0, 1, -1, -2), (0, 1, -3, -4)])#.numpy()

        dG = sum1_2b + 0.5*sum2_2b - sum3_2b

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

        self._h = h
        self._occ_t = occ_t

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

        gen2b = WegnerGenerator(self._h, self._occ_t)
        gen2b.f = f
        gen2b.G = G
        
        partition = super().flow(f,G,gen2b)
        dE = partition[0]
        df = partition[1]
        dG = partition[2]
        del gen2b

        partition = gen.calc_eta()
        eta1B = partition[0]
        eta2B = partition[1]
        eta3B = partition[2]

        occA = self._occA
        occA4 = self._occA4
        occB = self._occB
        occB4 = self._occB4
        occC = self._occC
        occC6 = self._occC6
        occD = self._occD
        occDv2 = self._occDv2
        occE = self._occE
        occF = self._occF
        occG = self._occG
        occH = self._occH
        occI = self._occI
        occJ = self._occJ

        # Calculate 0B flow equation
        sum3_0b_1 = np.multiply(eta3B, occE)
        sum3_0b = tn.ncon([sum3_0b_1, W], [(1,2,3,4,5,6), (4,5,6,1,2,3)])#.numpy()
        dE += (1/18)*sum3_0b

        # Calculate 1B flow equation
        # fourth term
        sum4_1b_1 = np.multiply(np.transpose(occDv2.tensor), eta2B)
        sum4_1b_2 = np.multiply(np.transpose(occDv2.tensor), G)
        sum4_1b_3 = tn.ncon([W,     sum4_1b_1], [(3,4,-1,1,2,-2),(1,2,3,4)])#.numpy()
        sum4_1b_4 = tn.ncon([eta3B, sum4_1b_2], [(3,4,-1,1,2,-2),(1,2,3,4)])#.numpy()
        sum4_1b = sum4_1b_3 - sum4_1b_4

        # fifth term
        # sum5_1b_1 = tn.ncon([eta3B, occF, W], [(6,7,-1,8,9,10),
        #                                        (6,7,8,9,10,1,2,3,4,5),
        #                                        (3,4,5,1,2,-2)])#.numpy()
        # sum5_1b_2 = tn.ncon([W, occF, eta3B], [(6,7,-1,8,9,10),
        #                                        (6,7,8,9,10,1,2,3,4,5),
        #                                        (3,4,5,1,2,-2)])#.numpy()
        # sum5_1b = sum5_1b_1 - sum5_1b_2

        
        sum5_1b_1 = np.multiply(np.transpose(occF.tensor,[0,1,5,2,3,4]), eta3B)
        sum5_1b_2 = np.multiply(np.transpose(occF.tensor,[0,1,5,2,3,4]), W)
        sum5_1b_3 = tn.ncon([sum5_1b_1,     W],[(1,2,-1,3,4,5),(3,4,5,1,2,-2)])
        sum5_1b_4 = tn.ncon([sum5_1b_2, eta3B],[(1,2,-1,3,4,5),(3,4,5,1,2,-2)])

        sum5_1b = sum5_1b_3 - sum5_1b_4
        
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
        sum4_2b_1 = np.multiply(-1.0*np.transpose(occA.tensor), f)
        sum4_2b_2 = np.multiply(-1.0*np.transpose(occA.tensor),  eta1B)
        sum4_2b_3 = tn.ncon([eta3B,  sum4_2b_1], [(1,-1,-2,2,-3,-4), (2,1)])#.numpy()
        sum4_2b_4 = tn.ncon([W, sum4_2b_2], [(1,-1,-2,2,-3,-4), (2,1)])#.numpy()
        sum4_2b = sum4_2b_3 - sum4_2b_4

        #fifth term
        sum5_2b_1 = np.multiply(np.transpose(occG.tensor, [1,2,0,3]), G)
        sum5_2b_2 = np.multiply(np.transpose(occG.tensor, [1,2,0,3]), eta2B)

        sum5_2b_3 = tn.ncon([eta3B, sum5_2b_1],[(1,-1,-2,2,3,-4),(2,3,1,-3)])
        sum5_2b_4 = tn.ncon([W,     sum5_2b_2],[(1,-1,-2,2,3,-4),(2,3,1,-3)])

        sum5_2b_5 = sum5_2b_3 - sum5_2b_4

        sum5_2b = sum5_2b_5 - np.transpose(sum5_2b_5, [2,3,1,0]) - \
                    np.transpose(sum5_2b_5, [0,1,3,2]) + \
                    np.transpose(sum5_2b_5, [2,3,0,1])


        # sum5_2b_1 = tn.ncon([eta3B, occG, G], [(4,-1,-2,5,6,-4),
        #                                        (4,5,6,1,2,3),
        #                                        (2,3,1,-3)])#.numpy()
        # sum5_2b_2 = tn.ncon([W, occG, eta2B], [(4,-1,-2,5,6,-4),
        #                                        (4,5,6,1,2,3),
        #                                        (2,3,1,-3)])#.numpy()

        # sum5_2b = sum5_2b_2 - np.transpose(sum5_2b_2, [3,2,0,1]) - \
        #             np.transpose(sum5_2b_2, [0,1,3,2]) + \
        #             np.transpose(sum5_2b_2, [2,3,0,1])
        
        # sum5_2b_1 = tn.ncon([occG, G], [(-1,-2,-4,0,1,2), (1,2,0,-3)])#.numpy()
        # sum5_2b_2 = tn.ncon([occG,  eta2B], [(-1,-2,-4,0,1,2), (1,2,0,-3)])#.numpy()
        # sum5_2b_3 = tn.ncon([eta3B,  sum5_2b_1], [(0,-1,-2,1,2,-4), (1,2,0,-3)])#.numpy()
        # sum5_2b_4 = tn.ncon([W, sum5_2b_2], [(0,-1,-2,1,2,-4), (1,2,0,-3)])#.numpy()
        # sum5_2b_5 = sum5_2b_3 - sum5_2b_4
        # sum5_2b = sum5_2b_5 - np.transpose(sum5_2b_5, [3,2,0,1]) - \
        #             np.transpose(sum5_2b_5, [0,1,3,2]) + \
        #             np.transpose(sum5_2b_5, [2,3,0,1])

        #sixth term
        
        sum5_2b_1 = np.multiply(np.transpose(occH.tensor,[0,4,5,1,2,3]), eta3B)
        sum5_2b_2 = np.multiply(np.transpose(occH.tensor,[0,4,5,1,2,3]), W)
        sum5_2b_3 = tn.ncon([sum5_2b_1,     W], [(1,-1,-2,2,3,4),(2,3,4,1,-3,-4)])
        sum5_2b_4 = tn.ncon([eta3B, sum5_2b_2], [(1,-1,-2,2,3,4),(2,3,4,1,-3,-4)])
        sum6_2b = sum5_2b_3 - sum5_2b_4
        
        # sum6_2b_1 = tn.ncon([eta3B, occH, W], [(5,-1,-2,6,7,8),
        #                                        (5,6,7,8,1,2,3,4),
        #                                        (2,3,4,1,-3,-4)])#.numpy()
        # sum6_2b_2 = tn.ncon([eta3B, occH, W], [(6,7,8,5,-3,-4),
        #                                        (5,6,7,8,1,2,3,4),
        #                                        (1,-1,-2,2,3,4)])#.numpy()
        # sum6_2b = sum6_2b_1 - sum6_2b_2

        # sum6_2b_1 = tn.ncon([occH, W], [(-1,-2,-3,-4,0,1,2,3),(1,2,3,0,-5,-6)])#.numpy()
        # sum6_2b_2 = tn.ncon([occH, W], [(-3,-4,-5,-6,0,1,2,3),(0,-1,-2,1,2,3)])#.numpy()
        # sum6_2b_3 = tn.ncon([eta3B, sum6_2b_1], [(0,-1,-2,1,2,3), (1,2,3,0,-3,-4)])#.numpy()
        # sum6_2b_4 = tn.ncon([eta3B, sum6_2b_2], [(1,2,3,0,-3,-4), (0,-1,-2,1,2,3)])#.numpy()
        # sum6_2b = sum6_2b_3 - sum6_2b_4

        #seventh term
        sum7_2b_1 = np.multiply(np.transpose(occI.tensor,[0,1,4,2,3,5]),eta3B)
        sum7_2b_2 = tn.ncon([sum7_2b_1, W], [(1,2,-1,3,4,-4),(3,4,-2,1,2,-3)])
        sum7_2b = sum7_2b_2 - np.transpose(sum7_2b_2,[1,0,2,3]) - \
                              np.transpose(sum7_2b_2,[0,1,3,2]) + \
                              np.transpose(sum7_2b_2,[1,0,3,2])        
        
        # sum7_2b_1 = tn.ncon([eta3B, occI, W], [(5,6,-1,7,8,-4),
        #                                        (5,6,7,8,1,2,3,4),
        #                                        (3,4,-2,1,2,-3)])#.numpy()
        # sum7_2b = sum7_2b_1 - np.transpose(sum7_2b_1,[1,0,2,3]) - \
        #                       np.transpose(sum7_2b_1,[0,1,3,2]) + \
        #                       np.transpose(sum7_2b_1,[1,0,3,2])
        
        # sum7_2b_1 = tn.ncon([occI, W], [(-1,-2,-3,-4,0,1,2,3), (2,3,-5,0,1,-6)])#.numpy()
        # sum7_2b_2 = tn.ncon([eta3B, sum7_2b_1], [(0,1,-1,2,3,-4),(2,3,-2,0,1,-3)])#.numpy()
        # sum7_2b = sum7_2b_2 - np.transpose(sum7_2b_2,[1,0,2,3]) - \
        #                       np.transpose(sum7_2b_2,[0,1,3,2]) + \
        #                       np.transpose(sum7_2b_2,[1,0,3,2])

        dG += sum4_2b + 0.5*sum5_2b + (1/6)*sum6_2b + 0.25*sum7_2b

        # Calculate 3B flow equation
        #first, second, and third terms (one index contraction)

        #terms with P(i/jk) -- line 1 and 2
        #P(i/jk) = 1 - Pij - Pik
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
        #P(ij/k) = 1 - Pik - Pjk
        sum1_3b_9  = tn.ncon([eta2B, G], [(-1,-2,-4,1),(1,-3,-5,-6)])
        sum1_3b_10 = tn.ncon([G, eta2B], [(-1,-2,-4,1),(1,-3,-5,-6)])
        sum1_3b_11 = sum1_3b_9 - sum1_3b_10
        sum1_3b_12 = sum1_3b_11 - np.transpose(sum1_3b_11, [0,1,2,4,3,5]) - \
                                  np.transpose(sum1_3b_11, [0,1,2,5,4,3])
        sum1_3b_13 = sum1_3b_12 - np.transpose(sum1_3b_12, [2,1,0,3,4,5]) - \
                                  np.transpose(sum1_3b_12, [0,2,1,3,4,5])

        sum1_3b = sum1_3b_4 + sum1_3b_8 + sum1_3b_13

        #fourth term
        sum4_3b_1 = np.multiply(np.transpose(occB4.tensor,[2,3,0,1]), eta2B)
        sum4_3b_2 = np.multiply(np.transpose(occB4.tensor,[2,3,0,1]), G)
        sum4_3b_3 = tn.ncon([sum4_3b_1,     W], [(-1,-2,1,2),(1,2,-3,-4,-5,-6)])
        sum4_3b_4 = tn.ncon([sum4_3b_2, eta3B], [(-1,-2,1,2),(1,2,-3,-4,-5,-6)])

        sum4_3b_5 = sum4_3b_3 - sum4_3b_4
        sum4_3b = sum4_3b_5 - np.transpose(sum4_3b_5, [1,0,2,3,4,5]) - \
                              np.transpose(sum4_3b_5, [2,1,0,3,4,5])


        # sum4_3b_1 = tn.ncon([eta2B, occB4, W], [(-1,-2,3,4),(3,4,1,2),(1,2,-3,-4,-5,-6)])#.numpy()
        # sum4_3b_2 = tn.ncon([G, occB4, eta3B], [(-1,-2,3,4),(3,4,1,2),(1,2,-3,-4,-5,-6)])#.numpy()
        # sum4_3b_3 = sum4_3b_1 - sum4_3b_2
        # sum4_3b = sum4_3b_3 - np.transpose(sum4_3b_3, [1,0,2,3,4,5]) - \
        #                       np.transpose(sum4_3b_3, [2,1,0,3,4,5])

        #fifth term
        sum5_3b_1 = np.multiply(occB4.tensor, eta2B)
        sum5_3b_2 = np.multiply(occB4.tensor, G)
        sum5_3b_3 = tn.ncon([sum5_3b_1,     W], [(1,2,-4,-5),(-1,-2,-3,1,2,-6)])
        sum5_3b_4 = tn.ncon([sum5_3b_2, eta3B], [(1,2,-4,-5),(-1,-2,-3,1,2,-6)])
        sum5_3b_5 = sum5_3b_3 - sum5_3b_4
        sum5_3b = sum5_3b_5 - np.transpose(sum5_3b_5, [0,1,2,5,4,3]) - \
                              np.transpose(sum5_3b_5, [0,1,2,3,5,4])
        

        # sum5_3b_1 = tn.ncon([eta2B, occB4, W], [(3,4,-4,-5),(3,4,1,2),(-1,-2,-3,1,2,-6)])#.numpy()
        # sum5_3b_2 = tn.ncon([G, occB4, eta3B], [(3,4,-4,-5),(3,4,1,2),(-1,-2,-3,1,2,-6)])#.numpy()
        # sum5_3b_3 = sum5_3b_1 - sum5_3b_2
        # sum5_3b = sum5_3b_3 - np.transpose(sum5_3b_3, [0,1,2,5,4,3]) - \
        #                       np.transpose(sum5_3b_3, [0,1,2,3,5,4])

        #sixth term
        sum6_3b_1 = np.multiply(np.transpose(occA4.tensor,[1,2,0,3]), eta2B)
        sum6_3b_2 = np.multiply(np.transpose(occA4.tensor,[1,2,0,3]), G)
        sum6_3b_3 = tn.ncon([sum6_3b_1,     W], [(2,-1,1,-4),(1,-2,-3,2,-5,-6)])
        sum6_3b_4 = tn.ncon([sum6_3b_2, eta3B], [(2,-1,1,-4),(1,-2,-3,2,-5,-6)])
        sum6_3b_5 = sum6_3b_3 - sum6_3b_4
        sum6_3b_6 = sum6_3b_5 - np.transpose(sum6_3b_5, [0,1,2,4,3,5]) - \
                                np.transpose(sum6_3b_5, [0,1,2,5,4,3])
        sum6_3b = sum6_3b_6 - np.transpose(sum6_3b_6, [1,0,2,3,4,5]) - \
                              np.transpose(sum6_3b_6, [2,1,0,3,4,5])

        # sum6_3b_1 = tn.ncon([eta2B, occA4, W], [(4,-1,3,-4),(3,4,1,2),(1,-2,-3,2,-5,-6)])#.numpy()
        # sum6_3b_2 = tn.ncon([G, occA4, eta3B], [(4,-1,3,-4),(3,4,1,2),(1,-2,-3,2,-5,-6)])#.numpy()
        # sum6_3b_3 = sum6_3b_1 - sum6_3b_2
        # sum6_3b_4 = sum6_3b_3 - np.transpose(sum6_3b_3, [0,1,2,4,3,5]) - \
        #                         np.transpose(sum6_3b_3, [0,1,2,5,4,3])
        # sum6_3b = sum6_3b_4 - np.transpose(sum6_3b_4, [1,0,2,3,4,5]) - \
        #                       np.transpose(sum6_3b_4, [2,1,0,3,4,5])

        #seventh term
        sum7_3b_1 = np.multiply(np.transpose(occJ.tensor,[3,4,5,0,1,2]), eta3B)
        sum7_3b_2 = np.multiply(np.transpose(occJ.tensor,[3,4,5,0,1,2]), W)
        sum7_3b_3 = tn.ncon([sum7_3b_1,     W], [(-1,-2,-3,1,2,3),(1,2,3,-4,-5,-6)])
        sum7_3b_4 = tn.ncon([sum7_3b_2, eta3B], [(-1,-2,-3,1,2,3),(1,2,3,-4,-5,-6)])
        sum7_3b = sum7_3b_3 - sum7_3b_4

        # sum7_3b_1 = tn.ncon([eta3B, occJ, W], [(-1,-2,-3,4,5,6), (4,5,6,1,2,3), (1,2,3,-4,-5,-6)])#.numpy()
        # sum7_3b_2 = tn.ncon([W, occJ, eta3B], [(-1,-2,-3,4,5,6), (4,5,6,1,2,3), (1,2,3,-4,-5,-6)])#.numpy()
        # sum7_3b = sum7_3b_1 - sum7_3b_2

        #eighth term
        sum8_3b_1 = np.multiply(np.transpose(occC6.tensor,[0,1,3,2,4,5]), eta3B)
        sum8_3b_2 = np.multiply(np.transpose(occC6.tensor,[2,3,4,0,1,5]), eta3B)
        sum8_3b_3 = tn.ncon([sum8_3b_1, W], [(1,2,-3,3,-5,-6),(3,-1,-2,1,2,-4)])
        sum8_3b_4 = tn.ncon([sum8_3b_2, W], [(3,-2,-3,1,2,-6),(-1,1,2,-4,-5,3)])        
        sum8_3b_5 = sum8_3b_3 - sum8_3b_4
        sum8_3b_6 = sum8_3b_5 - np.transpose(sum8_3b_5, [0,1,2,4,3,5]) - \
                                np.transpose(sum8_3b_5, [0,1,2,5,4,3])
        sum8_3b = sum8_3b_6 - np.transpose(sum8_3b_6, [2,1,0,3,4,5]) - \
                              np.transpose(sum8_3b_6, [0,2,1,3,4,5])

        
        # sum8_3b_1 = tn.ncon([eta3B, occC6, W], [(4,5,-3,6,-5,-6), (4,5,6,1,2,3), (3,-1,-2,1,2,-4)])#.numpy()
        # sum8_3b_2 = tn.ncon([eta3B, occC6, W], [(6,-2,-3,4,5,-6), (4,5,6,1,2,3), (-1,1,2,-4,-5,3)])#.numpy()
        # sum8_3b_3 = sum8_3b_1 - sum8_3b_2
        # sum8_3b_4 = sum8_3b_3 - np.transpose(sum8_3b_3, [0,1,2,4,3,5]) - \
        #                         np.transpose(sum8_3b_3, [0,1,2,5,4,3])
        # sum8_3b = sum8_3b_4 - np.transpose(sum8_3b_4, [2,1,0,3,4,5]) - \
        #                       np.transpose(sum8_3b_4, [0,2,1,3,4,5])

        dW = sum1_3b + 0.5*sum4_3b + (-0.5)*sum5_3b + (-1.0)*sum6_3b + (1/6)*sum7_3b + 0.5*sum8_3b

        return (dE, df, dG, dW)

class Flow_MRIMSRG2(Flow):

    def __init__(self, h, occ_t):
        """Class constructor. Instantiates Flow_IMSRG2 object.

        Arguments:

        h -- Hamiltonian object
        occ_t -- OccupationTensors object"""

        #assert isinstance(h, Hamiltonian), "Arg 0 must be PairingHamiltonian object"
        assert isinstance(occ_t, OccupationTensors), "Arg 1 must be OccupationTensors object"
        assert h._dens_weights is not None, "initialize H with DM weights to use MR-IMSRG"
        # self.f = h.f
        # self.G = h.G

        self._holes = h.holes
        self._particles = h.particles

        self._occA = occ_t.occA
        self._occA4 = occ_t.occA4
        self._occB = occ_t.occB
        self._occB4 = occ_t.occB4
        self._occC = occ_t.occC
        self._occD = occ_t.occD
        self._occG = occ_t.occG

        # rho1b = density_1b(len(h.holes), len(h.particles), weights=h._dens_weights)
        # rho2b = density_2b(len(h.holes), len(h.particles), weights=h._dens_weights)
        rho1b = h._rho1b
        rho2b = h._rho2b        
        self._lambda2b = h._lambda2b
        self._lambda3b = h._lambda3b

        numstates = len(self._holes)+len(self._particles)
        print('norm of lambda2b = {: .8f}'.format(np.linalg.norm(np.reshape(self._lambda2b, (numstates**2, numstates**2)))))

        self._h = h

    def get_vacuum_coeffs(self, E, f, G, basis, holes):
            
        H2B = G
        H1B = f - np.trace(G[np.ix_(basis,holes,basis,holes)], axis1=1,axis2=3) 

        Gnode = tn.Node(G[np.ix_(holes,holes,holes,holes)])
        Gnode[0] ^ Gnode[2]
        Gnode[1] ^ Gnode[3]
        result_ij = Gnode @ Gnode


        H0B = E - np.trace(H1B[np.ix_(holes,holes)]) - 0.5*result_ij.tensor
        
        return (H0B, H1B, H2B)


    def flow(self, E, f, G, gen):
        """Iterates the MR-IMSRG2 flow equations once.

        Arugments:

        gen -- Generator object; generator produces the flow

        Returns:

        (dE, -- zero-body tensor
         df, -- one-body tensor
         dG) -- two-body tensor"""

        assert isinstance(gen, Generator), "Arg 0 must be Generator object"

        eta1B, eta2B = gen.calc_eta()

        occA = self._occA
        occA4 = self._occA4
        occB = self._occB
        occB4 = self._occB4
        occC = self._occC
        occD = self._occD
        occG = self._occG

        lambda2b = (self._lambda2b)
        lambda3b = (self._lambda3b)
        
        # - Calculate dG/ds
        # first term (single index sum)
        sum1_2b_1 = tn.ncon([eta1B, G], [(-1,1),(1,-2,-3,-4)])
        sum1_2b_2 = tn.ncon([eta1B, G], [(-2,1),(-1,1,-3,-4)])
        sum1_2b_3 = tn.ncon([eta1B, G], [(1,-3),(-1,-2,1,-4)])
        sum1_2b_4 = tn.ncon([eta1B, G], [(1,-4),(-1,-2,-3,1)])
        sum1_2b_5 = tn.ncon([f, eta2B], [(-1,1),(1,-2,-3,-4)])
        sum1_2b_6 = tn.ncon([f, eta2B], [(-2,1),(-1,1,-3,-4)])
        sum1_2b_7 = tn.ncon([f, eta2B], [(1,-3),(-1,-2,1,-4)])
        sum1_2b_8 = tn.ncon([f, eta2B], [(1,-4),(-1,-2,-3,1)])

        sum1_2b = sum1_2b_1 + sum1_2b_2 - sum1_2b_3 - sum1_2b_4 \
                  - sum1_2b_5 - sum1_2b_6 + sum1_2b_7 + sum1_2b_8

        # second term (two index sum)        
        GPrime = np.multiply(occB4.tensor, G)
        eta2BPrime = np.multiply(occB4.tensor, eta2B)
        sum2_2b_1 = tn.ncon([eta2B, GPrime], [(-1,-2,1,2), (1,2,-3,-4)])
        sum2_2b_2 = tn.ncon([G, eta2BPrime], [(-1,-2,1,2), (1,2,-3,-4)])

        sum2_2b = sum2_2b_1 - sum2_2b_2

        GPrime = np.multiply(np.transpose(occA4.tensor, [2,0,3,1]), G)
        eta2BPrime = np.multiply(np.transpose(occA4.tensor, [2,0,3,1]), eta2B)
        sum3_2b_1 = tn.ncon([eta2BPrime, G], [(-1,1,-3,2), (-2,2,-4,1)])
        sum3_2b_2 = tn.ncon([GPrime, eta2B], [(-1,1,-3,2), (-2,2,-4,1)])
        sum3_2b_3 = tn.ncon([eta2BPrime, G], [(-2,1,-3,2), (-1,2,-4,1)])
        sum3_2b_4 = tn.ncon([GPrime, eta2B], [(-2,1,-3,2), (-1,2,-4,1)])

        sum3_2b = sum3_2b_1 - sum3_2b_2 - sum3_2b_3 + sum3_2b_4

        dG = sum1_2b + 0.5*sum2_2b + sum3_2b

        # numStates = len(self._particles)+len(self._holes)
        # dG_rs = np.reshape(dG, (numStates**2, numStates**2))
        # dG_rs = +1*dG_rs.conj().T
        # dG = np.reshape(dG_rs, (numStates, numStates, numStates, numStates))

        norm = np.linalg.norm
        #print(norm(dG), norm(sum1_2b), norm(sum2_2b), norm(sum3_2b))

        # - Calculate df/ds
        # first term
        sum1_1b_1 = tn.ncon([eta1B, f], [(-1,1), (1,-2)])#.numpy()
        sum1_1b_2 = tn.ncon([f, eta1B], [(-1,1), (1,-2)])
        sum1_1b = sum1_1b_1 - sum1_1b_2

        # second term (might need to fix)
        eta1BPrime = np.multiply(occA.tensor, eta1B)
        fPrime = np.multiply(occA.tensor, f)
        sum2_1b_1 = tn.ncon([eta1BPrime, G], [(1,2), (2,-1,1,-2)])
        sum2_1b_2 = tn.ncon([fPrime, eta2B], [(1,2), (2,-1,1,-2)])
        sum2_1b = sum2_1b_1 - sum2_1b_2

        # third term
        GPrime = np.multiply(np.transpose(occG.tensor,[3,0,1,2]), G) #np.multiply(tn.outer_product(tn.Node(occC), tn.Node(np.ones(8))).tensor, G)
        eta2BPrime = np.multiply(np.transpose(occG.tensor,[3,0,1,2]), eta2B)
        sum3_1b_1 = tn.ncon([eta2BPrime, G], [(-1,1,2,3), (2,3,-2,1)])
        sum3_1b_2 = tn.ncon([GPrime, eta2B], [(-1,1,2,3), (2,3,-2,1)])
        sum3_1b = sum3_1b_1 - sum3_1b_2

        # fourth term (now adding in lambda2b)
        sum4_1b_1 = tn.ncon([eta2B, lambda2b, G], [(-1,1,2,3), (4,5,2,3), (4,5,-2,1)])
        sum4_1b_2 = tn.ncon([G, lambda2b, eta2B], [(-1,1,2,3), (4,5,2,3), (4,5,-2,1)])
        sum4_1b = sum4_1b_1 - sum4_1b_2

        sum5_1b_1 = tn.ncon([eta2B, lambda2b, G], [(-1,1,2,3), (1,5,3,4), (2,5,-2,4)])
        sum5_1b_2 = tn.ncon([G, lambda2b, eta2B], [(-1,1,2,3), (1,5,3,4), (2,5,-2,4)])
        sum5_1b = sum5_1b_1 - sum5_1b_2

        sum6_1b_1 = tn.ncon([eta2B, lambda2b, G], [(-1,1,-2,2), (3,4,2,5), (3,4,1,5)])
        sum6_1b_2 = tn.ncon([G, lambda2b, eta2B], [(-1,1,-2,2), (3,4,2,5), (3,4,1,5)])
        sum6_1b = sum6_1b_1 - sum6_1b_2

        sum7_1b_1 = tn.ncon([eta2B, lambda2b, G], [(-1,1,-2,2), (1,3,4,5), (2,3,4,5)])
        sum7_1b_2 = tn.ncon([G, lambda2b, eta2B], [(-1,1,-2,2), (1,3,4,5), (2,3,4,5)])
        sum7_1b = sum7_1b_1 - sum7_1b_2
        
        #sum2_1b,sum3_1b,sum4_1b,sum5_1b,sum6_1b,sum7_1b =0,0,0,0,0,0

        df = sum1_1b + sum2_1b + 0.5*sum3_1b + 0.25*sum4_1b + sum5_1b - 0.5*sum6_1b + 0.5*sum7_1b
        
        # df = +1*df.conj().T

        # - Calculate dE/ds
        # first term
        sum1_0b_1 = np.multiply(occA.tensor, eta1B)
        sum1_0b = tn.ncon([sum1_0b_1, f], [(1,2),(2,1)])

        # second term
        sum2_0b_1 = np.multiply(eta2B, occD.tensor)
        sum2_0b_2 = np.multiply(G, occD.tensor)
        sum2_0b_3 = tn.ncon([sum2_0b_1, G],     [(1,2,3,4), (3,4,1,2)])
        sum2_0b_4 = tn.ncon([sum2_0b_2, eta2B], [(1,2,3,4), (3,4,1,2)])
        sum2_0b = sum2_0b_3 - sum2_0b_4

        sum3_0b = tn.ncon([dG, lambda2b], [(1,2,3,4), (1,2,3,4)])

        sum4_0b_1 = tn.ncon([eta2B, lambda3b, G], [(1,2,3,4), (2,5,6,3,4,7), (5,6,1,7)])
        sum4_0b_2 = tn.ncon([G, lambda3b, eta2B], [(1,2,3,4), (2,5,6,3,4,7), (5,6,1,7)])
        sum4_0b = sum4_0b_1 - sum4_0b_2

        dE = sum1_0b + 0.25*sum2_0b + 0.25*sum3_0b + 0.25*sum4_0b

        return (dE, df, dG)
        
