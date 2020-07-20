import numpy as np
import os
from oop_imsrg.hamiltonian import *
import sys
import numba

class OccupationTensors(object):
    """Functions as a container for important occupation tensors
    defined by the reference state (usually, ground state) of
    the Hamiltonian. The occupation basis is a vector n that
    contains a 1 if the corresponding single particle basis
    index is occupied, or 0 if the same index is unoccupied,
    in the reference state."""

    DATA_TYPE = float

    def __init__(self, sp_basis, reference):
        """Class constructor. Instantiates an OccupationTensors object.

        Arugments:

        sp_basis -- list containing indices of single particle basis
        reference -- contains reference state as string of 1/0's"""

        self._reference = reference
        self._sp_basis = sp_basis

        self._occA = self.__get_occA()
        self._occA4 = self.__get_occA(flag=1)
        self._occB = self.__get_occB()
        self._occB4 = self.__get_occB(flag=1)
        self._occC = self.__get_occC()
        self._occD = self.__get_occD(flag=1)
        # self._occE = self.__get_occE()
        # self._occF = self.__get_occF()
        # self._occG = self.__get_occG()
        # self._occH = self.__get_occH()
        # self._occI = self.__get_occI()
        # self._occJ = self.__get_occJ()

        if not os.path.exists("occ_storage/"):
            os.mkdir("occ_storage/")

    # @property
    # def occRef1(self):
    #     """Returns:

    #     occRef1 -- represents n_a(1-n_b)."""

        

    @property
    def occA(self):
        """Returns:

        occA -- represents n_a - n_b."""
        return self._occA

    @property
    def occA4(self):
        """Built from flag = 1; rank 4 tensor

        Returns:

        occA4 -- represents n_a - n_b."""
        return self._occA4

    @property
    def occB(self):
        """Returns:

        occB -- represents 1 - n_a - n_b."""
        return self._occB

    @property
    def occB4(self):
        """Returns:

        occB4 -- represents 1 - n_a - n_b."""
        return self._occB4

    @property
    def occC(self):
        """Returns:

        occC -- represents n_a*n_b*(1-n_c) + (1-n_a)*(1-n_b)*n_c"""
        return self._occC

    @property
    def occD(self):
        """Returns:

        occD -- represents  n_a*n_b*(1-n_c)*(1-n_d)"""
        return self._occD

    @property
    def occE(self):
        """Returns:

        occE -- represents n_a*n_b*n_c*(1-n_d)*(1-n_e)*(1-n_f)"""
        return self._occE

    @property
    def occF(self):
        """Returns:

        occF -- represents n_a*n_b*(1-n_c)*(1-n_d)*(1-n_e) +
                (1-n_a)*(1-n_b)*n_c*n_d*n_e"""
        return self._occF

    @property
    def occG(self):
        """Returns:

        occG -- represents n_a*(1-n_b)*(1-n_c) + (1-n_a)*n_b*n_c"""
        return self._occG

    @property
    def occH(self):
        """Returns:

        occH -- represents n_a*(1-n_b)*(1-n_c)*(1-n_d) - (1-n_a)*n_b*n_c*n_d"""
        return self._occH

    @property
    def occI(self):
        """Returns:

        occI -- represents (1-n_a)*(1-n_b)*n_c*n_d - n_a*n_b*(1-n_c)*(1-n_d)"""
        return self._occI

    @property
    def occJ(self):
        """Returns:

        occJ -- represents n_a*n_b*n_c + (1-n_a)*(1-n_b)*(1-n_c)"""
        return self._occJ


    # ---- BUILD OCCUPATION TENSORS ---
    # def __get_occRef1(self):
    #     """Builds the occupation tensor occRef1, necessary for occupation number
    #     representation of the Hamiltonian.
        
    #     Returns:

    #     occRef1 -- n_a(1-n-b)
    #     """
        
        



    #@jit#(nopython=True)
    def __get_occA(self, flag=0):
        """Builds the occupation tensor occA.

        Keyword arguments:

        flag -- toggle rank 4 or rank 2 tensor behavior (default: 0)

        Returns:

        occA -- n_a - n_b"""

        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        if flag == 0: # default

            # occA = np.zeros((n,n,n,n),dtype=np.float32)

            # for a in bas1B:
            #     for b in bas1B:
            #         occA[a,b,a,b] = ref[a] - ref[b]

            # print(sys.getsizeof(occA)/10**6)


            # TENSOR TRAIN DECOMPOSITION
            # We find a TT-decomposition by hand, because the the tensor we want 
            # to decompose is small enough to do so. This is an exact decomposition
            # of the rank 2 tensor described by n_a-n_b.

            #Ga = tn.Node(np.array([ [1,1], [1,1], [1,1], [1,1], [0,1], [0,1], [0,1], [0,1] ]))
            #Gb = tn.Node(np.transpose(np.array([ [1,-1], [1,-1], [1,-1], [1,-1], [1,0], [1,0], [1,0], [1,0] ])))

            Ga = tn.Node(np.append(ref[:,np.newaxis], np.ones((n,1)),axis=1).astype(self.DATA_TYPE))
            Gb = tn.Node(np.transpose(np.append(np.ones((n,1)), -1*ref[:,np.newaxis],axis=1).astype(self.DATA_TYPE)))
            Gab = tn.ncon([Ga,Gb], [(-1,1),(1,-2)])
            #final = tn.outer_product(Gab, tn.Node(np.ones((n,n)))).tensor

            # PARALLELIZE NESTED LOOPS FOR BETTER PERFORMANCE
            # @numba.jit(nopython=True)
            # def enforce_delta(n, tensor):

            #     bas1B = range(n)
            
            #     for a in bas1B:
            #         for b in bas1B:
            #             for c in bas1B:
            #                 for d in bas1B:
            #                     if not(a == c and b == d):
            #                         tensor[a,b,c,d] = 0

            #     return tensor

            occA = Gab#tn.Node(enforce_delta(n, final))

#            print(sys.getsizeof(occA)/10**6)
            

        if flag == 1:
            # occA = np.zeros((n,n),dtype=np.float32)

            # for a in bas1B:
            #     for b in bas1B:
            #         occA[a,b] = ref[a] - ref[b]

            Ga = tn.Node(np.append(ref[:,np.newaxis], np.ones((n,1)),axis=1).astype(self.DATA_TYPE))
            Gb = tn.Node(np.transpose(np.append(np.ones((n,1)), -1*ref[:,np.newaxis],axis=1).astype(self.DATA_TYPE)))
            Gab = tn.ncon([Ga,Gb], [(-1,1),(1,-2)])
            
            final = tn.outer_product(Gab, tn.Node(np.ones((n,n))))

            occA = final

        return occA

    def __get_occB(self, flag=0):
        """Builds the occupation tensor occB.

        Keyword arguments:

        flag -- toggle rank 4 or rank 2 tensor behavior (default: 0)

        Returns:

        occB -- 1 - n_a - n_b"""

        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        if flag == 0: # default
            # occB = np.zeros((n,n,n,n),dtype=np.float32)

            # for a in bas1B:
            #     for b in bas1B:
            #         occB[a,b,a,b] = 1 - ref[a] - ref[b]

            #occB = tn.Node(1 - self.occA.tensor)
            
            # TENSOR TRAIN DECOMPOSITION of rank 4 tensor with elements given 
            # by 1 - n_a - n_b.

            #Ga = tn.Node(np.array([ [0,1], [0,1], [0,1], [0,1], [1,1], [1,1], [1,1], [1,1] ]))
            #Gb = tn.Node(np.transpose(np.array([ [1,-1], [1,-1], [1,-1], [1,-1], [1,0], [1,0], [1,0], [1,0] ])))

            Ga = tn.Node(np.append(1-ref[:,np.newaxis], np.ones((n,1)),axis=1).astype(self.DATA_TYPE))
            Gb = tn.Node(np.transpose(np.append(np.ones((n,1)), -1*ref[:,np.newaxis],axis=1).astype(self.DATA_TYPE)))
            Gab = tn.ncon([Ga,Gb], [(-1,1),(1,-2)])
            #final = tn.outer_product(Gab, tn.Node(np.ones((n,n)))).tensor

            # PARALLELIZE NESTED LOOPS FOR BETTER PERFORMANCE
            # @numba.jit(nopython=True)
            # def enforce_delta(n, tensor):

            #     bas1B = range(n)

            #     for a in bas1B:
            #         for b in bas1B:
            #             for c in bas1B:
            #                 for d in bas1B:
            #                     if not(a == c and b == d):
            #                         tensor[a,b,c,d] = 0

            #     return tensor

            #occB = tn.Node(enforce_delta(n, final))
            occB = Gab

        if flag == 1:
            # occB = np.zeros((n,n),dtype=np.float32)

            # for a in bas1B:
            #     for b in bas1B:
            #         occB[a,b] = 1 - ref[a] - ref[b]

            Ga = tn.Node(np.append(1-ref[:,np.newaxis], np.ones((n,1)),axis=1).astype(self.DATA_TYPE))
            Gb = tn.Node(np.transpose(np.append(np.ones((n,1)), -1*ref[:,np.newaxis],axis=1).astype(self.DATA_TYPE)))
            Gab = tn.ncon([Ga,Gb], [(-1,1),(1,-2)])
            final = tn.outer_product(Gab, tn.Node(np.ones((n,n))))

            occB = final

        return occB

    def __get_occC(self, flag=0):
        """Builds the occupation tensor occC.

        Keyword arguments:

        flag -- toggle rank 6 or rank 3 tensor behavior (default: 0)

        Returns:

        occC -- n_a*n_b*(1-n_c) + (1-n_a)*(1-n_b)*n_c"""

        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        if flag == 0: # default

            # occC = np.zeros((n,n,n,n,n,n),dtype=np.float32)

            # for a in bas1B:
            #     for b in bas1B:
            #         for c in bas1B:
            #             occC[a,b,c,a,b,c] = ref[a]*ref[b]*(1-ref[c]) + \
            #                                 (1-ref[a])*(1-ref[b])*ref[c]

            # print(sys.getsizeof(occC)/10**6)

            # TENSOR TRAIN DECOMPOSITION of rank 6 tensor with elements
            # given by n_a*n_b + (1 - n_b - n_a)*n_c.

            #Ga = tn.Node(np.array([ [1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1], [0,1,1,0], [0,1,1,0], [0,1,1,0], [0,1,1,0] ]))
            # Gb = tn.Node(np.array([ [[1,0,0,0], [0,1,0,0],[0,0,-1,0],[0,0,0,-1]], 
            #                         [[1,0,0,0], [0,1,0,0],[0,0,-1,0],[0,0,0,-1]], 
            #                         [[1,0,0,0], [0,1,0,0],[0,0,-1,0],[0,0,0,-1]],
            #                         [[1,0,0,0], [0,1,0,0],[0,0,-1,0],[0,0,0,-1]], 
            #                         [[0,0,0,0], [0,1,0,0],[0,0,0,0],[0,0,0,-1]],
            #                         [[0,0,0,0], [0,1,0,0],[0,0,0,0],[0,0,0,-1]],
            #                         [[0,0,0,0], [0,1,0,0],[0,0,0,0],[0,0,0,-1]],
            #                         [[0,0,0,0], [0,1,0,0],[0,0,0,0],[0,0,0,-1]] ]))
            #Gc = tn.Node(np.transpose(np.array([ [1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0] ])))

            Ga1 = np.append(ref[:,np.newaxis], np.ones((n,2)),axis=1).astype(self.DATA_TYPE)
            Ga2 = np.append(Ga1, ref[:,np.newaxis],axis=1).astype(self.DATA_TYPE)
            Ga = tn.Node(Ga2) 

            Gb1= np.append(ref[np.newaxis,np.newaxis,:],np.zeros((1,1,n)),axis=1).astype(self.DATA_TYPE)
            Gb2= np.append(Gb1, np.append(np.zeros((1,1,n)), np.ones((1,1,n)),axis=1), axis=0).astype(self.DATA_TYPE)
            Gb3 = np.array([[1,0],[0,-1]])
            Gb = tn.Node(np.kron(Gb3, np.transpose(Gb2)))
           
            Gc1 = np.append(np.ones((n,1)), np.repeat(ref[:,np.newaxis],3,axis=1), axis=1).astype(self.DATA_TYPE)
            Gc = tn.Node(np.transpose(Gc1))

            Gabc = tn.ncon([Ga, Gb, Gc], [(-1, 1), (-2, 1, 2), (2, -3)])

            # final = tn.outer_product(Gabc, tn.Node(np.ones((n,n,n)))).tensor

            # @numba.jit(nopython=True)#, parallel=True)
            # def enforce_delta(n, tensor):

            #     bas1B = range(n)

            #     for a in bas1B:
            #         for b in bas1B:
            #             for c in bas1B:
            #                 for d in bas1B:
            #                     for e in bas1B:
            #                         for f in bas1B:
            #                             if not(a == d and b == e and c == f):
            #                                 tensor[a,b,c,d,e,f] = 0
            #     return tensor

            
            occC = tn.outer_product(Gabc, tn.Node(np.ones(n))) #tn.Node(enforce_delta(n, final))
     
        if flag == 1:
            occC = np.zeros((n,n,n),dtype=np.float32)

            for a in bas1B:
                for b in bas1B:
                    for c in bas1B:
                        occC[a,b,c] = ref[a]*ref[b]*(1-ref[c]) + \
                                      (1-ref[a])*(1-ref[b])*ref[c]
        return occC

    def __get_occD(self, flag=0):
        """Builds the occupation tensor occD.

            Keyword arguments:

            flag -- toggle rank 8 or rank 4 tensor behavior (default: 0)

            Returns:

            occD -- n_a*n_b*(1-n_c)*(1-n_d)"""

        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)
        # occD = np.einsum('i,j,k,l->ijkl',ref,ref,(1-ref),(1-ref))
        
        # if flag == 0: # default
        #     occD = np.einsum('ijkl,mnop->ijklmnop', occD, occD)
        #     return occD
        # if flag == 1:
        #     return occD
        
        if flag == 0: # default

            # occD = np.zeros((n,n,n,n,n,n,n,n),dtype=np.float32)

            # for a in bas1B:
            #     for b in bas1B:
            #         for c in bas1B:
            #             for d in bas1B:
            #                 occD[a,b,c,d,a,b,c,d] = ref[a]*ref[b]*\
            #                                         (1-ref[c])*(1-ref[d])

            
            # Ga = tn.Node(np.array([ [1,0],[1,0],[1,0],[1,0],[0,0],[0,0],[0,0],[0,0] ]))
            # Gb = tn.Node(np.transpose(np.array([ [1,0],[1,0],[1,0],[1,0],[0,0],[0,0],[0,0],[0,0] ])))
            # Gc = tn.Node(np.array([ [0,0],[0,0],[0,0],[0,0],[1,0],[1,0],[1,0],[1,0] ]))
            # Gd = tn.Node(np.transpose(np.array([ [0,0],[0,0],[0,0],[0,0],[1,0],[1,0],[1,0],[1,0] ])))

            Ga = tn.Node(np.transpose(np.append(ref[np.newaxis,:], np.zeros((1,n)), axis=0).astype(self.DATA_TYPE)))
            Gb = tn.Node(np.transpose(Ga.tensor))
            Gc = tn.Node(np.transpose(np.append(ref[::-1][np.newaxis,:],np.zeros((1,n)), axis=0).astype(self.DATA_TYPE)))
            Gd = tn.Node(np.transpose(Gc.tensor))

            Gabcd = tn.ncon([Ga,Gb,Gc,Gd], [(-1,1),(1,-2),(-3,2),(2,-4)])

            final = tn.outer_product(Gabcd, tn.Node(np.ones((8,8,8,8)))).tensor

            @numba.jit(nopython=True)#, parallel=True)
            def enforce_delta(n, tensor):

                bas1B = range(n)

                for a in bas1B:
                    for b in bas1B:
                        for c in bas1B:
                            for d in bas1B:
                                for e in bas1B:
                                    for f in bas1B:
                                        for g in bas1B:
                                            for h in bas1B:
                                                if not(a == e and b == f and c == g and d == h):
                                                    tensor[a,b,c,d,e,f,g,h] = 0
                return tensor

            occD = tn.Node(enforce_delta(n, final))


        if flag == 1:
            # occD = np.zeros((n,n,n,n),dtype=np.float32)

            # for a in bas1B:
            #     for b in bas1B:
            #         for c in bas1B:
            #             for d in bas1B:
            #                 occD[a,b,c,d] = ref[a]*ref[b]*\
            #                                 (1-ref[c])*(1-ref[d])

            # Ga = tn.Node(np.array([ [1,0],[1,0],[1,0],[1,0],[0,0],[0,0],[0,0],[0,0] ]))
            # Gb = tn.Node(np.transpose(np.array([ [1,0],[1,0],[1,0],[1,0],[0,0],[0,0],[0,0],[0,0] ])))
            # Gc = tn.Node(np.array([ [0,0],[0,0],[0,0],[0,0],[1,0],[1,0],[1,0],[1,0] ]))
            # Gd = tn.Node(np.transpose(np.array([ [0,0],[0,0],[0,0],[0,0],[1,0],[1,0],[1,0],[1,0] ])))

            Ga = tn.Node(np.transpose(np.append(ref[np.newaxis,:], np.zeros((1,n)), axis=0).astype(self.DATA_TYPE)))
            Gb = tn.Node(np.transpose(Ga.tensor))
            Gc = tn.Node(np.transpose(np.append(ref[::-1][np.newaxis,:],np.zeros((1,n)), axis=0).astype(self.DATA_TYPE)))
            Gd = tn.Node(np.transpose(Gc.tensor))

            Gabcd = tn.ncon([Ga,Gb,Gc,Gd], [(-1,1),(1,-2),(-3,2),(2,-4)])

            occD = Gabcd

        return occD
                            
# ---- ALL ABOVE REQUIRED FOR IMSRG(2) ---

    def __get_occE(self):
        """Builds the occupation tensor occE. Treat as a rank 6 tensor.

            Returns:

            occE -- n_a*n_b*n_c*(1-n_d)*(1-n_e)*(1-n_f)"""
        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        occE = np.einsum('i,j,k,l,m,n->ijklmn',ref,ref,ref,(1-ref),(1-ref),(1-ref))
        
        # occE = np.zeros((n,n,n,n,n,n),dtype=np.float32)

        # for a in bas1B:
        #     for b in bas1B:
        #         for c in bas1B:
        #             for d in bas1B:
        #                 for e in bas1B:
        #                     for f in bas1B:
        #                         occE[a,b,c,d,e,f] = ref[a]*ref[b]*ref[c]*\
        #                                             (1-ref[d])*(1-ref[e])*\
        #                                             (1-ref[f])

        return occE
    
    def __get_occF(self):
        """Builds the occupation tensor __get_occF. Treat as a rank 10 tensor.

            Returns:

            occF -- n_a*n_b*(1-n_c)*(1-n_d)*(1-n_e) +
                    (1-n_a)*(1-n_b)*n_c*n_d*n_e"""
        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        # occF1 = np.einsum('i,j,k,l,m->ijklm',ref,ref,(1-ref),(1-ref),(1-ref))
        # occF2 = np.einsum('i,j,k,l,m->ijklm',(1-ref),(1-ref),ref,ref,ref)
        # occF3 = occF1 + occF2
        # occF = np.einsum('ijklm,nopqr->ijklmnopqr',occF3,occF3)
        
        occF = np.zeros((n,n,n,n,n,n,n,n,n,n),dtype=np.float32)

        for a in bas1B:
            for b in bas1B:
                for c in bas1B:
                    for d in bas1B:
                        for e in bas1B:
                            occF[a,b,c,d,e,a,b,c,d,e] = ref[a]*ref[b]*\
                                           (1-ref[c])*(1-ref[d])*(1-ref[e]) + \
                                           (1-ref[a])*(1-ref[b])*\
                                           ref[c]*ref[d]*ref[e]

        return occF

    def __get_occG(self):
        """Builds the occupation tensor occG. Treat as a rank 6 tensor.

            Returns:

            occG -- n_a*(1-n_b)*(1-n_c) + (1-n_a)*n_b*n_c"""
        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        occG = np.zeros((n,n,n,n,n,n),dtype=np.float32)

        for a in bas1B:
            for b in bas1B:
                for c in bas1B:
                    occG[a,b,c,a,b,c] = ref[a]*(1-ref[b])*(1-ref[c]) + \
                                        (1-ref[a])*ref[b]*ref[c]

        return occG

    def __get_occH(self):
        """Builds the occupation tensor occH. Treat as a rank 8 tensor.

            Returns:

            occH -- n_a*(1-n_b)*(1-n_c)*(1-n_d) - (1-n_a)*n_b*n_c*n_d"""
        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        occH = np.zeros((n,n,n,n,n,n,n,n),dtype=np.float32)

        for a in bas1B:
            for b in bas1B:
                for c in bas1B:
                    for d in bas1B:
                        occH[a,b,c,d,a,b,c,d] = ref[a]*(1-ref[b])*(1-ref[c])*\
                                    (1-ref[d]) - (1-ref[a])*ref[b]*ref[c]*ref[d]

        return occH

    def __get_occI(self):
        """Builds the occupation tensor occI. Treat as a rank 8 tensor.

            Returns:

            occI -- (1-n_a)*(1-n_b)*n_c*n_d - n_a*n_b*(1-n_c)*(1-n_d)"""

        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        occI = np.zeros((n,n,n,n,n,n,n,n),dtype=np.float32)

        for a in bas1B:
            for b in bas1B:
                for c in bas1B:
                    for d in bas1B:
                        occI[a,b,c,a,b,c] = (1-ref[a])*(1-ref[b])*ref[c]*ref[d]-\
                                           ref[a]*ref[b]*(1-ref[c])*(1-ref[d])

        return occI

    def __get_occJ(self):
        """Builds the occupation tensor occJ. Treat as a rank 6 tensor.

            Returns:

            occJ -- n_a*n_b*n_c + (1-n_a)*(1-n_b)*(1-n_c)"""
        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        occJ = np.zeros((n,n,n,n,n,n),dtype=np.float32)

        for a in bas1B:
            for b in bas1B:
                for c in bas1B:
                    occJ[a,b,c,a,b,c] = ref[a]*ref[b]*ref[c] + \
                                        (1-ref[a])*(1-ref[b])*(1-ref[c])

        return occJ

# ---- ALL ABOVE REQUIRED FOR IMSRG(3) ---
