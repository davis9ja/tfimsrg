import numpy as np

class OccupationTensors(object):
    """Functions as a container for important occupation tensors
    defined by the reference state (usually, ground state) of
    the Hamiltonian. The occupation basis is a vector n that
    contains a 1 if the corresponding single particle basis
    index is occupied, or 0 if the same index is unoccupied,
    in the reference state."""

    def __init__(self, sp_basis, reference):
        """Class constructor. Instantiates an OccupationTensors object.

        Arugments:

        sp_basis -- list containing indices of single particle basis
        reference -- contains reference state as string of 1/0's"""

        self._reference = reference
        self._sp_basis = sp_basis

        self._occA = self.__get_occA()
        self._occA2 = self.__get_occA(flag=1)
        self._occB = self.__get_occB()
        self._occC = self.__get_occC()
        self._occD = self.__get_occD(flag=1)
        self._occE = self.__get_occE()
        self._occF = self.__get_occF()
        self._occG = self.__get_occG()
        self._occH = self.__get_occH()
        self._occI = self.__get_occI()
        self._occJ = self.__get_occJ()

    @property
    def occA(self):
        """Returns:

        occA -- represents n_a - n_b."""
        return self._occA

    @property
    def occA2(self):
        """Built from flag = 1; rank2 tensor

        Returns:

        occA2 -- represents n_a - n_b."""
        return self._occA2

    @property
    def occB(self):
        """Returns:

        occB -- represents 1 - n_a - n_b."""
        return self._occB

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
            occA = np.zeros((n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    occA[a,b,a,b] = ref[a] - ref[b]

        if flag == 1:
            occA = np.zeros((n,n))

            for a in bas1B:
                for b in bas1B:
                    occA[a,b] = ref[a] - ref[b]

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
            occB = np.zeros((n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    occB[a,b,a,b] = 1 - ref[a] - ref[b]

        if flag == 1:
            occB = np.zeros((n,n))

            for a in bas1B:
                for b in bas1B:
                    occB[a,b] = 1 - ref[a] - ref[b]

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
            occC = np.zeros((n,n,n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    for c in bas1B:
                        occC[a,b,c,a,b,c] = ref[a]*ref[b]*(1-ref[c]) + \
                                            (1-ref[a])*(1-ref[b])*ref[c]


        if flag == 1:
            occC = np.zeros((n,n,n))

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

        if flag == 0: # default
            occD = np.zeros((n,n,n,n,n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    for c in bas1B:
                        for d in bas1B:
                            occD[a,b,c,d,a,b,c,d] = ref[a]*ref[b]*\
                                                    (1-ref[c])*(1-ref[d])

        if flag == 1:
            occD = np.zeros((n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    for c in bas1B:
                        for d in bas1B:
                            occD[a,b,c,d] = ref[a]*ref[b]*\
                                            (1-ref[c])*(1-ref[d])

        return occD

# ---- ALL ABOVE REQUIRED FOR IMSRG(2) ---

    def __get_occE(self):
        """Builds the occupation tensor occE. Treat as a rank 6 tensor.

            Returns:

            occE -- n_a*n_b*n_c*(1-n_d)*(1-n_e)*(1-n_f)"""
        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        occE = np.zeros((n,n,n,n,n,n))

        for a in bas1B:
            for b in bas1B:
                for c in bas1B:
                    for d in bas1B:
                        for e in bas1B:
                            for f in bas1B:
                                occE[a,b,c,d,e,f] = ref[a]*ref[b]*ref[c]*\
                                                    (1-ref[d])*(1-ref[e])*\
                                                    (1-ref[f])

        return occE
    def __get_occF(self):
        """Builds the occupation tensor __get_occF. Treat as a rank 10 tensor.

            Returns:

            occF -- n_a*n_b*(1-n_c)*(1-n_d)*(1-n_e) +
                    (1-n_a)*(1-n_b)*n_c*n_d*n_e"""
        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        occF = np.zeros((n,n,n,n,n,n,n,n,n,n))

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

        occG = np.zeros((n,n,n,n,n,n))

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

        occH = np.zeros((n,n,n,n,n,n,n,n))

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

        occI = np.zeros((n,n,n,n,n,n,n,n))

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

        occJ = np.zeros((n,n,n,n,n,n))

        for a in bas1B:
            for b in bas1B:
                for c in bas1B:
                    occJ[a,b,c,a,b,c] = ref[a]*ref[b]*ref[c] + \
                                        (1-ref[a])*(1-ref[b])*(1-ref[c])

        return occJ

# ---- ALL ABOVE REQUIRED FOR IMSRG(3) ---
