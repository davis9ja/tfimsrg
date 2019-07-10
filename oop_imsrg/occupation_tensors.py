import numpy as np

class OccupationTensors(object):

    def __init__(self, sp_basis, reference):
        self._reference = reference
        self._sp_basis = sp_basis

        self._occA = self.__get_occA()
        self._occB = self.__get_occB()
        self._occC = self.__get_occC()
        self._occD = self.__get_occD(flag=1)

    @property
    def occA(self):
        return self._occA
    
    @property
    def occB(self):
        return self._occB
        
    @property
    def occC(self):
        return self._occC
    
    @property
    def occD(self):
        return self._occD
    
    def __get_occA(self, flag=0):
        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        if flag == 0: # default
            occA = np.zeros((n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    occA[a,b,a,b] = ref[a] - ref[b]

            return occA
        
        if flag == 1:
            occA = np.zeros((n,n))

            for a in bas1B:
                for b in bas1B:
                    occA[a,b] = ref[a] - ref[b]

            return occA    
        
    # covers (1-na-nb)
    def __get_occB(self, flag=0):
        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        if flag == 0: # default
            occB = np.zeros((n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    occB[a,b,a,b] = 1 - ref[a] - ref[b]

            return occB
        
        if flag == 1:
            occB = np.zeros((n,n))

            for a in bas1B:
                for b in bas1B:
                    occB[a,b] = 1 - ref[a] - ref[b]

            return occB        
            
    # covers na*nb + (1-na-nb)*nc
    def __get_occC(self, flag=0):
        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        if flag == 0: # default
            occC = np.zeros((n,n,n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    for c in bas1B:
                        occC[a,b,c,a,b,c] = ref[a]*ref[b] + (1-ref[a]-ref[b])*ref[c]

            return occC
        
        if flag == 1: 
            occC = np.zeros((n,n,n))

            for a in bas1B:
                for b in bas1B:
                    for c in bas1B:
                        occC[a,b,c] = ref[a]*ref[b] + (1-ref[a]-ref[b])*ref[c]


    # covers na*nb*(1-nc-nd) + na*nb*nc*nd
    def __get_occD(self, flag=0):
        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        if flag == 0: # default 
            occD = np.zeros((n,n,n,n,n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    for c in bas1B:
                        for d in bas1B:
                            occD[a,b,c,d,a,b,c,d] = ref[a]*ref[b]*(1-ref[c]-ref[d])+ref[a]*ref[b]*ref[c]*ref[d]

            return occD
        
        if flag == 1:
            occD = np.zeros((n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    for c in bas1B:
                        for d in bas1B:
                            occD[a,b,c,d] = ref[a]*ref[b]*(1-ref[c]-ref[d])+ref[a]*ref[b]*ref[c]*ref[d]

            return occD    