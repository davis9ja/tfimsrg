# Main program for IM-SRG(2).


# Author: Jacob Davison
# Date:   07/10/2019

# import packages, libraries, and modules
# libraries
from scipy.integrate import odeint, ode
import numpy as np
import time
import pickle
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import tracemalloc
import os, sys
#from memory_profiler import profile
import itertools
import random
import tensornetwork as tn
#tn.set_default_backend("tensorflow")

# USER MODULES
from oop_imsrg.spin_sq import *
from oop_imsrg.hamiltonian import *
from oop_imsrg.occupation_tensors import *
from oop_imsrg.generator import *
from oop_imsrg.flow import *
from oop_imsrg.plot_data import *
# from oop_imsrg.display_memory import *
import oop_imsrg.ci_pairing.cipy_pairing_plus_ph as ci_matrix
from oop_imsrg.tests2B import *

sys.path.append('/mnt/home/daviso53/Research/')
from pyci.density_matrix.density_matrix import density_1b, density_2b
import pyci.imsrg_ci.pyci_p3h as pyci

def get_vacuum_coeffs(E, f, G, basis, holes):

    H2B = G
    H1B = f - np.trace(G[np.ix_(basis,holes,basis,holes)], axis1=1,axis2=3) 

    Gnode = tn.Node(G[np.ix_(holes,holes,holes,holes)])
    Gnode[0] ^ Gnode[2]
    Gnode[1] ^ Gnode[3]
    result_ij = Gnode @ Gnode


    H0B = E - np.trace(H1B[np.ix_(holes,holes)]) - 0.5*result_ij.tensor

    #print(result_ij.tensor, result_ji.tensor)
    return (H0B, H1B, H2B)


# @profile
def derivative(t, y, inputs):
    """Defines the derivative to pass into ode object.

    Arguments:
    (required by scipy.integrate.ode)
    t -- points at which to solve for y
    y -- in this case, 1D array that contains E, f, G

    (additional parameters)
    hamiltonian -- Hamiltonian object
    occ_tensors -- OccupationTensors object
    generator   -- Generator object
    flow        -- Flow object

    Returns:

    dy -- next step in flow"""

    hamiltonian, occ_tensors, generator, flow, tspinsq, generator_spin, flow_spin = inputs
    #assert isinstance(hamiltonian, Hamiltonian), "Arg 2 must be Hamiltonian object"
    assert isinstance(occ_tensors, OccupationTensors), "Arg 3 must be OccupationTensors object"
    assert isinstance(generator, Generator), "Arg 4 must be Generator object"
    assert isinstance(flow, Flow), "Arg 5 must be a Flow object"

    half = int(len(y)/2)

    # Hamiltonian flow
    E, f, G = ravel(y[0:half], hamiltonian.n_sp_states)
    generator.f = f
    generator.G = G
    dE, df, dG = flow.flow(generator)

    # Spin-squared flow
    E_spin, f_spin, G_spin = ravel(y[half::], tspinsq.n_sp_states)
    generator_spin.f = f_spin
    generator_spin.G = G_spin
    dE_spin, df_spin, dG_spin = flow_spin.flow(generator)

    dy = np.concatenate([unravel(dE, df, dG), unravel(dE_spin, df_spin, dG_spin)], axis=0)

    return dy

# @profile
def unravel(E, f, G):
    """Transforms E, f, and G into a 1D array. Facilitates
    compatability with scipy.integrate.ode.

    Arguments:

    E, f, G -- normal-ordered pieces of Hamiltonian

    Returns:

    concatenation of tensors peeled into 1D arrays"""
    unravel_E = np.reshape(E, -1)
    unravel_f = np.reshape(f, -1)
    unravel_G = np.reshape(G, -1)

    return np.concatenate([unravel_E, unravel_f, unravel_G], axis=0)

# @profile
def ravel(y, bas_len):
    """Transforms 1D array into E, f, and G. Facilitates
    compatability with scipy.integrate.ode.

    Arguments:

    y       -- 1D data array (output from unravel)
    bas_len -- length of single particle basis

    Returns:

    E, f, G -- normal-ordered pieces of Hamiltonian
    """

    # bas_len = len(np.append(holes,particles))

    ravel_E = np.reshape(y[0], ())
    ravel_f = np.reshape(y[1:bas_len**2+1], (bas_len, bas_len))
    ravel_G = np.reshape(y[bas_len**2+1:bas_len**2+1+bas_len**4],
                         (bas_len, bas_len, bas_len, bas_len))
    

    return(ravel_E, ravel_f, ravel_G)

# @profile
def main(n_holes, n_particles, ref=[], d=1.0, g=0.5, pb=0.0, verbose=1, flow_data_log=0, generator='wegner', output_root='.'):
    """Main method uses scipy.integrate.ode to solve the IMSRG(2) flow
    equations.

    Arguments:
    
    n_holes -- number of hole states in the SP basis (int)
    n_particles -- number of particle states in the SP basis (int)
    
    Keyword arguments:
    
    ref           -- reference state for the IM-SRG flow (1D array)
    d             -- energy spacing in Pairing model (default: 1.0)
    g             -- pairing strength in Pairing model (default: 0.5)
    pb            -- pair-breaking in Pairing-plus-ph model (default: 0.0)
    verbose       -- toggles output of flow information (default: 1)
    flow_data_log -- toggles output of flow data (pickled IM-SRG coefficients every 10 integrator steps) (default: 0)
    generator     -- specify generator to produce IM-SRG flow (default: wegner)
    output_root   -- specify folder for output files

    Returns:

    convergence -- 0 if diverged, 1 if converged (little bit outdated)
    iters       -- number of iterations before integrator stopped
    d           -- energy spacing in pairing model
    g           -- pairing strength in pairing model
    pb          -- pair-breaking strength in Pairing-plus-ph model
    num_sp      -- number of single particle states
    s_vals      -- 1D array of flow parameter values
    E_vals      -- 1D array of zero-body energy values
    time_str    -- time taken for flow completion (string)
    """
    
    start = time.time() # start full timer

    initi = time.time() # start instantiation timer

    if not os.path.exists(output_root):
        os.mkdir(output_root)

    if ref == []:
        ha = PairingHamiltonian2B(n_holes, n_particles, d=d, g=g, pb=pb)
        ref = ha.reference # this is just for printing
        ss = TSpinSq(n_holes, n_particles)
    else:
        ha = PairingHamiltonian2B(n_holes, n_particles, ref=ref, d=d, g=g, pb=pb)
        ss = TSpinSq(n_holes, n_particles)

    ot = OccupationTensors(ha.sp_basis, ha.reference)

    generator_dict = {'wegner':WegnerGenerator(ha, ot), 
                      'white':WhiteGenerator(ha),
                      'white_mp':WhiteGeneratorMP(ha),
                      'brillouin':BrillouinGenerator(ha),
                      'imtime':ImTimeGenerator(ha)}

    wg = generator_dict[generator] #WegnerGenerator(ha, ot)
    fl = Flow_IMSRG2(ha, ot) 

    wg_spin = WhiteGenerator(ss)
    fl_spin = Flow_IMSRG2(ss, ot)

    initf = time.time() # finish instantiation timer

    if verbose:
        print("Initialized objects in {:2.4f} seconds\n".format(initf-initi))

    if verbose:
        print("""Pairing model IM-SRG(2) flow:
        Generator      = {}
        d              = {:2.4f}
        g              = {:2.4f}
        pb             = {:2.4f}
        SP basis size  = {:2d}
        n_holes        = {:2d}
        n_particles    = {:2d}
        ref            = {d}""".format(generator, ha.d, ha.g, ha.pb, ha.n_sp_states,
                                       len(ha.holes), len(ha.particles),
                                       d=ref) )
    if verbose:
        print("Flowing...")
    flowi = time.time()

    # --- Solve the IM-SRG flow
    y0 = np.concatenate([unravel(ha.E, ha.f, ha.G), unravel(ss.E, ss.f, ss.G)], axis=0)

    H0B, H1B, H2B = get_vacuum_coeffs(ha.E, ha.f, ha.G, ha.sp_basis, ha.holes)
    zero, eta1B_vac, eta2B_vac = get_vacuum_coeffs(0.0, wg.eta1B, wg.eta2B, ha.sp_basis, ha.holes)

    pickle.dump( (H0B, H1B, H2B, eta1B_vac, eta2B_vac), open( output_root+"/vac_coeffs_unevolved.p", "wb" ) )

    solver = ode(derivative,jac=None)
    solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
    solver.set_f_params([ha, ot, wg, fl, ss, wg_spin, fl_spin])
    solver.set_initial_value(y0, 0.)

    sfinal = 50
#    ds = 1
    s_vals = []
    E_vals = []
    
    E_gs_lst = []
    s_expect_lst = []

    iters = 0
    convergence = 0

    if verbose:
        print("iter,      \t    s, \t         E, \t         ||eta1B||, \t         ||eta2B||")
        
    while solver.successful() and solver.t < sfinal:

        #print('solver success,', solver.successful())

        ys = solver.integrate(sfinal, step=True)

        half = int(len(ys)/2)
        Es, fs, Gs = ravel(ys[0:half], ha.n_sp_states)
        E_spins, f_spins, G_spins = ravel(ys[half::], ss.n_sp_states)

        s_vals.append(solver.t)
        E_vals.append(Es)
        
        # compute density matrices, get expectation values
        H0B, H1B, H2B = get_vacuum_coeffs(Es, fs, Gs, ha.sp_basis, ha.holes)
        SS0B, SS1B, SS2B = get_vacuum_coeffs(E_spins, f_spins, G_spins, ss.sp_basis, ss.holes)
        hme = pyci.matrix(n_holes, n_particles, H0B, H1B, H2B, H2B, imsrg=True)
        w,v = np.linalg.eigh(hme)
        v0 = v[:,0]

        rho1b = density_1b(n_holes,n_particles, weights=v0)
        rho2b = density_2b(n_holes,n_particles, weights=v0)

        contract_1b = np.einsum('ij,ij', rho1b, SS1B)

        rho_reshape_2b = np.reshape(rho2b, (ss.n_sp_states**2,ss.n_sp_states**2))
        h2b_reshape_2b = np.reshape(SS2B, (ss.n_sp_states**2,ss.n_sp_states**2))
        contract_2b = np.einsum('ij,ij', rho_reshape_2b, h2b_reshape_2b)

        s_expect = SS0B + contract_1b + 0.25*contract_2b


        E_gs_lst.append(w[0])
        s_expect_lst.append(s_expect)
        

        if iters %10 == 0 and verbose: 
            norm_eta1B = np.linalg.norm(np.ravel(wg.eta1B))
            norm_eta2B = np.linalg.norm(np.ravel(wg.eta2B))
            print("{:>6d}, \t {:0.4f}, \t {:0.8f}, \t {:0.8f}, \t {:0.8f}".format(iters, 
                                                                                  solver.t, 
                                                                                  Es,
                                                                                  norm_eta1B,
                                                                                  norm_eta2B))
        
        if flow_data_log and iters %10 == 0:
            H0B, H1B, H2B = get_vacuum_coeffs(Es, fs, Gs, ha.sp_basis, ha.holes)
            zero, eta1B_vac, eta2B_vac = get_vacuum_coeffs(0.0, wg.eta1B, wg.eta2B, ha.sp_basis, ha.holes)
            fname = output_root+'/vac_coeffs_flow_c{}.p'.format(iters)
            pickle.dump((solver.t, H0B, H1B, H2B, eta1B_vac, eta2B_vac), open(fname, 'wb'))

#        if iters %20 == 0 and verbose:
#            coeffs = get_vacuum_coeffs(Es, fs, Gs, ha.sp_basis, ha.holes)
#            pickle.dump( coeffs, open( "mixed_state_test/pickled_coeffs/vac_coeffs_s{}.p".format(iters), "wb" ) )

        if len(E_vals) > 100 and abs(E_vals[-1] - E_vals[-2]) < 10**-8 and E_vals[-1] != E_vals[0]:

            if verbose: print("---- Energy converged at iter {:>06d} with energy {:1.8f}\n".format(iters,E_vals[-1]))
            convergence = 1
            break

        # if len(E_vals) > 100 and abs(E_vals[-1] - E_vals[-2]) > 1:
        #     if verbose: print("---- Energy diverged at iter {:>06d} with energy {:3.8f}\n".format(iters,E_vals[-1]))
        #     break

        if iters > 10000:
            if verbose: print("---- Energy diverged at iter {:>06d} with energy {:3.8f}\n".format(iters,E_vals[-1]))
            break

        # if iters % 1000 == 0 and verbose:
        #     print('Iteration {:>06d}'.format(iters))

        iters += 1
        #print(solver.successful())
    flowf = time.time()
    end = time.time()
    time_str = "{:2.5f}".format(end-start)

    if verbose: print("IM-SRG(2) converged in {:2.5f} seconds".format(flowf-flowi))

    H0B, H1B, H2B = get_vacuum_coeffs(Es, fs, Gs, ha.sp_basis, ha.holes)
    zero, eta1B_vac, eta2B_vac = get_vacuum_coeffs(0.0, wg.eta1B, wg.eta2B, ha.sp_basis, ha.holes)
    #pickle.dump( coeffs, open( "mixed_state_test/pickled_coeffs/vac_coeffs_evolved.p", "wb" ) )
    pickle.dump((H0B, H1B, H2B, eta1B_vac, eta2B_vac), open(output_root+'/vac_coeffs_evolved.p', 'wb'))

    num_sp = n_holes+n_particles

    del ha, ot, wg, fl, solver, y0, sfinal

    expect_flow_df = pd.DataFrame({'s':s_vals, 'E_gs':E_gs_lst, 's_expect':s_expect_lst})
    pickle.dump(expect_flow_df, open('expect_flow.p', 'wb'))

    return (convergence, iters, d, g, pb, num_sp, s_vals, E_vals, time_str)
 

if __name__ == '__main__':


    #test_exact('plots_exact_2b/', main)
    # refs = list(map("".join, itertools.permutations('11110000')))
    # refs = list(dict.fromkeys(refs)) # remove duplicates
    # refs = [list(map(int, list(ref))) for ref in refs]


    # TESTING MIXED STATE --------------------------------------------
    # refs = [[1,1,1,1,0,0,0,0],[1,1,0,0,1,1,0,0],[1,1,0,0,0,0,1,1],
    #         [0,0,1,1,1,1,0,0],[0,0,1,1,0,0,1,1],[0,0,0,0,1,1,1,1]]
    

    # gsws = [0.99, 0.985,0.975,0.95,0.9,0.85,0.8]#,0.75,0.7,0.65,0.6,0.55,0.5]
    
    # E_data_full = []
    # for gsw in gsws:
    #     E_data = []
    #     count = 0
    #     rsw = (1-gsw)
    #     for i in range(len(refs)):
    #         ref = np.zeros_like(refs[0])
    #         E = 0.0
    #         if i == 0: 
    #             ref = refs[0]
    #             data = main(4,4,ref=ref,verbose=0)
    #             E = (data[7])[-1]
    #             count += 1
    #         else:
    #             esum = np.zeros_like(refs[0])
    #             for state in refs[1:count+1]:
    #                 esum = np.add(esum,state)
    #             print(esum)
    #             ref = gsw*np.array(refs[0]) + rsw/count*(esum)
    #             data = main(4,4,ref=ref,verbose=0)
    #             E = (data[7])[-1]
    #             count += 1
    #         print("{:0.8f}, {}, {f}".format(E, sum(ref), f=ref))
    #         E_data.append(E)
    #     E_data_full.append(E_data)

    # exact = ci_matrix.exact_diagonalization(1.0, 0.5, 0.0)

    # pickle.dump( E_data_full, open( "save.p", "wb" ) )

    # ----------------------------------------------------------------

    # main(4,4, generator='white')
    # data = pickle.load(open('expect_flow.p', 'rb'))

    # fig = plt.figure(figsize=(8,4))
    # sns.lineplot(x='s', y=data['E_gs']/data['E_gs'][0], data=data)
    # sns.lineplot(x='s', y=data['s_expect']/data['s_expect'][0], data=data)
    # plt.legend(['E(s)/E(s=0)', 'SS(s)/SS(s=0)'])
    # plt.savefig('flow_conservation.png')


    ref = 0.7*np.array([1,1,1,1,0,0,0,0])+0.3*np.array([1,1,0,0,1,1,0,0])
    main(4,4, g=2.0, ref=ref, generator='white')
    data = pickle.load(open('expect_flow.p', 'rb'))

    fig = plt.figure(figsize=(8,4))
    sns.lineplot(x='s', y=data['E_gs']/data['E_gs'][0], data=data)
    sns.lineplot(x='s', y=data['s_expect']/data['s_expect'][0], data=data)
    plt.legend(['E(s)/E(s=0)', 'SS(s)/SS(s=0)'])
    plt.savefig('flow_conservation_ensemble_best_g2.png')

    hme = pyci.matrix(4,4,0.0,1.0,2.0,0.0)
    w,v = np.linalg.eigh(hme)
    fig = plt.figure(figsize=(8,4))
    sns.lineplot(x='s', y=data['E_gs'], data=data)
    sns.lineplot(x='s', y=w[0], data=data)
    plt.legend(['E_gs evolution', 'FCI gs'])
    plt.savefig('E_gs_error_g2.png')

    # refs = [[1,1,1,1,0,0,0,0],[1,1,0,0,1,1,0,0],[1,1,0,0,0,0,1,1],
    #         [0,0,1,1,1,1,0,0],[0,0,1,1,0,0,1,1]]

    # ref = 0.5*np.asarray(refs[0]) + (1.0-0.5)/4.0*(np.asarray(refs[1]) + np.asarray(refs[2]) + np.asarray(refs[3]) + np.asarray(refs[4]))
    # # main(4,4, g=5, ref=[1,1,1,1,0,0,0,0])

    
    # main(4,4,g=1.0, pb=0.1, flow_data_log=0, generator='white')

    # H1B_true, H2B_true = pickle.load(open('comparison.p','rb'))
    # H1B, H2B = pickle.load(open('vac_coeffs_unevolved.p', 'rb'))
    # print(H1B, H1B_true)

    
    #print(np.array_equal(H2B_true, H2B))
    #main(4,4)
    

    # ref1 = 0.9*refs[0] + 0.1*refs[1]
    # data = main(4,4,ref=ref1)
    # E1 = data[7]

    # ref2 = 0.9*refs[0] + 0.05*(refs[1] + refs[2])
    # data = main(4,4,ref=ref2)
        
#    ref = 0.9*np.array([1,1,1,1,0,0,0,0])+0.1*np.array([0,1,1,1,1,0,0,0])
    
    #main(4,4,ref=ref)

    # exact = ci_matrix.exact_diagonalization(1.0, 0.5, 0.0)

    # fig = plt.figure()

    # for i in range(0,10):
    #     data = main(4,4)
    #     s_vals = data[6]
    #     E_vals = data[7]/exact
    #     plt.plot(s_vals, E_vals)
    # plt.ylim([1+10**-8, 1+10**-6])
    # plt.xlabel('scale param (s)')
    # plt.ylabel('energy')
    # plt.show()


    #-- TESTING TIMINGS -------------------------
    #holes = int(sys.argv[1])

    # print('convergence,iters,d,g,pb,num states,GS energy,total time')
    # for num_pairs in range(1,21):
    #     #print(n)55
    #     #n = int(num_sp/2)
    #     #holes = num_pairs*2
    #     particles = num_pairs*2

    #     convergence,iters,d,g,pb,sp_states,s_vals,E_vals,time_str = main(holes, particles, verbose=0)
    #     print('{},{},{},{},{},{},{},{}'.format(convergence,iters,d,g,pb,sp_states,E_vals[-1],time_str))
    
    #     del convergence, iters, d, g, pb, sp_states, s_vals, E_vals, time_str
