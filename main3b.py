# Main program for IM-SRG(3).


# Author: Jacob Davison
# Date:   07/10/2019

# import packages, libraries, and modules
# libraries
from scipy.integrate import odeint, ode
import numpy as np
import time
import pickle
import tracemalloc
import os, sys
#from memory_profiler import profile
import itertools
import random
import tensornetwork as tn

# user files
# sys.path.append('C:\\Users\\davison\\Research\\exact_diagonalization\\')
from tfimsrg.oop_imsrg.hamiltonian import *
from tfimsrg.oop_imsrg.occupation_tensors import *
from tfimsrg.oop_imsrg.generator import *
from tfimsrg.oop_imsrg.flow import *
from tfimsrg.oop_imsrg.plot_data import *
# from oop_imsrg.display_memory import *
import tfimsrg.oop_imsrg.ci_pairing.cipy_pairing_plus_ph as ci_matrix

def derivative(t, y, hamiltonian, occ_tensors, generator, flow):
    """Defines the derivative to pass into ode object.

    Arguments:
    (required by scipy.integrate.ode)
    t -- points at which to solve for y
    y -- in this case, 1D array that contains E, f, G

    (additional parameters)
    hamiltonian -- Hamiltonian object
    occ_tensors -- OccupationTensors object
    generator -- Generator object
    flow -- Flow object

    Returns:

    dy -- next step in flow"""

    assert isinstance(hamiltonian, Hamiltonian), "Arg 2 must be Hamiltonian object"
    assert isinstance(occ_tensors, OccupationTensors), "Arg 3 must be OccupationTensors object"
    assert isinstance(generator, Generator), "Arg 4 must be Generator object"
    assert isinstance(flow, Flow), "Arg 5 must be a Flow object"

    E, f, G, W = ravel(y, hamiltonian.n_sp_states)

    generator.f = f
    generator.G = G
    generator.W = W

    dE, df, dG, dW = flow.flow(generator)

    dy = unravel(dE, df, dG, dW)

    return dy

# @profile
def unravel(E, f, G, W):
    """Transforms E, f, G and W into a 1D array. Facilitates
    compatability with scipy.integrate.ode.

    Arguments:

    E, f, G, W -- normal-ordered pieces of Hamiltonian

    Returns:

    concatenation of tensors peeled into 1D arrays"""
    unravel_E = np.reshape(E, -1).astype(np.float32)
    unravel_f = np.reshape(f, -1).astype(np.float32)
    unravel_G = np.reshape(G, -1).astype(np.float32)
    unravel_W = np.reshape(W, -1).astype(np.float32)

    return np.concatenate([unravel_E, unravel_f, unravel_G, unravel_W], axis=0)

# @profile
def ravel(y, bas_len):
    """Transforms 1D array into E, f, G, and W. Facilitates
    compatability with scipy.integrate.ode.

    Arugments:

    y -- 1D data array (output from unravel)
    bas_len -- length of single particle basis

    Returns:

    E, f, G, W -- normal-ordered pieces of Hamiltonian"""

    # bas_len = len(np.append(holes,particles))

    ravel_E = np.reshape(y[0], ()).astype(np.float32)
    ravel_f = np.reshape(y[1:bas_len**2+1], (bas_len, bas_len)).astype(np.float32)
    ravel_G = np.reshape(y[bas_len**2+1:bas_len**2+1+bas_len**4],
                         (bas_len, bas_len, bas_len, bas_len)).astype(np.float32)
    ravel_W = np.reshape(y[bas_len**2+1+bas_len**4:bas_len**2+1+bas_len**4+bas_len**6],
                         (bas_len,bas_len,bas_len,bas_len,bas_len,bas_len)).astype(np.float32)

    return (ravel_E, ravel_f, ravel_G, ravel_W)

def main3b(n_holes, n_particles, ref=None, d=1.0, g=0.5, pb=0.0):
    """Main method uses scipy.integrate.ode to solve the IMSRG3 flow
    equations."""
    tracemalloc.start()

    start = time.time() # start full timer

    initi = time.time() # start instantiation timer

    if ref is None:
        ha = PairingHamiltonian2B(n_holes, n_particles, d=d, g=g, pb=pb)
        ref = ha.reference # this is just for printing
    else:
        ha = PairingHamiltonian2B(n_holes, n_particles, ref=ref, d=d, g=g, pb=pb)
    print("Built Hamiltonian")

    ot = OccupationTensors(ha.sp_basis, ha.reference)
    print("Built occupation tensors")

    wg = WegnerGenerator3B(ha, ot)
    #wg = WhiteGeneratorMP3B(ha)
    print("Built Wegner's generator")

    fl = Flow_IMSRG3(ha, ot)
    print("Built IMSRG3 flow object")

    initf = time.time() # finish instatiation timer

    print("Initialized objects in {:2.4f} seconds\n".format(initf-initi))

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f MB" % (total / 1024**2))


    print("""Pairing model IM-SRG(3) flow:
    d              = {:2.4f}
    g              = {:2.4f}
    pb             = {:2.4f}
    SP basis size  = {:2d}
    n_holes        = {:2d}
    n_particles    = {:2d}
    ref            = {d}""".format(ha.d, ha.g, ha.pb, ha.n_sp_states,
                                        len(ha.holes), len(ha.particles),
                                        d=ref) )

    print("Flowing...")

    # --- Solve the IM-SRG flow
    y0 = unravel(ha.E, ha.f, ha.G, wg.W)
    print(wg.W.shape, wg.eta3B.shape)
    solver = ode(derivative,jac=None)
    solver.set_integrator('vode', method='bdf', order=5, nsteps=500)
    #solver.set_integrator('lsoda', max_order_s=10, nsteps=500)
    solver.set_f_params(ha, ot, wg, fl)
    solver.set_initial_value(y0, 0.)

    sfinal = 50
    ds = 0.1
    s_vals = []
    E_vals = []

    iters = 0
    convergence = 0

    print_columns = ['iter', 
                     's', 
                     'E', 
                     '||eta1b||',
                     '||eta2b||',
                     '||eta3b||']
    column_string = '{: >6}, '
    for i, string in enumerate(print_columns[1::]):
        if i != len(print_columns)-2:
            column_string += '{: >'+str(11)+'}, '
        else:
            column_string += '{: >'+str(11)+'}'
    print(column_string.format(*print_columns))


    while solver.successful() and solver.t < sfinal:

        ys = solver.integrate(solver.t+ds, step=True)
        Es, fs, Gs, Ws = ravel(ys, ha.n_sp_states)
        s_vals.append(solver.t)
        E_vals.append(Es)
        
        iters += 1
        # if iters == 1:
        #     break
        
        # if iters == 10:

        #     for a in range(ha.n_sp_states):
        #         for b in range(ha.n_sp_states):
        #             for c in range(ha.n_sp_states):
        #                 for d in range(ha.n_sp_states):
        #                     me = Gs[a,b,c,d]
        #                     if me != 0:
        #                         print(a,b,c,d, me)
        #     for a in range(ha.n_sp_states):
        #         for b in range(ha.n_sp_states):
        #             for c in range(ha.n_sp_states):
        #                 for d in range(ha.n_sp_states):
        #                     for e in range(ha.n_sp_states):
        #                         for f in range(ha.n_sp_states):
        #                             me = Ws[a,b,c,d,e,f]
        #                             if me != 0:
        #                                 print(a,b,c,d,e,f, me)

        if iters %10 == 0:
#            print("iter: {:>6d} \t scale param: {:0.4f} \t E = {:0.9f}".format(iters, solver.t, Es))

            norm_eta1B = np.linalg.norm(np.ravel(wg.eta1B))
            norm_eta2B = np.linalg.norm(np.ravel(wg.eta2B))
            norm_eta3B = np.linalg.norm(np.ravel(wg.eta3B))

            data_columns = [iters, 
                            solver.t, 
                            Es,
                            norm_eta1B,
                            norm_eta2B,
                            norm_eta3B]
            column_string = '{:>6d}, '
            for i, string in enumerate(print_columns[1::]):
                if i != len(print_columns)-2:
                    column_string += '{: .8f}, '
                else:
                    column_string += '{: .8f}'

#            print(column_string.format(*data_columns))

            print(column_string.format(*data_columns))


        if len(E_vals) > 100 and abs(E_vals[-1] - E_vals[-2]) < 10**-8:
            print("---- Energy converged at iter {:>06d} with energy {:1.8f}\n".format(iters,E_vals[-1]))
            convergence = 1
            break

        if len(E_vals) > 100 and abs(E_vals[-1] - E_vals[-2]) > 1:
            print("---- Energy diverged at iter {:>06d} with energy {:3.8f}\n".format(iters,E_vals[-1]))
            break

        if iters > 20000:
            print("---- Energy diverged at iter {:>06d} with energy {:3.8f}\n".format(iters,E_vals[-1]))
            break

        if iters % 1000 == 0:
            print('Iteration {:>06d}'.format(iters))

    end = time.time() # finish full timer

    time_str = "Program finished in {:2.5f} seconds\n".format(end-start) # output total time
    print(time_str)
    del ha, ot, wg, fl, solver, y0, sfinal, ds # clear resources

    return (convergence, iters, d, g, pb, n_holes+n_particles, s_vals, E_vals, time_str)

def test_exact(plots_dir):

    assert isinstance(plots_dir, str), "Enter plots directory as string"
    assert (plots_dir[-1] == '\\' or
            plots_dir[-1] == '/'), "Directory must end in slash (\\ or /)"

    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    start = -1.0
    stop = 1.0
    num = 21

    g_vals = np.linspace(start, stop, num)

    E_corrs = []
    E_exacts = []
    for g in g_vals:
        data = main3b(4,4, d=1.0, g=g, pb=0.0)
        E_vals = data[7]
        E_corr = E_vals[-1]
        E_exact = ci_matrix.exact_diagonalization(1.0, g, 0.0)

        E_corrs.append(E_corr - (2-g))
        E_exacts.append(E_exact - (2-g))

        plt.figure(figsize=[12,8])
        plt.plot(data[6], data[7])
        plt.ylabel('Energy')
        plt.xlabel('scale parameter')
        plt.title('Convergence for \n g={:2.4f}, pb={:2.4f}'.format(g,0.0))
        
        pb_plots_dir = plots_dir+'pb{:2.4f}\\'.format(0.0)
        if not os.path.exists(pb_plots_dir):
            os.mkdir(pb_plots_dir)

        plt.savefig(pb_plots_dir+'g{:2.4f}_pb{:2.4f}.png'.format(g,0.0))
        plt.close()

    plt.figure(figsize=[12,8])
    plt.plot(g_vals, E_exacts, marker='s')
    plt.plot(g_vals, E_corrs, marker='v')
    plt.ylabel('E$_{corr}$')
    plt.xlabel('g')
    plt.legend(['exact', 'IMSRG(3)'])
    plt.title('Correlation energy with pb = {:2.4f}'.format(pb))
    plt.savefig(plots_dir+'pb{:2.4f}.png'.format(pb))
    plt.close()
    print(E_exacts)



if __name__ == '__main__':
    #test_exact("plots3b/")
    main3b(2,2,g=0.5,pb=0.1)
    #test = main3b(4,4)
    
    #tracemalloc.start()
    
    # for i in range(5):
    #     test = main3b(4,4)
    #     print(test[-1])

    #     snapshot = tracemalloc.take_snapshot()
    #     top_stats = snapshot.statistics('lineno')
    #     total = sum(stat.size for stat in top_stats)
    #     print("Total allocated size: %.1f KiB" % (total / 1024))
    
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    # total = sum(stat.size for stat in top_stats)
    # print("Final allocated size: %.1f KiB" % (total / 1024))
