# Main program for IM-SRG.


# Author: Jacob Davison
# Date:   07/10/2019

# import packages, libraries, and modules
# user files
from hamiltonian import *
from occupation_tensors import *
from generator import *
from flow import *
from plot_data import *
from display_memory import *

# libraries
from scipy.integrate import odeint, ode
import numpy as np
import time
import pickle
import tracemalloc
import os
from memory_profiler import profile

# @profile
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

    E, f, G = ravel(y, hamiltonian.n_sp_states)

    generator.f = f
    generator.G = G

    dE, df, dG = flow.flow(generator)

    dy = unravel(dE, df, dG)

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

    Arugments:

    y -- 1D data array (output from unravel)
    bas_len -- length of single particle basis

    Returns:

    E, f, G -- normal-ordered pieces of Hamiltonian"""

    # bas_len = len(np.append(holes,particles))

    ravel_E = np.reshape(y[0], ())
    ravel_f = np.reshape(y[1:bas_len**2+1], (bas_len, bas_len))
    ravel_G = np.reshape(y[bas_len**2+1:bas_len**2+1+bas_len**4],
                         (bas_len, bas_len, bas_len, bas_len))

    return(ravel_E, ravel_f, ravel_G)

# @profile
def main(n_holes, n_particles, d=1.0, g=0.5, pb=0.0):
    """Main method uses scipy.integrate.ode to solve the IMSRG flow
    equations."""
    start = time.time()

    ha = PairingHamiltonian2B(n_holes, n_particles, d, g, pb)
    ot = OccupationTensors(ha.sp_basis, ha.reference)
    wg = WegnerGenerator(ha, ot)
    fl = Flow_IMSRG2(ha, ot)

    print("""Pairing model IM-SRG flow:
    d              = {:2.4f}
    g              = {:2.4f}
    pb             = {:2.4f}
    SP basis size  = {:2d}
    n_holes        = {:2d}
    n_particles    = {:2d}""".format(ha.d, ha.g, ha.pb, ha.n_sp_states, len(ha.holes), len(ha.particles)) )

    print("Flowing...")

    # --- Solve the IM-SRG flow
    y0 = unravel(ha.E, ha.f, ha.G)

    solver = ode(derivative,jac=None)
    solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
    solver.set_f_params(ha, ot, wg, fl)
    solver.set_initial_value(y0, 0.)

    sfinal = 50
    ds = 0.1
    s_vals = []
    E_vals = []

    iters = 0
    convergence = 0
    while solver.successful() and solver.t < sfinal:

        ys = solver.integrate(sfinal, step=True)
        Es, fs, Gs = ravel(ys, ha.n_sp_states)
        s_vals.append(solver.t)
        E_vals.append(Es)

        iters += 1

#         if iters %10 == 0: print("iter: {:>6d} \t scale param: {:0.4f} \t E = {:0.9f}".format(iters, solver.t, Es))

        if len(E_vals) > 20 and abs(E_vals[-1] - E_vals[-2]) < 10**-8:
            print("---- Energy converged at iter {:>06d} with energy {:1.8f}\n".format(iters,E_vals[-1]))
            convergence = 1
            break

        if len(E_vals) > 20 and abs(E_vals[-1] - E_vals[-2]) > 2:
            print("---- Energy diverged at iter {:>06d} with energy {:3.8f}\n".format(iters,E_vals[-1]))
            break

    end = time.time()
    time_str = "{:2.5f}\n".format(end-start)

    del ha, ot, wg, fl, solver, y0, sfinal, ds

    return (convergence, iters, d, g, pb, n_holes+n_particles, s_vals, E_vals, time_str)

def exact_diagonalization(d, g):
    """Result of exact diagonalization in spin=0 block of
    pairing Hamiltonian, given 8 single particle states (4 hole states
    and 4 particle states).

    Arguments:

    d -- energy level spacing
    g -- pairing strength

    Returns:

    E -- ground state energy
    """
    H = [[2*d-g, -g/2, -g/2, -g/2, -g/2, 0],
         [-g/2, 4*d-g, -g/2, -g/2, 0, -g/2],
         [-g/2, -g/2, 6*d-g, 0, -g/2, -g/2],
         [-g/2, -g/2, 0, 6*d-g, -g/2, -g/2],
         [-g/2, 0, -g/2, -g/2, 8*d-g, -g/2],
         [0, -g/2, -g/2, -g/2, -g/2, 10*d-g]]

    w, v = np.linalg.eig(H)
    E = w[0]

    return E

# @profile
def scan_params():
    tracemalloc.start()

    log_dir = "logs\\"
    plot_dir = "plots\\"

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        print("Created log directory at "+log_dir)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        print("Created plots directory at "+plot_dir+"\n")


    if os.path.isfile(log_dir+'total_mem.txt'):
        os.remove(log_dir+'total_mem.txt')

    start = 0.0001
    stop = 5
    num = 200

    gsv = np.linspace(start, stop, num)
    pbv = np.copy(gsv)
    # gsv = np.append(np.linspace(-stop,-start,num), np.linspace(start, stop,num))
    # pbs = np.copy(gsv)

    # data_container = np.array([])
    for g in gsv:
        pb_list = np.array(['convergence', 'iters', 'd', 'g', 'pb', 'n_sp_states', 's_vals', 'E_vals', 'time_str'])
        for pb in pbv:
            data = main(4,4, d=stop, g=g, pb=pb) # (convergence, s_vals, E_vals)
            pb_list = np.vstack([pb_list, data])

            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            total_mem = sum(stat.size for stat in top_stats)

            with open(log_dir+'total_mem.txt', 'a') as f:
                f.write('{:.1f}\n'.format(total_mem))

            if data[0] == 0:
                print("Energy diverged. Continuing to next g value...\n")
                break

            del data, snapshot, top_stats, total_mem

        with open('{:s}g-{:2.4f}.pickle'.format(log_dir,g), 'wb') as f:
            pickle.dump(pb_list, f, pickle.HIGHEST_PROTOCOL)

        plot_data(log_dir, plot_dir)

        del pb_list # delete resources that have been written

        # data_container = np.append(data_container, pb_list)
def test_exact():
    start = -1.0
    stop = 1.0
    num = 15

    g_vals = np.linspace(start, stop, num)

    for pb in g_vals:
        E_corrs = []
        E_exacts = []
        for g in g_vals:
            data = main(4,4, d=1.0, g=g, pb=pb)
            E_vals = data[7]
            E_corr = E_vals[-1]
            E_exact = exact_diagonalization(1.0, g)

            E_corrs.append(E_corr - (2-g))
            E_exacts.append(E_exact - (2-g))

            plt.figure()
            plt.plot(data[6], data[7])
            plt.ylabel('Energy')
            plt.xlabel('scale parameter')
            plt.title('Convergence for \n g={:2.4f}, pb={:2.4f}'.format(g,pb))

            pb_plots_dir = 'plots\\pb{:2.4f}\\'.format(pb)
            if not os.path.exists(pb_plots_dir):
                os.mkdir(pb_plots_dir)

            plt.savefig(pb_plots_dir+'g{:2.4f}_pb{:2.4f}.png'.format(g,pb))
            plt.close()

        plt.figure()
        plt.plot(g_vals, E_exacts, marker='s')
        plt.plot(g_vals, E_corrs, marker='v')
        plt.ylabel('E$_{corr}$')
        plt.xlabel('g')
        plt.legend(['exact', 'IMSRG(2)'])
        plt.title('Correlation energy with pb = {:2.4f}'.format(pb))
        plt.savefig('plots\\pb{:2.4f}.png'.format(pb))
        plt.close()

def test_refs():
    start = -1.0
    stop = 1.0
    num = 10

    g_vals = np.linspace(start, stop, num)
    refs = [[1,1,1,1,0,0,0,0],
            [1,1,0,0,1,1,0,0],
            [1,1,0,0,0,0,1,1],
            [0,0,1,1,1,1,0,0],
            [0,0,1,1,0,0,1,1],
            [0,0,0,0,1,1,1,1]]

    for pb in g_vals:
        E_corrs = []
        E_exacts = []
        for g in g_vals:
            data = main(4,4, d=1.0, g=g, pb=pb)
            E_vals = data[7]
            E_corr = E_vals[-1]
            E_exact = exact_diagonalization(1.0, g)

            E_corrs.append(E_corr - (2-g))
            E_exacts.append(E_exact - (2-g))

            plt.figure()
            plt.plot(data[6], data[7])
            plt.ylabel('Energy')
            plt.xlabel('scale parameter')
            plt.title('Convergence for \n g={:2.4f}, pb={:2.4f}'.format(g,pb))

            pb_plots_dir = 'plots\\pb{:2.4f}\\'.format(pb)
            if not os.path.exists(pb_plots_dir):
                os.mkdir(pb_plots_dir)

            plt.savefig(pb_plots_dir+'g{:2.4f}_pb{:2.4f}.png'.format(g,pb))
            plt.close()

        plt.figure()
        plt.plot(g_vals, E_exacts, marker='s')
        plt.plot(g_vals, E_corrs, marker='v')
        plt.ylabel('E$_{corr}$')
        plt.xlabel('g')
        plt.legend(['exact', 'IMSRG(2)'])
        plt.title('Correlation energy with pb = {:2.4f}'.format(pb))
        plt.savefig('plots\\pb{:2.4f}.png'.format(pb))
        plt.close()

if __name__ == '__main__':
    test_exact()
