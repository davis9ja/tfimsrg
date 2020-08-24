#
# Various IM-SRG(2) tests. Not intended for use outside of main.py module.
# 
# Author:  Jacob Davison
# Version: 08/23/2019

import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import random
import pickle

import oop_imsrg.ci_pairing.cipy_pairing_plus_ph as ci_matrix
import benchmarking_wd.imsrg_pairing as pypairing

def test_refs(plots_dir, main):
    """The purpose of this test is to compare the result of full 
    configuration interaction (i.e. exact diagonalization) to the result of 
    IM-SRG(2) for every possible reference state created from 8 single
    particle states and 4 particles.
    
    Arguments:

    plots_dir -- directory to store plot outputs
    main -- main method for IM-SRG(2) (PASSED IN FROM main.py)

    Returns:

    Nothing"""
    
    assert isinstance(plots_dir, str), "Enter plots directory as string"
    assert (plots_dir[-1] == '\\' or
            plots_dir[-1] == '/'), "Directory must end in slash (\\ or /)"

    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    start = -1.0
    stop = 1.0
    num = 5

    g_vals = np.linspace(start, stop, num)
    refs = list(map("".join, itertools.permutations('11110000')))
    refs = list(dict.fromkeys(refs)) # remove duplicates
    refs = [list(map(int, list(ref))) for ref in refs]

    fig_corr = plt.figure(figsize=[12,8])
    fig_conv = plt.figure(figsize=[12,8])
    ax_corr = fig_corr.add_subplot(1,1,1)
    ax_conv = fig_conv.add_subplot(1,1,1)

    for pb in g_vals:
        E_exacts = []

        for g in g_vals:
            E_corrs = []
            # plt.figure(figsize=[12,8])
            ref_rand = random.sample(refs, 20)
            E_exact = ci_matrix.exact_diagonalization(1.0, g, pb)
            # E_exacts.append(E_exact - (2-g))
            E_exacts.append(E_exact)
            refs_conv = []
            for ref in refs:

                data = main(4,4, ref=ref, d=1.0, g=g, pb=pb)

                if data[0] == 1:
                    E_vals = data[7]
                    E_corr = E_vals[-1]

                    # E_corrs.append(E_corr - (2-g))
                    E_corrs.append(E_corr)
                    refs_conv.append(ref)

                    ax_conv.plot(data[6], data[7])

            # ymin, ymax = ax_conv.get_ylim()
            # ax_conv.set_ylim(bottom=0.7*ymin, top=0.7*ymax)
            ax_conv.set_ylabel('Energy')
            ax_conv.set_xlabel('scale parameter')
            ax_conv.set_title('Convergence for \n g={:2.4f}, pb={:2.4f}'.format(g,pb))
            ax_conv.legend(refs_conv)
            pb_plots_dir = plots_dir+'pb{:2.4f}\\'.format(pb)
            if not os.path.exists(pb_plots_dir):
                os.mkdir(pb_plots_dir)

            fig_conv.savefig(pb_plots_dir+'g{:2.4f}_pb{:2.4f}.png'.format(g,pb))
            ax_conv.clear()

            with open(plots_dir+'g{:2.4f}_pb{:2.4f}.txt'.format(g,pb), 'w') as f:
                f.write('Pairing model: d = 1.0, g = {:2.4f}, pb = {:2.4f}\n'.format(g, pb))
                f.write('Full CI diagonalization -- correlation energy: {:2.8f}\n'.format(E_exacts[-1]))
                f.write('IMSRG(2) variable reference state -- correlation energies:\n')
                if len(refs_conv) == 0:
                    f.write('No convergence for range of reference states tested\n')
                else:
                    for i in range(len(refs_conv)):
                        f.write('{:2.8f} | {d}\n'.format(E_corrs[i], d=refs_conv[i]))

                    f.write('Ground state from IMSRG(2):\n')
                    e_sort_ind = np.argsort(E_corrs)
                    print(e_sort_ind)
                    print(E_corrs)
                    f.write('{:2.8f} | {d}\n'.format(E_corrs[e_sort_ind[0]], d=refs_conv[e_sort_ind[0]]))

        # corr_data = np.reshape(E_corrs, (len(g_vals), 2))
        # ax_corr.plot(g_vals, E_exacts, marker='s')
        # for i in range(10):
        #     ax_corr.plot(g_vals, corr_data[:,i], marker='v')
        # ymin, ymax = ax_corr.get_ylim()
        # ax_corr.set_ylim(bottom=0.7*ymin, top=0.7*ymax)
        # ax_corr.set_ylabel('E$_{corr}$')
        # ax_corr.set_xlabel('g')
        # ax_corr.legend(['exact', 'IMSRG(2)'])
        # ax_corr.set_title('Correlation energy with pb = {:2.4f}'.format(pb))
        # fig_corr.savefig(plots_dir+'pb{:2.4f}.png'.format(pb))
        # ax_corr.clear()

        
def test_exact(plots_dir, main):
    """
    The purpose of this test is to compare the result of full configuration
    interaction (i.e. exact diagonalization) to the result of IM-SRG(2) for
    several values of g, at a fixed value of d=1.0 and pb=0.0.

    Arguments:

    plots_dir -- directory to store plot outputs
    main -- main method for IM-SRG(2) (PASSED IN FROM main.py)

    Returns:

    Nothing"""
    assert isinstance(plots_dir, str), "Enter plots directory as string"
    assert (plots_dir[-1] == '\\' or
            plots_dir[-1] == '/'), "Directory must end in slash (\\ or /)"

    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    start = -1.0
    stop = 1.0
    num = 21

    g_vals = np.linspace(start, stop, num)

    for pb in g_vals:
        pb = 0.0
        E_corrs = []
        E_exacts = []
        E_pys = []
        for g in g_vals:
            data = main(4,4, d=1.0, g=g, pb=0.0)
            E_vals = data[7]
            E_corr = E_vals[-1]
            E_exact = ci_matrix.exact_diagonalization(1.0, g, pb)
            E_py = pypairing.main(4, g)

            E_corrs.append(E_corr - (2-g))
            E_exacts.append(E_exact - (2-g))
            E_pys.append(E_py - (2-g))
            
            # plt.figure(figsize=[12,8])
            # plt.plot(data[6], data[7])
            # plt.ylabel('Energy')
            # plt.xlabel('scale parameter')
            # plt.title('Convergence for \n g={:2.4f}, pb={:2.4f}'.format(g,pb))

            # pb_plots_dir = plots_dir+'pb{:2.4f}/'.format(pb)
            # if not os.path.exists(pb_plots_dir):
            #     os.mkdir(pb_plots_dir)

            # plt.savefig(pb_plots_dir+'g{:2.4f}_pb{:2.4f}.png'.format(g,pb))
            #plt.close()

        # plt.figure(figsize=[12,8])
        # plt.plot(g_vals, E_exacts, marker='s')
        # plt.plot(g_vals, E_corrs, marker='v')
        # plt.plot(g_vals, E_pys, marker='x')
        # plt.ylabel('E$_{corr}$')
        # plt.xlabel('g')
        # plt.legend(['CI', 'IMSRG(2) TN', 'IMSRG(2) PY'])
        # plt.title('Correlation energy with pb = {:2.4f}'.format(pb))
        # plt.savefig(plots_dir+'pb{:2.4f}.png'.format(pb))
        
        if not os.path.exists('data_pickles1/'):
            os.mkdir('data_pickles1/')
            
        fulldata = [g_vals, E_exacts, E_corrs, E_pys]
        with open('data_pickles1/fulldata.pickle', 'wb') as f:
            pickle.dump(fulldata, f, pickle.HIGHEST_PROTOCOL)
        
        #plt.close()
        #print(E_exacts)
        break

def scan_params(main):
    """
    The purpose of this test is to study the convergence of the IM-SRG(2)
    for several values of d, g, and pb. Also tracks memory allocation. This
    test probably needs to be updated to be useful.

    Arguments:

    main -- main method for IM-SRG(2) (PASSED IN FROM main.py)

    Returns:

    Nothing"""

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
