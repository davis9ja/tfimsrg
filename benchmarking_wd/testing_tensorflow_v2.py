#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/davis9ja/im-srg_tensorflow/blob/master/testing_tensorflow_v2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.INFO)
import numpy as np
import sys
from scipy.integrate import odeint, ode
import matplotlib.pyplot as plt
print(tf.__version__)

# In[2]:


# --- BUILD HAMILTONIAN -----------
#@profile
def build_hamiltonian(n_hole_states, n_particle_states):
    numh = n_hole_states
    nump = n_particle_states
    nums = numh + nump
    
    ref = np.append(np.ones(numh), np.zeros(nump))
    holes = np.arange(numh)
    particles = np.arange(numh,numh+nump)
    B1 = np.append(holes,particles)
    
    # one body part of Hamiltonian is floor-division of basis index
    # matrix elements are (P-1) where P is energy level
    H1B = np.diag(np.floor_divide(B1,2))

    H2B = np.zeros((nums, nums, nums, nums))
    for p in B1:
        for q in B1:
            for r in B1:
                for s in B1:

                    pp = np.floor_divide(p,2)
                    qp = np.floor_divide(q,2)
                    rp = np.floor_divide(r,2)
                    sp = np.floor_divide(s,2)

                    ps = 1 if p%2==0 else -1
                    qs = 1 if q%2==0 else -1
                    rs = 1 if r%2==0 else -1
                    ss = 1 if s%2==0 else -1

                    if pp != qp or rp != sp:
                        continue
                    if ps == qs or rs == ss:
                        continue
                    if ps == rs and qs == ss:
                        H2B[p,q,r,s] = -0.25
                    if ps == ss and qs == rs:
                        H2B[p,q,r,s] = 0.25
                        
    return (H1B, H2B, ref, holes, particles, B1)

# covers na - nb
#@profile
def get_occA(B1_basis, ref):
    n = len(B1_basis)
    occA = np.zeros((n,n,n,n))
    
    for a in B1_basis:
        for b in B1_basis:
            occA[a,b,a,b] = ref[a] - ref[b]
            
    return occA
        
# covers (1-na-nb)
#@profile
def get_occB(B1_basis, ref):
    n = len(B1_basis)    
    occB = np.zeros((n,n,n,n))
    
    for a in B1_basis:
        for b in B1_basis:
            occB[a,b,a,b] = 1 - ref[a] - ref[b]
            
    return occB
        
# covers na*nb + (1-na-nb)*nc
#@profile
def get_occC(B1_basis, ref):
    n = len(B1_basis)        
    occC = np.zeros((n,n,n,n,n,n))
    
    for a in B1_basis:
        for b in B1_basis:
            for c in B1_basis:
                occC[a,b,c,a,b,c] = ref[a]*ref[b] + (1-ref[a]-ref[b])*ref[c]
                
    return occC

# covers na*nb*(1-nc-nd) + na*nb*nc*nd
#@profile
def get_occD(B1_basis, ref):
    n = len(B1_basis)    
    occD = np.zeros((n,n,n,n))
    
    for a in B1_basis:
        for b in B1_basis:
            for c in B1_basis:
                for d in B1_basis:
                    occD[a,b,c,d] = ref[a]*ref[b]*(1-ref[c]-ref[d])+ref[a]*ref[b]*ref[c]*ref[d]
                    
    return occD
                


# In[3]:


# --- NORMAL ORDER HAMILTONIAN -----------
# Calculate 0b, 1b, 2b pieces 
#
# zero-body piece is scalar
# one-body piece is rank 2 tensor
# two-body piece is rank 4 tensor
#@profile
def normal_order(H1B_t, H2B_t, holes):
    
    H1B_t = tf.convert_to_tensor(H1B_t, dtype=tf.float32, name='a')
    H2B_t = tf.convert_to_tensor(H2B_t, dtype=tf.float32, name='b')
    holes = tf.convert_to_tensor(holes, dtype=tf.int32)
    
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        
        # - Calculate 0B tensor
        # E = tf.Variable(0.0)
        contr_1b = tf.map_fn(lambda i: H1B_t[i,i], holes, dtype=tf.float32)
        contr_2b = tf.map_fn(lambda i: H2B_t[i,:,i,:], holes, dtype=tf.float32)

        E_1b = tf.reduce_sum(contr_1b, 0)
        E_2b = 0.5*tf.reduce_sum(contr_2b, [0,1,2])
        E = tf.add_n([E_1b, E_2b])

        # - Calculate 1B tensor
        contr_2b = tf.map_fn(lambda i: H2B_t[:,i,:,i], holes, dtype=tf.float32)
        contr_2b = tf.reduce_sum(contr_2b,0) # sum over holes

        f = tf.add_n([H1B_t, contr_2b])

        # - Calculate 2B tensor
        G = tf.identity(H2B_t)
        
        E_e, f_e, G_e = sess.run([E, f, G])
#         E_e = E.eval()
#         f_e = f.eval()
#         G_e = G.eval()
        
#         print(sess.run([E, f, G]))
        
    tf.reset_default_graph()
    
    return (E_e, f_e, G_e)


# In[4]:


# # --- SET UP WHITE'S GENERATOR -----------

# # TODO: try reduce_sum, einsum, tensordot?, other methods

# def white(f, G, holes, particles):
  
#     # - Calculate 1b generator tensor
#     # indices are constructed by all possible combinations of pp and hh states
#     p1_ind = tf.reshape(tf.broadcast_to(particles,[4,4]), [-1])
#     p2_ind = tf.reshape(tf.transpose(tf.broadcast_to(particles,[4,4])), [-1])
#     pp_indices = tf.stack([p1_ind, p2_ind], axis=1)
#     pp_updates = tf.gather_nd(f, pp_indices)
    
#     h1_ind = tf.reshape(tf.broadcast_to(holes,[4,4]), [-1])
#     h2_ind = tf.reshape(tf.transpose(tf.broadcast_to(holes,[4,4])), [-1])
#     hh_indices = tf.stack([h1_ind, h2_ind], axis=1)
#     hh_updates = tf.gather_nd(f, hh_indices)
    
#     fpp = tf.scatter_nd(pp_indices, pp_updates, f.shape)
#     fhh = tf.scatter_nd(hh_indices, hh_updates, f.shape)
    
#     # indices are constructed by all possible combinations of phph states
#     ind1_C = tf.broadcast_to(particles,[64,4])
#     ind1_TC = tf.transpose(ind1_C) 
#     ind1 = tf.reshape(ind1_TC,[-1])

#     ind2_C = tf.broadcast_to(holes,[16,16])
#     ind2_TC = tf.transpose(ind2_C)
#     ind2 = tf.reshape(ind2_TC,[-1])

#     ind3_C = tf.broadcast_to(particles,[4,64])
#     ind3_TC = tf.transpose(ind3_C) 
#     ind3 = tf.reshape(ind3_TC,[-1])

#     ind4_C = tf.broadcast_to(holes,[1,256])
#     ind4_TC = tf.transpose(ind4_C)
#     ind4 = tf.reshape(ind4_TC,[-1])

#     phph_indices = tf.stack([ind1,ind2,ind3,ind4],axis=1)
#     phph_updates = tf.gather_nd(G, phph_indices)

#     Gphph = tf.scatter_nd(phph_indices, phph_updates, G.shape)
#     Gphph_red = tf.reduce_sum(Gphph, [0,1])
    
#     delta_ph = tf.add(tf.subtract(fpp, fhh), Gphph_red)
    
#     ph_indices = tf.stack([p1_ind, h2_ind], axis=1)
#     ph_updates = tf.gather_nd(f, ph_indices)
#     fph = tf.scatter_nd(ph_indices, ph_updates, f.shape)
    
#     eta1B = tf.div_no_nan(fph, delta_ph)
    
#     # - Calculate 2b generator tensor
#     eta2B = tf.zeros(G.shape)
    
    
    
    
# #     return temp
#     return (eta1B, pp_indices, delta_ph)
# #     return (eta1B, eta2B)


# In[5]:


# --- SET UP WEGNER'S GENERATOR -----------
#@profile
def wegner(f, G, holes, particles, occA, occB, occC, occD):
    
    f = tf.convert_to_tensor(f, dtype=tf.float32)
    G = tf.convert_to_tensor(G, dtype=tf.float32)
    holes = tf.convert_to_tensor(holes, dtype=tf.int32)
    particles = tf.convert_to_tensor(particles, dtype=tf.int32)
    occA_t = tf.convert_to_tensor(occA, dtype=tf.float32)
    occB_t = tf.convert_to_tensor(occB, dtype=tf.float32)
    occC_t = tf.convert_to_tensor(occC, dtype=tf.float32)
    occD_t = tf.convert_to_tensor(occD, dtype=tf.float32)
    
    with tf.Session() as sess:
        plen = tf.size(particles)
        hlen = tf.size(holes)

        # --- Need to decouple diagonal and off-diagonal elements; procedure in Ch.10 AACCNP
        
        # Decoupling 1B piece
        # indices are constructed by all possible combinations of particle-hole(hole-particle) states
        particles_b = tf.broadcast_to(particles, [plen,plen])
        holes_b = tf.broadcast_to(holes, [hlen, hlen])
        ph_comb = tf.concat([particles_b, holes_b], 0)
        hp_comb = tf.transpose(tf.concat([holes_b, particles_b], 1))
        
        col_indices =tf.reshape(ph_comb, [-1])
        row_indices = tf.reshape(hp_comb, [-1])
        ph_indices = tf.stack([row_indices, col_indices], axis=1)
        ph_updates = tf.gather_nd(f, ph_indices)

        fod = tf.scatter_nd(ph_indices,ph_updates,f.shape)
        fd = tf.subtract(f,fod)

        # Decoupling 2B piece
        # indices are constructed by all possible combinations of pphh(hhpp) states
        ind1_C = tf.concat([tf.broadcast_to(holes,[hlen**3,hlen]), tf.broadcast_to(particles,[plen**3,plen])],1)
        ind1_TC = tf.transpose(ind1_C) 
        ind1 = tf.reshape(ind1_TC,[-1])

        ind2_C = tf.concat([tf.broadcast_to(holes,[hlen**2,hlen**2]),tf.broadcast_to(particles,[plen**2,plen**2])],1)
        ind2_TC = tf.transpose(ind2_C)
        ind2 = tf.reshape(ind2_TC,[-1])

        ind3_C = tf.concat([tf.broadcast_to(particles,[plen,plen**3]),tf.broadcast_to(holes,[hlen,hlen**3])],1)
        ind3_TC = tf.transpose(ind3_C) 
        ind3 = tf.reshape(ind3_TC,[-1])

        ind4_C = tf.concat([tf.broadcast_to(particles,[1,plen**4]),tf.broadcast_to(holes,[1,plen**4])],1)
        ind4_TC = tf.transpose(ind4_C)
        ind4 = tf.reshape(ind4_TC,[-1])

        pphh_indices = tf.stack([ind1,ind2,ind3,ind4],axis=1)
        pphh_updates = tf.gather_nd(G, pphh_indices)

        God = tf.scatter_nd(pphh_indices,pphh_updates,G.shape)
        Gd = tf.subtract(G,God)


        # --- 1B piece

        # Calculate 1B-1B contribution
        fd_fod = tf.tensordot(fd,fod,1)
        fd_fod_T = tf.transpose(fd_fod)
        eta1B_1b1b = tf.subtract(fd_fod, fd_fod_T)

        # Calculate 1B-2B contribution
        fd_God = tf.tensordot(fd, tf.tensordot(occA_t,God,([0,1],[2,0])),([0,1],[2,0]))
        fod_Gd = tf.tensordot(fod, tf.tensordot(occA_t,Gd,([0,1],[2,0])),([0,1],[2,0]))
        eta1B_1b2b = tf.subtract(fd_God, fod_Gd)

        # Calculate 2B-2B contribution
        Gd_God = tf.tensordot(Gd, tf.tensordot(occC_t,God,([0,1,2],[0,1,2])),([2,3,1],[0,1,2]))
        Gd_God_T = tf.transpose(Gd_God)
        scaled_sub = 0.5*tf.subtract(Gd_God,Gd_God_T)
        eta1B_2b2b = scaled_sub

        eta1B = tf.add_n([eta1B_1b1b, eta1B_1b2b, eta1B_2b2b])



        # --- 2B piece

        # Calculate 1B-2B contribution
        fdGod_fodGd_ij = tf.subtract( tf.tensordot(fd,God,[[1],[0]]), tf.tensordot(fod,Gd,[[1],[0]]) )
        fdGod_fodGd_ij_T = tf.transpose(fdGod_fodGd_ij, perm=[1,0,2,3])
        ij_term = tf.subtract(fdGod_fodGd_ij,fdGod_fodGd_ij_T)

        fdGod_fodGd_kl = tf.subtract( tf.tensordot(fd,God,[[0],[2]]), tf.tensordot(fod,Gd,[[0],[2]]) )
        fdGod_fodGd_kl = tf.transpose(fdGod_fodGd_kl,perm=[1,2,0,3]) # permute back to i,j,k,l order
        fdGod_fodGd_kl_T = tf.transpose(fdGod_fodGd_kl,perm=[0,1,3,2])
        kl_term = tf.subtract(fdGod_fodGd_kl,fdGod_fodGd_kl_T)

        eta2B_1b2b = tf.subtract(ij_term,kl_term)


        # Calculate 2B-2B contribution
        GdGod_occB = tf.tensordot(Gd, tf.tensordot(occB_t, God, [[0,1],[0,1]]), [[2,3],[0,1]])
        GodGd_occB = tf.tensordot(God, tf.tensordot(occB_t, Gd, [[0,1],[0,1]]), [[2,3],[0,1]])
        scaled_sub = 0.5*tf.subtract(GdGod_occB,GodGd_occB)

        eta2B_2b2b_B = scaled_sub

        GdGod = tf.tensordot(Gd,God,[[0,2],[2,0]])
        GdGod = tf.transpose(GdGod,perm=[0,2,1,3]) # permute back to i,j,k,l order
        GdGod_occA = tf.tensordot(occA_t,GdGod,[[2,3],[0,1]])
        GdGod_occA_Tij = tf.transpose(GdGod_occA,perm=[1,0,2,3])
        GdGod_occA_Tkl = tf.transpose(GdGod_occA,perm=[0,1,3,2])
        GdGod_occA_Tijkl = tf.transpose(GdGod_occA,perm=[1,0,3,2])
        sub1 = tf.subtract(GdGod_occA,GdGod_occA_Tij)
        sub2 = tf.subtract(sub1,GdGod_occA_Tkl)
        add3 = tf.add(sub2,GdGod_occA_Tijkl)

        eta2B_2b2b_A = add3

        eta2B = tf.add_n([eta2B_1b2b, eta2B_2b2b_B, eta2B_2b2b_A])
        
        eta1B_e = eta1B.eval()
        eta2B_e = eta2B.eval()
        
    tf.reset_default_graph()
    
    return (eta1B_e, eta2B_e)


# In[6]:


# --- WRITE FLOW EQUATIONS -----------
#@profile
def flow(f, G, eta1B, eta2B, holes, particles, occA, occB, occC, occD):
    
    f = tf.convert_to_tensor(f, dtype=tf.float32)
    G = tf.convert_to_tensor(G, dtype=tf.float32)
    eta1B = tf.convert_to_tensor(eta1B, dtype=tf.float32)
    eta2B = tf.convert_to_tensor(eta2B, dtype=tf.float32)
    holes = tf.convert_to_tensor(holes, dtype=tf.int32)
    particles = tf.convert_to_tensor(particles, dtype=tf.int32)
    occA_t = tf.convert_to_tensor(occA, dtype=tf.float32)
    occB_t = tf.convert_to_tensor(occB, dtype=tf.float32)
    occC_t = tf.convert_to_tensor(occC, dtype=tf.float32)
    occD_t = tf.convert_to_tensor(occD, dtype=tf.float32)
    
    with tf.Session() as sess:
        
        # --- 0B piece

        # Calculate 1B-1B contribution (full contraction)
        occA_e1 = tf.tensordot(occA_t, eta1B, [[0,1],[0,1]])
        occA_e1_f = tf.tensordot(occA_e1, f, [[0,1],[1,0]])
        dE_1b1b = tf.identity(occA_e1_f)

        # Calculate 2B-2B contribution (full contraction)
    #     e2_occD = tf.tensordot(eta2B, occD_t, [[0,1,2,3],[0,1,2,3]])
        e2_occD = tf.matmul(eta2B, occD_t)
        e2_occD_G = 0.5*tf.tensordot(e2_occD, G, [[0,1,2,3],[2,3,0,1]])
        dE_2b2b = tf.identity(e2_occD_G)

        dE = tf.add_n([dE_1b1b, dE_2b2b])

        # --- 1B piece

        # Calculate 1B-1B contribution (contraction over 1 index)
        e1_f = tf.tensordot(eta1B,f,[[1],[0]])
        e1_f_T = tf.transpose(e1_f)
        e1_f_add = tf.add(e1_f,e1_f_T)
        df_1b1b = tf.identity(e1_f_add)

        # Calculate 1B-2B contribution (contraction over 2 indices)
        occA_e1_G = tf.tensordot(occA_t, tf.tensordot(eta1B,G,[[0,1],[2,0]]), [[2,3],[0,1]])
        occA_f_e2 = tf.tensordot(occA_t, tf.tensordot(f,eta2B,[[0,1],[2,0]]), [[2,3],[0,1]])
        sub_1b2b = tf.subtract(occA_e1_G, occA_f_e2)
        df_1b2b = tf.identity(sub_1b2b)

        # Calculate 2B-2B contribution (contraction over 3 indices)
        e2_occC_G = tf.tensordot(eta2B, tf.tensordot(occC_t,G,[[3,4,5],[0,1,2]]), [[2,3,0],[0,1,2]])
        e2_occC_G_T = tf.transpose(e2_occC_G)
        add_2b2b = 0.5*tf.add(e2_occC_G,e2_occC_G_T)
        df_2b2b = tf.identity(add_2b2b)

        df = tf.add_n([df_1b1b, df_1b2b, df_2b2b])

        # --- 2B piece

        # Calculate 1B-2B contribution (contraction over 1 index)
        e1G_fe2_ij = tf.subtract(tf.tensordot(eta1B,G,[[1],[0]]), tf.tensordot(f,eta2B,[[1],[0]]))
        e1G_fe2_ij_T = tf.transpose(e1G_fe2_ij, perm=[1,0,2,3])
        ij_term = tf.subtract(e1G_fe2_ij,e1G_fe2_ij_T)

        e1G_fe2_kl = tf.subtract(tf.tensordot(eta1B,G,[[0],[2]]), tf.tensordot(f,eta2B,[[0],[2]]))
        e1G_fe2_kl = tf.transpose(e1G_fe2_kl, perm=[1,2,0,3]) # permute to i,j,k,l order
        e1G_fe2_kl_T = tf.transpose(e1G_fe2_kl, perm=[0,1,3,2])
        kl_term = tf.subtract(e1G_fe2_kl,e1G_fe2_kl_T)

        dG_1b2b = tf.identity(tf.subtract(ij_term, kl_term))

        # Calculate 2B-2B contribution (occB term)
        e2_occB_G = tf.tensordot(eta2B, tf.tensordot(occB_t, G, [[2,3],[0,1]]), [[2,3],[0,1]])
        G_occB_e2 = tf.tensordot(G, tf.tensordot(occB_t, eta2B, [[2,3],[0,1]]), [[2,3],[0,1]])
        sub_term = 0.5*tf.subtract(e2_occB_G, G_occB_e2)

        dG_2b2b_B = tf.identity(sub_term)

        # Calculate 2B-2B contribution (occA term)
        e2G = tf.tensordot(eta2B, G, [[0,2],[2,0]])
        e2G = tf.transpose(e2G, perm=[0,2,1,3]) # permute back to i,j,k,l order
        e2G_occA = tf.tensordot(occA_t, e2G, [[2,3],[0,1]])
        e2G_occA_Tij = tf.transpose(e2G_occA, perm=[1,0,2,3])
        e2G_occA_Tkl = tf.transpose(e2G_occA, perm=[0,1,3,2])
        e2G_occA_Tijkl = tf.transpose(e2G_occA, perm=[1,0,3,2])
        sub1 = tf.subtract(e2G_occA, e2G_occA_Tij)
        sub2 = tf.subtract(sub1, e2G_occA_Tkl)
        add3 = tf.add(sub2, e2G_occA_Tijkl)

        dG_2b2b_A = tf.identity(add3)

        dG = tf.add_n([dG_1b2b, dG_2b2b_B, dG_2b2b_A])
        
        dE_e = dE.eval()
        df_e = df.eval()
        dG_e = dG.eval()
    
    tf.reset_default_graph()
    
    return (dE_e, df_e, dG_e)


# In[7]:


# --- DEFINE DERIVATIVE TO PASS INTO ODEINT SOLVER -----------
#@profile
def derivative(t, y, holes, particles, occA, occB, occC, occD):
    
    E, f, G = ravel(y, holes, particles)

    eta1B, eta2B = wegner(f, G, holes, particles, occA, occB, occC, occD)
    
    dE, df, dG = flow(f, G, eta1B, eta2B, holes, particles, occA, occB, occC, occD)
    
    dy = unravel(dE, df, dG)
    
    return dy


# In[8]:


# --- CONVERT NORMAL ORDERED TENSORS INTO RANK 1 TENSOR -----------
# Quality-of-life methods that facilite compatibility with scipy.ode
#@profile
def unravel(E, f, G):
    unravel_E = np.reshape(E, -1)
    unravel_f = np.reshape(f, -1)
    unravel_G = np.reshape(G, -1)
    
    return np.concatenate([unravel_E, unravel_f, unravel_G], axis=0)

#@profile
def ravel(y, holes, particles):
    
    bas_len = len(np.append(holes,particles))
    
    ravel_E = np.reshape(y[0], ())
    ravel_f = np.reshape(y[1:bas_len**2+1], (bas_len, bas_len))
    ravel_G = np.reshape(y[bas_len**2+1:bas_len**2+1+bas_len**4], 
                         (bas_len, bas_len, bas_len, bas_len))
    
    return(ravel_E, ravel_f, ravel_G)
    


# In[9]:


# --- MAIN PROCEDURE -----------
# Build and normal-order Hamiltonian (numpy tensors)
# Build some useful occupation numpy tensors (for Wegner's generator and flow equations)
# Iterate flow equation until convergence

# The functions -wegner- and -flow- build and evaluate graphs in TensorFlow. The inputs
# are numpy arrays which are dynamically converted into Tensor objects; the graphs
# are built from these objects. At the end of each function, we evaluate the graph,
# remember the results, and reset that session's graph (might not be necessary).
# --- Test timing of one flow iteration (for optimization testing)

# with tf.device('/cpu:0'):
#@profile
def run(n_holes):
    H1B_t, H2B_t, ref, holes, particles, B1 = build_hamiltonian(n_holes, n_holes)

    occA = get_occA(B1, ref)
    occB = get_occB(B1, ref)
    occC = get_occC(B1, ref)
    occD = get_occD(B1, ref)
    
    E, f, G = normal_order(H1B_t, H2B_t, holes)

    
    y0 = unravel(E, f, G)

    t = 1
    dy = derivative(t, y0, holes, particles, occA, occB, occC, occD)

    dE, df, dG = ravel(dy, holes, particles)
    print(dE)

# if __name__ == '__main__':
#     # import timeit
#     # import sys
#     # print(timeit.timeit("run(int(sys.argv[1]))", setup="from __main__ import run", number=10))
    
#     run(int(sys.argv[1]))
import pytest
def test02ph_run(benchmark):
    benchmark(lambda: run(2))
def test04ph_run(benchmark):
    benchmark(lambda: run(4))
def test06ph_run(benchmark):
    benchmark(lambda: run(6))
def test08ph_run(benchmark):
    benchmark(lambda: run(8))
def test10ph_run(benchmark):
    benchmark(lambda: run(10))
# %timeit run(E, f, G)


# --- Solve the IM-SRG flow
# y0 = unravel(E, f, G)

# solver = ode(derivative,jac=None)
# solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
# solver.set_f_params(holes, particles, occA, occB, occC, occD)
# solver.set_initial_value(y0, 0.)

# sfinal = 2
# ds = 0.1
# s_vals = []
# E_vals = []

# while solver.successful() and solver.t < sfinal:
    
#     ys = solver.integrate(sfinal, step=True)
#     Es, fs, Gs = ravel(ys) 

#     print("scale param: {:0.4f} \t E = {:0.5f}".format(solver.t, Es))
#     s_vals.append(solver.t)
#     E_vals.append(Es)

#open mp

