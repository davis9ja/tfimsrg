from hamiltonian import PairingHamiltonian
from occupation_tensors import OccupationTensors
from generator import WegnerGenerator
from flow import Flow_IMSRG2
from scipy.integrate import odeint, ode
import numpy as np
import time

ha = PairingHamiltonian(4,4)
ot = OccupationTensors(ha.sp_basis, ha.reference)
wg = WegnerGenerator(ha, ot)
fl = Flow_IMSRG2(ha, ot)

def derivative(t, y, hamiltonian, occ_tensors, generator, flow):
    assert isinstance(hamiltonian, PairingHamiltonian), "Arg 2 must be Hamiltonian object"
    assert isinstance(occ_tensors, OccupationTensors), "Arg 3 must be OccupationTensors object"
    assert isinstance(generator, WegnerGenerator), "Arg 4 must be WegnerGenerator object"
    assert isinstance(flow, Flow_IMSRG2), "Arg 5 must be a Flow_IMSRG2 object"

    E, f, G = ravel(y, hamiltonian.holes, hamiltonian.particles)

    generator.f = f
    generator.G = G
    eta1B = generator.calc_eta1B()
    eta2B = generator.calc_eta2B()
    
    dE, df, dG = flow.flow(generator, occ_tensors)
    
    dy = unravel(dE, df, dG)
    
    return dy

def unravel(E, f, G):
    unravel_E = np.reshape(E, -1)
    unravel_f = np.reshape(f, -1)
    unravel_G = np.reshape(G, -1)
    
    return np.concatenate([unravel_E, unravel_f, unravel_G], axis=0)

def ravel(y, holes, particles):
    
    bas_len = len(np.append(holes,particles))
    
    ravel_E = np.reshape(y[0], ())
    ravel_f = np.reshape(y[1:bas_len**2+1], (bas_len, bas_len))
    ravel_G = np.reshape(y[bas_len**2+1:bas_len**2+1+bas_len**4], 
                         (bas_len, bas_len, bas_len, bas_len))
    
    return(ravel_E, ravel_f, ravel_G)


def main():

    print("""Pairing model IM-SRG flow: 
    d              = {:2d}
    g              = {:2.4f}
    pb             = {:2.4f}
    SP basis size  = {:2d}
    n_holes        = {:2d}
    n_particles    = {:2d}""".format(1, 0.5, 0.0, 8, 4,4) )
    
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
        Es, fs, Gs = ravel(ys, ha.holes, ha.particles) 
        s_vals.append(solver.t)
        E_vals.append(Es)

        iters += 1

#         if iters %10 == 0: print("iter: {:>6d} \t scale param: {:0.4f} \t E = {:0.9f}".format(iters, solver.t, Es))

        if len(E_vals) > 20 and abs(E_vals[-1] - E_vals[-2]) < 10**-8:
            print("---- Energy converged at iter {:>06d} with energy {:1.8f}\n".format(iters,E_vals[-1]))
            convergence = 1
            break

        if len(E_vals) > 20 and abs(E_vals[-1] - E_vals[-2]) > 1:
            print("---- Energy diverged at iter {:>06d} with energy {:3.8f}\n".format(iters,E_vals[-1]))
            break
    
    return (convergence, iters, ha.d, ha.G, ha.pb, ha.n_sp_states, s_vals, E_vals)


if __name__ == '__main__':
    # start = time.time()
    # main()
    # end = time.time()
    # print(end-start)

    start = time.time()    
    dE, df, dG = fl.flow(wg, ot)
    end = time.time()
    print(dG[0,1,4,5])
    print(end-start)