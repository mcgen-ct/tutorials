import numpy as np
from vector import FourVector
from particle import Particle


class eeTojj:
    def __init__(self, alphas, pdb, ecms=91.2):
        """
        Initialize e+ e- -> jet jet matrix element generator.

        Parameters:
        alphas : AlphaS
            An instance of the AlphaS class for strong coupling calculations.
        pdb : ParticleDataBase
            A particle database instance (not used in this implementation).
        ecms : float
            Center-of-mass energy in GeV (default is 91.2 GeV).
        """
        self.alphas = alphas
        self.ecms = ecms
        self.MZ2 = pow(91.1876, 2.0)
        self.GZ2 = pow(2.4952, 2.0)
        self.alpha = 1.0 / 128.802
        self.sin2tw = 0.22293
        self.amin = 1.0e-10
        self.ye = 0.5
        self.ze = 0.01
        self.ws = 0.25
        self.pdb = pdb

    def ME2(self, fl, s, t):
        """
        Calculate the squared matrix element for e+ e- -> f fbar.
        """
        # Electron charge and couplings
        qe = -1.0  # Electric charge of electron
        ae = -0.5  # Axial-vector coupling of electron
        ve = ae - 2.0 * qe * self.sin2tw  # Vector coupling of electron

        # Final state fermion charge and couplings
        qf = (
            2.0 / 3.0 if fl in [2, 4] else -1.0 / 3.0
        )  # Electric charge of final state fermion
        af = (
            0.5 if fl in [2, 4] else -0.5
        )  # Axial-vector coupling of final state fermion
        vf = af - 2.0 * qf * self.sin2tw  # Vector coupling of final state fermion

        # Electroweak mixing factor (from weak mixing angle)
        kappa = 1.0 / (4.0 * self.sin2tw * (1.0 - self.sin2tw))

        # Z propagator factors (Breit-Wigner shape)
        chi1 = (
            kappa
            * s
            * (s - self.MZ2)
            / (np.power(s - self.MZ2, 2.0) + self.GZ2 * self.MZ2)
        )
        chi2 = np.power(kappa * s, 2.0) / (
            np.power(s - self.MZ2, 2.0) + self.GZ2 * self.MZ2
        )

        # Matrix element squared terms
        term1 = (1 + np.power(1.0 + 2.0 * t / s, 2.0)) * (
            np.power(qf * qe, 2.0)
            + 2.0 * (qf * qe * vf * ve) * chi1
            + (ae * ae + ve * ve) * (af * af + vf * vf) * chi2
        )
        term2 = (1.0 + 2.0 * t / s) * (
            4.0 * qe * qf * ae * af * chi1 + 8.0 * ae * ve * af * vf * chi2
        )
        return np.power(4.0 * np.pi * self.alpha, 2.0) * 3.0 * (term1 + term2)

    def generate_event(self):
        """
        Generate a random phase space point for e+ e- -> f fbar.

        Returns:
        tuple: A tuple containing:
            - list of Particle: List of generated particles (e+, e-, f, fbar).
            - float: Differential cross section in pb.
            - float: Squared matrix element value.
        """
        # Generate random cos(theta) and phi for outgoing fermion direction
        ct = 2.0 * np.random.random() - 1.0  # cos(theta) uniformly in [-1, 1]
        st = np.sqrt(1.0 - ct * ct)  # sin(theta) from cos(theta)
        phi = 2.0 * np.pi * np.random.random()  # phi uniformly in [0, 2pi)

        # Outgoing fermion 4-momentum in CM frame (energy, px, py, pz)
        p1 = FourVector(1, st * np.cos(phi), st * np.sin(phi), ct) * (self.ecms / 2)

        # Outgoing anti-fermion 4-momentum (back-to-back)
        p2 = FourVector(p1[0], -p1[1], -p1[2], -p1[3])

        # Incoming e+ and e- four-momenta in CM frame
        pa = FourVector(self.ecms / 2, 0, 0, self.ecms / 2)  # e-
        pb = FourVector(self.ecms / 2, 0, 0, -self.ecms / 2)  # e+

        # Randomly select final state fermion flavor (1-4)
        flav = np.random.randint(1, 5)

        # Mandelstam variables s and t for the process
        pab_plus_M2 = (
            (pa[0] + pb[0]) ** 2
            - (pa[1] + pb[1]) ** 2
            - (pa[2] + pb[2]) ** 2
            - (pa[3] + pb[3]) ** 2
        )
        pa1_minus_M2 = (
            (pa[0] - p1[0]) ** 2
            - (pa[1] - p1[1]) ** 2
            - (pa[2] - p1[2]) ** 2
            - (pa[3] - p1[3]) ** 2
        )

        # Compute squared matrix element for this flavor and kinematics
        loME = self.ME2(flav, pab_plus_M2, pa1_minus_M2)

        # Differential cross section (pb) for this phase space point
        dxs = 5.0 * loME * 3.89379656e8 / (8.0 * np.pi) / (2.0 * np.power(self.ecms, 2))

        # Return list of Particle objects (incoming and outgoing), cross section, and ME^2
        return (
            [
                Particle(self.pdb["e+"], -pa),  # e+ (incoming)
                Particle(self.pdb["e-"], -pb),  # e- (incoming)
                Particle(self.pdb[flav], p1, [1, 0]),  # outgoing fermion (color [1,0])
                Particle(self.pdb[-flav], p2, [0, 1]),
            ],  # outgoing anti-fermion (color [0,1])
            dxs,
            loME,
        )

    def generate_LO_event(self):
        lo = self.generate_event()
        return (lo[0], lo[1])


if __name__ == "__main__":
    from particle import ParticleDatabase
    from alphaS import AlphaS

    # Initialize particle database
    pdb = ParticleDatabase()
    print(pdb["e+"])
    print(pdb["e-"])

    # Initialize strong coupling constant
    alphas = AlphaS(91.1880, 0.1180)

    # Intialize the eeTojj generator
    hardxs = eeTojj(alphas, pdb)

    # Generate events
    nevents = 1
    events = []
    for i in range(nevents):
        ev = hardxs.generate_event()
        events.append(ev)
        print(f"Event {i+1}:")
        for particle in ev[0]:
            print(
                f"  Particle ID: {particle.data.pid}, Momentum: {particle.p}, Color: {particle.c if hasattr(particle, 'c') else 'N/A'}"
            )
        print(f"  Cross Section: {ev[1]} pb")
        print(f"  Matrix Element Squared: {ev[2]}")
        print()
