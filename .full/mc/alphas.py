import numpy as np


class SUN:
    def __init__(self, NC):
        self.NC = NC
        self.TR = 1.0 / 2.0
        self.CA = NC
        self.CF = (NC**2 - 1.0) / (2.0 * NC)


class AlphaS(SUN):
    def __init__(self, mZ, asmZ, nc=3, order=1, mb=4.75, mc=1.3):
        super().__init__(nc)
        self.order = order
        self.mc2 = mc**2
        self.mb2 = mb**2
        self.mZ2 = mZ**2
        self.asmZ = asmZ

        # Lambdas for beta functions
        self.beta0 = lambda nf: (11.0 / 6.0) * self.CA - (2.0 / 3.0) * self.TR * nf
        self.beta1 = (
            lambda nf: (17.0 / 6.0) * self.CA**2
            - ((5.0 / 3.0) * self.CA + self.CF) * self.TR * nf
        )

        # Set alphas at key scales
        self.asmb = self(self.mb2)
        self.asmc = self(self.mc2)
        print(f"\\alpha_s({mZ} GeV) = {self(self.mZ2)}")
        assert self(self.mZ2) == asmZ

    def as0(self, t):
        if t >= self.mb2:
            tref = self.mZ2
            asref = self.asmZ
            b0 = self.beta0(5) / (2.0 * np.pi)
        elif t >= self.mc2:
            tref = self.mb2
            asref = self.asmb
            b0 = self.beta0(4) / (2.0 * np.pi)
        else:
            tref = self.mc2
            asref = self.asmc
            b0 = self.beta0(3) / (2.0 * np.pi)
        return 1.0 / (1.0 / asref + b0 * np.log(t / tref))

    def as1(self, t):
        if t >= self.mb2:
            tref = self.mZ2
            asref = self.asmZ
            b0 = self.beta0(5) / (2.0 * np.pi)
            b1 = self.beta1(5) / (2.0 * np.pi) ** 2
        elif t >= self.mc2:
            tref = self.mb2
            asref = self.asmb
            b0 = self.beta0(4) / (2.0 * np.pi)
            b1 = self.beta1(4) / (2.0 * np.pi) ** 2
        else:
            tref = self.mc2
            asref = self.asmc
            b0 = self.beta0(3) / (2.0 * np.pi)
            b1 = self.beta1(3) / (2.0 * np.pi) ** 2
        w = 1.0 + b0 * asref * np.log(t / tref)
        return asref / w * (1.0 - b1 / b0 * asref * np.log(w) / w)

    def __call__(self, t):
        if self.order == 0:
            return self.as0(t)
        return self.as1(t)
