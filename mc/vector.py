###########################################################################


class Vector:
    """
    The 'Vector' class represents a vector of length 4.
    """

    def __str__(self):
        """
        Return a string to print of this vector.
        """
        return "%r %r %r %r" % tuple(self.v)

    def __repr__(self):
        """
        Return the representation of this vector.
        """
        return "%s(%r, %r, %r, %r)" % tuple([self.__class__.__name__] + self.v)

    def __init__(self, v0, v1, v2, v3):
        """
        Initialise the class with its four components.
        """
        self.v = [v0, v1, v2, v3]

    def __getitem__(self, i):
        """
        Return the 'i' component of the vector. Note, this can be a
        slice.
        """
        return self.v[i]

    def __setitem__(self, i, s):
        """
        Set the 'i' component of the vector with the scalar
        's'. Alternatively, 'i' can be a slice and 's' a sub-vector.
        """
        self.v[i] = s

    def __len__(self):
        """
        Return the length of the vector, e.g. 4.
        """
        return 4

    def __pos__(self):
        """
        Return a copy of this vector with the '+' operator applied to
        each element.
        """
        return self.__class__(+self[0], +self[1], +self[2], +self[3])

    def __neg__(self):
        """
        Return a copy of this vector with the '-' operator applied to
        each element.
        """
        return self.__class__(-self[0], -self[1], -self[2], -self[3])

    def __iadd__(self, v):
        """
        Augmented assignment '+=' for adding a vector 'v' to this vector.
        """
        for i in range(0, 4):
            self[i] += v[i]
        return self

    def __isub__(self, v):
        """
        Augmented assignment '-=' for subtracting a vector 'v' from
        this vector.
        """
        for i in range(0, 4):
            self[i] -= v[i]
        return self

    def __add__(self, v):
        """
        Return the addition of this vector with the vector 'v'.
        """
        u = +self
        u += v
        return u

    def __sub__(self, v):
        """
        Return the subtraction of this vector with the vector 'v'.
        """
        u = +self
        u -= v
        return u

    def __invert__(self):
        """
        Return the complex transpose of this vector.
        """
        v = +self
        for i in range(0, 4):
            try:
                v[i] = self[i].conjugate()
            except:
                v[i] = self[i]
        return v

    def __imul__(self, x):
        """
        Augmented assignment '*=' for multiplying this vector with a
        vector, matrix, or scalar 'x'.
        """
        # The vector case.
        try:
            x[0]
            self = sum(self[i] * x[i] for i in range(0, 4))
        except:
            # The matrix case.
            try:
                x[0, 0]
                u = +self
                for i in range(0, 4):
                    self[i] = sum([u[j] * x[j, i] for j in range(0, 4)])
            # The scalar case.
            except:
                for i in range(0, 4):
                    self[i] *= x
        return self

    def __mul__(self, x):
        """
        Return the multiplication of this vector with a vector, matrix, or
        scalar 'x'.
        """
        u = +self
        u *= x
        return u

    def __rmul__(self, x):
        """
        Return the multiplication of a vector, matrix, or scalar 'x'
        with this vector. The operation x*v where x is a vector or
        matrix is not used.
        """
        return self * x

    def __itruediv__(self, s):
        """
        Augmented assignment '/=' for dividing this vector with a
        scalar 's'.
        """
        for i in range(0, 4):
            self[i] /= s
        return self

    def __truediv__(self, s):
        """
        Return the division of this vector by a scalar 's'. The
        reflected operator, 's/v', cannot be implemented since this is
        not a defined mathematical operation.
        """
        u = +self
        u /= s
        return u

    def __ipow__(self, i):
        """
        Augmented assignment '**=' for raising this vector to the
        integer power 'i'. For even 'i' this is a scalar and odd 'i' a
        vector.
        """
        if i < 0:
            raise ValueError("power must be positive")
        u = (~self) * self
        if i == 2:
            self = u
        # The even case.
        elif i % 2 == 0:
            self = u ** int(i / 2)
        # The odd case.
        elif i % 2 == 1:
            self *= u ** int((i - 1) / 2)
        return self

    def __pow__(self, i):
        """
        Return this vector raised to the integer power 'i'. For even
        'i' this is a scalar and odd 'i' a vector.
        """
        u = +self
        u **= i
        return u

    def __abs__(self):
        """
        Return the norm of the vector.
        """
        from math import sqrt

        return sqrt(self**2)


###########################################################################


class Matrix:
    """
    The 'Matrix' class represents a matrix of size 4x4.
    """

    def __str__(self):
        """
        Return a string to print of this vector.
        """
        return "\n".join("%10r %10r %10r %10r" % tuple(mi) for mi in self.m)

    def __repr__(self):
        """
        Return the representation of this vector.
        """
        return "%s(m0 = %r, m1 = %r, m2 = %r, m3 = %r)" % tuple(
            [self.__class__.__name__] + self.m
        )

    def __init__(self, m0, m1, m2, m3):
        """
        Initialise the matrix with the row vectors 'm0', 'm1', 'm2',
        and 'm3'.
        """
        self.m = [[mij for mij in mi] for mi in [m0, m1, m2, m3]]

    def __getitem__(self, k):
        """
        Return the 'ij' component of the matrix.
        """
        i, j = k
        return self.m[i][j]

    def __setitem__(self, k, s):
        """
        Set the 'ij' component of the matrix with the scalar 's'.
        """
        i, j = k
        self.m[i][j] = s

    def __pos__(self):
        """
        Return a copy of this matrix with the '+' operator applied to
        each element.
        """
        return self.__class__(m0=self.m[0], m1=self.m[1], m2=self.m[2], m3=self.m[3])

    def __neg__(self):
        """
        Return a copy of this matrix with the '-' operator applied to
        each element.
        """
        m = +self
        for i in range(0, 4):
            for j in range(0, 4):
                m[i, j] = -m[i, j]
        return m

    def __iadd__(self, m):
        """
        Augmented assignment '+=' for adding a matrix 'm' to this matrix.
        """
        for i in range(0, 4):
            for j in range(0, 4):
                self[i, j] += m[i, j]
        return self

    def __isub__(self, m):
        """
        Augmented assignment '-=' for subtracting a matrix 'm' from
        this matrix.
        """
        for i in range(0, 4):
            for j in range(0, 4):
                self[i, j] -= m[i, j]
        return self

    def __add__(self, m):
        """
        Return the addition of this matrix with the matrix 'm'.
        """
        l = +self
        l += m
        return l

    def __sub__(self, m):
        """
        Return the subtraction of this matrix with the matrix 'm'.
        """
        l = +self
        l -= m
        return l

    def __invert__(self):
        """
        Return the complex transpose of this matrix.
        """
        m = +self
        for i in range(0, 4):
            for j in range(0, 4):
                try:
                    m[j, i] = self[i, j].conjugate()
                except:
                    m[j, i] = self[i, j]
        return m

    def __imul__(self, x):
        """
        Augmented assignment '*=' for multiplying this matrix with a
        vector, matrix, or scalar 'x'.
        """
        # The vector case.
        try:
            x[0]
            v = +x
            for i in range(0, 4):
                v[i] = sum([self[i, j] * x[j] for j in range(0, 4)])
            self = v
        except:
            # The matrix case.
            try:
                x[0, 0]
                m = +self
                for i in range(0, 4):
                    for j in range(0, 4):
                        self[i, j] = sum([m[i, k] * x[k, j] for k in range(0, 4)])
            # The scalar case.
            except:
                for i in range(0, 4):
                    for j in range(0, 4):
                        self[i, j] *= x
        return self

    def __mul__(self, x):
        """
        Return the multiplication of this matrix with either a
        matrix or a scalar.
        """
        l = +self
        l *= x
        return l

    def __rmul__(self, x):
        """
        Return the multiplication of a vector, matrix, or scalar 'x'
        with this matrix. The operation x*m where x is a vector or
        matrix is not used.
        """
        return self * x

    def __itruediv__(self, s):
        """
        Augmented assignment '/=' for dividing this matrix with a
        scalar 's'.
        """
        for i in range(0, 4):
            for j in range(0, 4):
                self[i, j] /= s
        return self

    def __truediv__(self, s):
        """
        Return the division of this matrix by a scalar 's'. The reflected
        operator, 's/m', requires the inverse of the matrix and is not
        implemented.
        """
        l = +self
        l /= s
        return l

    def __abs__(self):
        """
        Return the norm of the matrix.
        """
        from math import sqrt

        n = 0
        for i in range(0, 4):
            for j in range(0, 4):
                try:
                    n += self[i, j].conjugate() * self[i, j]
                except:
                    n += self[i, j] * self[i, j]
        try:
            return sqrt(n.real)
        except:
            return sqrt(n)

    def __ipow__(self, i):
        """
        Augmented assignment '**=' for raising this matrix to the
        integer power 'i'.
        """
        if i < 0:
            raise ValueError("power must be positive")
        # Special case for 0.
        if i == 0:
            for j in range(0, 4):
                for k in range(0, 4):
                    self[j, k] = float(j == k)
        # When i > 1.
        else:
            u = +self
            for j in range(1, i):
                self *= u
        return self

    def __pow__(self, i):
        """
        Return this vector raised to the integer power 'i'. For even
        'i' this is a scalar and odd 'i' a vector.
        """
        u = +self
        u **= i
        return u


###########################################################################


class FourVector(Vector):
    """
    The 'FourVector' class represents a physics four-vector.
    """

    def __init__(self, *args):
        """
        Constructs the four-vector from its base Vector class.
        """
        Vector.__init__(self, *args)

    def __invert__(self):
        """
        Return the lowered index of this vector, e.g. p_mu.
        """
        v = +self
        for i in range(1, 4):
            v[i] = -v[i]
        return v


class BoostMatrix(Matrix):
    """
    The 'BoostMatrix' class represents a Lorentz boost matrix.
    """

    def __init__(self, p=None, mass=None, m0=None, m1=None, m2=None, m3=None):
        """
        Initialise the boost matrix given a momentum four-vector, 'p'. An
        additional 'mass' can be passed which can be used to stabilise
        large boosts. Finally, the 'mi' vectors can be passed so this
        can be constructed using the same initialisation as the
        'Matrix' class.
        """
        from math import sqrt

        if m0 and m1 and m2 and m3:
            Matrix.__init__(self, m0, m1, m2, m3)
        else:
            betas = [float(p[i]) / p[0] for i in range(1, 4)]
            if mass:
                gamma = p[0] / mass
            else:
                gamma = 1 / sqrt(1 - sum([b**2 for b in betas]))
            alpha = gamma**2 / (1 + gamma)
            m0 = [gamma, -gamma * betas[0], -gamma * betas[1], -gamma * betas[2]]
            m1 = [
                m0[1],
                1 + alpha * betas[0] ** 2,
                alpha * betas[0] * betas[1],
                alpha * betas[0] * betas[2],
            ]
            m2 = [m0[2], m1[2], 1 + alpha * betas[1] ** 2, alpha * betas[1] * betas[2]]
            m3 = [m0[3], m1[3], m2[3], 1 + alpha * betas[2] ** 2]
            self.m = [m0, m1, m2, m3]


###############################################################################

if __name__ == "__main__":

    def show(expr):
        """
        Print and evaluate an expression.
        """
        print("%s\n%10s:\n%s\n%s\n" % ("-" * 20, expr, "-" * 20, eval(expr)))

    def vector(p, idx):
        """
        Return a vector, given the mass and momentum, 'idx' specifies
        energy index.
        """
        from math import sqrt

        e = sqrt(sum([pi**2 for pi in p[0:3]]) + p[3] ** 2)
        if idx == 0:
            return (e, p[0], p[1], p[2])
        else:
            return (p[0], p[1], p[2], e)

    # Vector class.
    v0 = Vector(1, 2, 3, 4)
    v1 = Vector(1.0, 2j, 3, 0)
    s0 = 4.0
    show("v0")
    show("v1")
    show("s0")
    show("v0 + v1")
    show("v0 - v1")
    show("~v1")
    show("v0 * v0")
    show("v0**2")
    show("v0**3")
    show("v0/s0")

    # Matrix class.
    v2 = [0.5, 6.0, 7.0, 8.0]
    v3 = [1.0, -5.3, 6.0, 0.0]
    m0 = Matrix(v0, v1, v2, v3)
    m1 = Matrix(v2, v2, v1, v1)
    show("m0")
    show("m1")
    show("m0 + m1")
    show("m0 - m1")
    show("~m1")
    show("s0*m0")
    show("m0*v0")
    show("v0*m0")
    show("m0*m0")
    show("m0/s0")
    show("abs(m0)")
    show("m0**0")
    show("m0**2")

    # FourVector class.
    p0 = [1e3, 4e2, 6.8e9, 5.0]
    p1 = [0.3, 1.2, 1, 5.0]
    fv0 = FourVector(*vector(p0, 0))
    fv1 = FourVector(*vector(p1, 0))
    show("fv0")
    show("fv1")
    show("fv0*fv0")
    show("~fv0*fv0")
    show("abs(fv0)")

    # BoostMatrix class.
    bm0 = BoostMatrix(fv0)
    bm1 = BoostMatrix(fv0, p0[3])
    show("bm0")
    show("bm0*fv1")
    show("bm1*fv1")

    # Check against Pythia 8.
    try:
        from math import sqrt
        import pythia8

        v0 = pythia8.Vec4(*vector(p0, 3))
        v1 = pythia8.Vec4(*vector(p1, 3))
        v1.bstback(v0)
        v1 = pythia8.Vec4(*vector(p1, 3))
        v1.bstback(v0, p0[3])
        show(v1)
    except:
        pass

    # Example boost for the LHC.
    mp = 0.93827
    pp = 7000.0
    ep = (mp**2 + pp**2) ** 0.5
    fv0 = FourVector(ep, 0.0, 0.0, pp)
    fv1 = FourVector(ep, 0.0, 0.0, -pp)
    bm0 = BoostMatrix(fv0)
    show("abs(fv0 + fv1)")
    show("bm0*fv0")
    show("(bm0*fv1)")
    fv0p = bm0 * fv0
    fv1p = bm0 * fv1
    show("abs(fv0p + fv1p)")
