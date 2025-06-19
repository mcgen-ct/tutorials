* INTRO

We have worked hard to make a good generator for uniformly-distributed random variates.
In practice, however, the probability distributions of interest are not uniform.
Fortunately, uniform random variates can either be transformed into a different distribution
or used as part of an accept/reject algorithm that converges to the desired probability distribution.

Random variates -- uniform or not -- are also a primary part of the Monte Carlo integration method,
so it is worthwhile to know how to transform uniform into complicated.

Note, in our presentation, we only consider _continuous_ distributions, but everything that we say
can be applied, with some modification, to discrete distributions.   

* INVERSE CDF METHOD

In many cases, we have a complicated, but analytic, expression for our probability distribution.
If that expression can be integrated in closed form, then we might be able to apply the inverse
CDF (cumulative distribution function) method.   The i-CDF method uses the fact that the integral
of the p.d.f. varies continuously from 0 to 1 as a function of the upper limit.   The integral of
the p.d.f. (== c.d.f.) can be compared to a uniform random variate and solved (we hope) for the
upper limit.

Example: lambda exp(-lambda x) ==> x = log(1/r)/lambda

Problem:   sample from the Breit-Wigner resonance (same as a Cauchy distribution)

Note:  even if the inverse of the integral does not have a closed form, the i-CDF method
can still be used in a brute-force way, i.e. solve CDF(x) - r = 0.  Though less elegant, this
method might be faster than other alternatives.

* ACCEPT/REJECT

This method should almost always work, though it might not be efficient.   
It requires having an estimate of the max(p.d.f.), but that can also be found by trial and
error, if needed.   In the simplest application, the variable of interest is sampled uniformly
using a uniform random variate.    Next, for that value, one compares p.d.f.(x) / max(x) to
another random variate, accepting if ratio < r and rejecting and repeating the process otherwise.

-- Majorizing

-- Squeezing

* INTEGRATION

Fundamental Theorem of Calculus

Simple (flat) sampling.

Jacobian to reduce variance.

Comparison to Trapezoidal Rule or Gaussian quadrature.

Multidimensional for hard test integrand.

* INTRO TO VEGAS

Vegas by hand.

Vegas python.
