from __future__ import division
import numpy as np
import scipy.integrate as integ
from scipy.integrate import odeint
from scipy.optimize import least_squares
import sys
sys.setrecursionlimit(10000)


class cosmology():
    '''
    Class: main constant or parameters
    for cosmology and an initialize function.
    Inputs:
        om0: total matter density;
        omb: baryon density;
        H0: current Hubble constant (in km/s/Mpc);
        T_CMB: CMB temperature in K;
        omq: dark energy density;
        omnu: neutrino density, not used in current version;
        w_0, w_1: DE equation of state;
        n_s: spectral index of primordial spectrum;
        sigma8: sigma_8 given to normlize the spectrum;
    '''

    def __init__(self, om0=0.307, omb=0.0486, H0=67.74, T_CMB=2.725, omq=0.691,
                 omnu=0, w_0=-1, w_1=0, n_s=0.9667, sigma8=0.8159,):
        self.omega_0 = om0
        self.omega_b = omb
        self.omega_c = om0 - omb
        self.omega_q = omq
        self.omega_nu = omnu
        self.omega_k = 1 - omq - om0 - omnu
        self.h = H0 / 100.
        self.omega_0hh = om0 * self.h ** 2
        self.omega_chh = self.omega_c * self.h ** 2
        self.omega_bhh = omb * self.h ** 2
        self.omega_qhh = omq * self.h ** 2
        self.T_CMB = T_CMB
        self.Theta_CMB = T_CMB / 2.7
        self.w_0 = w_0
        self.w_1 = w_1
        self.n_s = n_s
        self.sigma8 = sigma8


class transfunction():

    '''
    Class: calculate transfer function.
    Input: compos.cosmology object.
    '''
    def __init__(self, cosmo=cosmology()):

        self.omega_0 = cosmo.omega_0
        self.omega_b = cosmo.omega_b
        self.omega_c = cosmo.omega_c
        self.omega_q = cosmo.omega_q
        self.omega_nu = cosmo.omega_nu
        self.omega_k = cosmo.omega_k
        self.h = cosmo.h
        self.omega_0hh = cosmo.omega_0hh
        self.omega_chh = cosmo.omega_chh
        self.omega_bhh = cosmo.omega_bhh
        self.omega_qhh = cosmo.omega_qhh
        self.T_CMB = cosmo.T_CMB
        self.Theta_CMB = cosmo.Theta_CMB
        self.w_0 = cosmo.w_0
        self.w_1 = cosmo.w_1
        self.n_s = cosmo.n_s
        self.sigma8 = cosmo.sigma8

        self.z_eq = 2.5 * 10 ** 4 * self.omega_0 * self.h ** \
                    2 * self.Theta_CMB ** (-4)

        self.k_eq = 7.46 * 10 ** (- 2) * self.omega_0 * self.h \
                    ** 2 * self.Theta_CMB ** (-2)

        self.b1 = 0.313 * (self.omega_0hh) ** (- 0.419) \
             * (1 + 0.607 * (self.omega_0hh) ** 0.674)
        self.b2 = 0.238 * (self.omega_0hh) ** 0.223
        self.z_d = 1291 * ((self.omega_0hh) ** 0.251 /
                      (1 + 0.659 * (self.omega_0hh) ** 0.828))\
                      * (1 + self.b1 * (self.omega_bhh) ** self.b2)

        self.R_eq = 31.5 * self.omega_bhh \
            * self.Theta_CMB ** (-4) * (self.z_eq/1000.) ** (-1)
        self.R_d = 31.5 * self.omega_bhh \
            * self.Theta_CMB ** (-4) * (self.z_d/1000.) ** (-1)

        self.s = (2. / (3. * self.k_eq)) * np.sqrt(6. / self.R_eq) * \
            np.log((np.sqrt(1 + self.R_d) + np.sqrt(self.R_d + self.R_eq)) \
                   / (1 + np.sqrt(self.R_eq)))

        self.k_silk = 1.6 * (self.omega_bhh) ** \
            0.52 * (self.omega_0hh) ** \
            0.73 * (1 + (10.4 * self.omega_0 *
                         self.h ** 2) ** (-0.95))


    def j_0(self, x):

        '''
        Function: calculating the zeroth order Bessel function.
        '''
        j = np.sin(x) / x
        return j

    # calculate T_tilde

    def Ttilde(self, k, alpha, beta):

        '''
        Function: Calculate Ttilde
        Input: k (in Mpc^-1), alpha and beta are two auxiliary parameters
        Reference: Eq(19, 20) in arxiv:9709112v1
        '''
        q = k / (13.41 * self.k_eq)
        C = 14.2 / alpha + 386. / (1 + 69.9 * q ** 1.08)
        T_tilde = np.log(np.e + 1.8 * beta * q) / (np.log
                                               (np.e + 1.8 * beta * q) +
                                               C * q ** 2)
        return T_tilde


    # calculate the transfer function for cdm.#

    def Tcdm(self, k):

        '''
        Function: Calculating transfer function for CDM.
        Parameters: k (in Mpc^-1)
        Reference: Eq(9-12, 17, 18) in arxiv:9709112v1
        '''

        s = self.s
        a_1 = (46.9 * self.omega_0hh)\
              ** 0.670 * (1 + (32.1 * self.omega_0hh) **
              (-0.532))
        a_2 = (12.0 * self.omega_0hh) ** 0.424 * \
              (1 + (45.0 * self.omega_0 * self.h**2) ** (-0.582))
        b_1 = 0.944 * (1 + (458 * self.omega_0hh) **
                       (-0.708)) ** (-1)
        b_2 = (0.395 * self.omega_0hh) ** (-0.0266)
        alpha_c = a_1 ** (-self.omega_b / self.omega_0) * a_2 ** \
                  (- (self.omega_b / self.omega_0) ** 3)
        beta_c = 1 / (1 + b_1 * ((self.omega_c / self.omega_0) ** b_2 - 1))
        f = 1 / (1 + (k * s / 5.4) ** 4)
        T_c = f * self.Ttilde(k, 1, beta_c) + (1 - f) \
              * self.Ttilde(k, alpha_c, beta_c)
        return T_c

    # calculate the transfer function of baryon#

    def Tbaryon(self, k):

        '''
        Function: Calculating transfer function for baryon.
        Parameters: k (in Mpc^-1)
        Reference: Eq(13-15, 21-24) in arxiv:9709112v1
        '''

        z_eq = self.z_eq
        z_d = self.z_d

        R_d = self.R_d
        k_eq = self.k_eq
        k_silk = self.k_silk
        s = self.s

        def Gfunc(y):
            G = y * (-6 * np.sqrt(1 + y) + (2 + 3 * y) * np.log(
                (np.sqrt(1 + y) + 1)/(np.sqrt(1 + y) - 1)))
            return G
        alpha_b = 2.07 * k_eq * s * (1 + R_d) ** (-0.75) * \
            Gfunc((1 + z_eq) / (1 + z_d))
        beta_b = 0.5 + self.omega_b / self.omega_0 + \
            (3 - 2 * self.omega_b / self.omega_0) * \
            np.sqrt((17.2 * self.omega_0hh) ** 2 + 1)
        beta_node = 8.41 * (self.omega_0hh) ** 0.435
        s_tilde = s / ((1 + (beta_node / (k * s)) ** 3) ** (1/3))
        T_b = (self.Ttilde(k, 1, 1) / (1 + (k * s / 5.2) ** 2) +
               (alpha_b / (1 + (beta_b / (k * s)) ** 3)) * np.e **
               (-(k / k_silk) ** 1.4)) * self.j_0(k * s_tilde)
        return T_b

    # calculate the transfer function#

    def transfunction(self, k):

        '''
        Function: Calculating total transfer function (CDM + Baryon).
        Parameters: k (in Mpc^-1)
        Reference: Eq(16) in arxiv:9709112v1
        '''

        T = (self.omega_b / self.omega_0) * self.Tbaryon(k) + \
            (self.omega_c / self.omega_0) * self.Tcdm(k)
        return T

    # calculate k of the first peak#

    def kpeak(self):

        '''
        Function: Calculating the scale of the first peak.
        Reference: Eq(25, 26) in arxiv:9709112v1
        '''

        s = 44.5 * np.log(9.83 / (self.omega_0hh)) /\
            np.sqrt(1 + 10 * (self.omega_bhh))
        k_peak = 5 * np.pi / (2 * s) * (1 + 0.217 * self.omega_0 *
                                        self.h ** 2)
        return k_peak


    # calculate the effective shape of transferfunction with zero-baryon case#

    def efshape(self, k):

        '''
        Function: Calculating effective shape parameter of transfer function zero-baryon.
        Parameters: k (in Mpc^-1)
        Reference: Eq(28, 29) in arxiv:9709112v1
        '''

        Gamma = self.omega_0 * self.h
        q = k / (self.h) * self.Theta_CMB ** 2 / Gamma
        L_0 = np.log(2 * np.e + 1.8 * q)
        C_0 = 14.2 + 731 / (1 + 62.5 * q)
        T_0 = L_0 / (L_0 + C_0 * q ** 2)
        return T_0


    # calculate the non-oscillatory part of the transfer function#


    def noosc(self, k):

        '''
        Function:  the non-oscillatory part of the transfer function.
        Parameters: k (in Mpc^-1)
        Reference: Eq(30, 31) in arxiv:9709112v1
        '''

        alpha_g = 1 - 0.328 * np.log(431 * self.omega_0hh) * \
                  self.omega_b/self.omega_0 + 0.38 * \
                  np.log(22.3 * self.omega_0hh) *\
                  (self.omega_b / self.omega_0) ** 2
        s = self.s
        Gamma_eff = self.omega_0 * self.h * (alpha_g + (1 - alpha_g) /
                                                 (1 + (0.43 * k * s) ** 4))
        return Gamma_eff

    # calculate shape fit with no wiggle#

    def t_nowiggle(self, k):

        '''
        Function:  the no-wiggle transfer function.
        Parameters: k (in Mpc^-1)
        Reference: Eq(28-31) in arxiv:9709112v1
        '''

        alpha_g = 1 - 0.328 * np.log(431 * self.omega_0hh) * \
                  self.omega_b/self.omega_0 + 0.38 * \
              np.log(22.3 * self.omega_0hh) \
              * (self.omega_b / self.omega_0) ** 2
        s = self.s
        Gamma_eff = self.omega_0 * self.h * \
                    (alpha_g + (1 - alpha_g) / (1 + (0.43 * k * s) ** 4))
        q_eff = k / (self.h) * self.Theta_CMB ** 2 / Gamma_eff
        L_0 = np.log(2 * np.e + 1.8 * q_eff)
        C_0 = 14.2 + 731 / (1 + 62.5 * q_eff)
        tnowiggle = L_0 / (L_0 + C_0 * q_eff ** 2)
        return tnowiggle


class matterps(transfunction):
    '''
    Class: caculate the linear matter power spectrum at given redshift.
    Input: compos.cosmology object.
    '''
    def __init__(self, cosmology):

        transfunction.__init__(self, cosmology)
        self.c = 2.998 * 10 ** 5  # the velocity of light

    def omega(self, z):

        '''
        Function: calculating matter density ad redshift z
        Reference: Eq(A5) in arxiv:9709112v1
        '''
        h = self.h
        om0 = self.omega_0
        Lambda = 1 - om0
        H0 = h * 100
        oml = Lambda / (1 * H0 ** 2)  # omega_Lambda
        omr = 1 - om0 - oml
        omega_z = om0 * (1 + z) ** 3 / (oml + omr * (1 + z) **
                                    2 + om0 * (1 + z) ** 3)
        return omega_z

    def omegaL(self, z):

        '''
        Function: calculating Lambda(as dark energy) density ad redshift z
        Reference: Eq(A6) in arxiv:9709112v1
        '''
        h = self.h
        om0 = self.omega_0
        Lambda = 1 - om0
        H0 = h * 100
        oml = Lambda / (1 * H0 ** 2)  # omega_Lambda
        omr = 1 - om0 - oml

        omega_L = oml / (oml + omr * (1 + z) ** 2 + om0 * (1 + z) ** 3)
        return omega_L

    def delta2(self, k):

        '''
        Function: calculating matter density
        Input: k (in unit of Mpc^-1)
        Reference: Eq(A1) in arxiv:9709112v1
        '''

        h = self.h
        om0 = self.omega_0
        ns = self.n_s
        Lambda = 1 - om0
        H0 = h * 100
        n_tilde = ns - 1
        if (Lambda == 0):
            delta_h = 1.95 * 10 ** (-5) * om0 ** (-0.35 - 0.19 * np.log(om0) -
                                                  0.17 * n_tilde) * np.e **\
                                                  (- n_tilde - 0.14 * n_tilde ** 2)
        else:
            delta_h = 1.94 * 10 ** (-5) * om0 ** \
                      (-0.785 - 0.05 * np.log(om0)) *\
                      np.e ** (- 0.95 * n_tilde - 0.169 * n_tilde ** 2)
        t = self.transfunction(k)

        delta = delta_h ** 2 * (self.c * k / H0 / 1000.) ** (3 + ns) * t ** 2
        return delta

    def matterps(self, k):

        '''
        Function: calculating matter power spectrum at z=0
        Input: k (in unit of Mpc^-1)
        Output: matter power spectrum P(k) in Mpc^-3
        Reference: Eq(A1) in arxiv:9709112v1
        '''

        T2 = self.transfunction(k)
        p = T2 ** 2 * k ** self.n_s

        return p

    def sigma8_calc(self):

        '''
        Function: calculating sigma8 from matter power spectrum
        '''

        s8 = self.sigma8
        def j1(z):
                j = (z * np.cos(z) - np.sin(z)) / (z ** 2)
                return j

        def Ws(k):
            R8 = 8. / self.h
            x = k * R8
            return 3*j1(x)/x

        def func(k):
            return 1./2./np.pi**2*Ws(k)**2*k**2*self.matterps(k)

        sigma2 = integ.quad(func, 0, np.inf, epsabs=1e-12)
        self.s8_calc = np.sqrt(sigma2[0])
        return self.s8_calc

    def z2a(self, z):

        '''
        Function: calculate the scale factor a at given redshift. a = 1 at z = 0
        Input: redshift
        '''

        return 1 / (1 + z)

    def a2z(self, a):

        '''
        Function: calculating the redshift z at given scale factor a
        Input: scale factor; must be smaller than 1
        '''

        return 1 / a - 1


    def w_z(self, z):

        '''
        Function: calculate the Dark Energy equation of state index w at redshift z.
        Input: redshift z
        '''

        w_0 = self.w_0
        w_1 = self.w_1
        return w_0 + w_1 * z / (1 + z)

    def w_a(self, a):

        '''
        Function: calculate the Dark Energy equation of state index w at scale factor a.
        Input: scale factor; must be smaller than 1
        '''

        z = self.a2z(a)
        return self.w_z(z)

    def exponent(self, z):
        def func(x):
            return (1 + self.w_z(x)) / (1 + x)
        return integ.quad(func, 0, z)[0]

    def hubble_z(self, z):

        '''
        Function: Calculate the Hubble constant at redshift z
        Input: redshift z; cosmology 'cosmo'
        Reference: Eq(1) in arxiv: astro-ph/0208087
        '''

        om0 = self.omega_0
        omq = self.omega_q
        omk = 1 - om0 - omq
        h = self.h
        H0 = h * 100
        H = H0 * np.sqrt((1 + z) ** 3 * om0 - (1 + z) **
                         2 * omk + np.exp(3 * self.exponent(z)) * omq)
        return H

    def om_0_z(self, z):
        om0 = self.omega_0
        h = self.h
        H0 = h * 100
        return om0 * (1 + z) ** 3 * (H0 / self.hubble_z(z)) ** 2

    def om_q_z(self, z):

        omq = self.omega_q
        h = self.h
        H0 = h * 100
        return omq * np.exp(3 * self.exponent(z)) * (H0 / self.hubble_z(z)) ** 2

    def growfunc_a(self, a):

        '''
        Function: calculate the growth function at scale factor a.
        Input: scale factor a (must be smaller than 1).
        Reference: (2) and (3) in arxiv: astro-ph/0208087.
        '''
        def ffunc(y, la):
            x = self.a2z(np.exp(la))
            f = y[0]
            return [- f ** 2 - (1 - self.om_0_z(x) / 2 -
                                (1 + 3 * self.w_z(x)) / 2 * self.om_q_z(x)) *
                    f + 3 / 2 * self.om_0_z(x), f]

        la = np.log(a)
        t = [np.log(0.0001), la]
        f0 = 1/(0.0001-1)+1
        ini = [f0, 1]
        d = np.exp(odeint(ffunc, ini, t)[1][1])
        return d

    def growfunc_z(self, z):

        '''
        Function: calculate the growth function at redshift z.
        Input: redshift
        Reference: (2) and (3) in arxiv: astro-ph/0208087.
        '''

        return (self.growfunc_a(self.z2a(z)) / self.growfunc_a(1))

    def normalizedmp(self, k, z=0):

        '''
        Function: calculating normalized matter power spectrum at redshift z.
        Input: k (in unit of Mpc^-1)
        Output: matter power spectrum P(k) in Mpc^-3
        '''

        s8 = self.sigma8_calc()
        normalp = self.matterps(k) * (self.sigma8 / s8) ** 2
        if z == 0:
            return normalp
        else:
            return normalp * (self.growfunc_z(z) /
                              self.growfunc_z(0)) ** 2

    def delta2nw(self, k):

        '''
        Function: calculating matter density without wiggle
        Input: k (in unit of Mpc^-1)
        Reference: Eq(A1) in arxiv:9709112v1
        '''

        h = self.h
        om0 = self.omega_0
        Lambda = 1 - om0
        H0 = h * 100
        oml = Lambda / (1 * H0 ** 2)  # omega_Lambda
        # power index,set to 1 for Harrison-Zel'dovich-Peebles scale invariant case
        n = self.n_s
        n_tilde = n - 1
        if (Lambda == 0):
            delta_h = 1.95 * 10 ** (-5) * om0 ** \
                      (-0.35 - 0.19 * np.log(om0) -
                       0.17 * n_tilde) * np.e ** (- n_tilde - 0.14 * n_tilde ** 2)
        else:
            delta_h = 1.94 * 10 ** (-5) * om0 **\
                      (-0.785 - 0.05 * np.log(om0)) * \
                      np.e ** (- 0.95 * n_tilde - 0.169 * n_tilde ** 2)
        t = self.t_nowiggle(k)
        delta = delta_h ** 2 * (self.c * k / H0 / 1000) ** (3 + n) * t ** 2
        return delta

    def matterpsnw(self, k):

        '''
        Function: calculating matter power spectrum at z=0 without wiggle
        Input: k (in unit of Mpc^-1)
        Output: matter power spectrum P(k) in Mpc^-3
        Reference: Eq(A1) in arxiv:9709112v1
        '''

        d = self.delta2nw(k)
        p = d * 2 * np.pi ** 2 / (k ** 3)
        return p

    def normalizedmpnw(self, k, z):

        '''
        Function: calculating normalized matter power spectrum at redshift z without wiggle.
        Input: k (in unit of Mpc^-1)
        Output: matter power spectrum P(k) in Mpc^-3
        '''

        s8 = self.sigma8_calc()
        s8_given = self.sigma8
        normalp = self.matterpsnw(k) * (s8_given / s8) ** 2
        if z == 0:
            return normalp
        else:
            return normalp * (self.growfunc_z(z) /
                          self.growfunc_z(0)) ** 2

    def kstar(self):
        sum = integ.quad(self.normalizedmp, 0, 3000)[0]
        kstar = ((1 / (3 * np.pi ** 2)) * sum) ** (-0.5)
        return kstar

    def twopcf(self, x):

        '''
        Function: calculate the two-point correlation function
        Input: distance x in Mpc.
        '''

        def func(k):
            f = self.normalizedmp(k) / (2 * np.pi ** 2) * k * np.sin(k * x) / x
            return f

        xi = np.zeros(np.size(x))

        k = np.linspace(0.0001, 500, 100000)
        p = self.normalizedmp(k)
        k0 = k[0]
        dk = k[1]-k[0]
        for i in range(xi.size):
            xi[i] = np.sum(dk*k**2*p*np.sin(k*x[i])/(k*x[i]))/2./np.pi**2
        return xi


class halofit(matterps):
    '''
    Class: calculate non-linear matter power spectrum with HALOFIT.
    The formulas are in Appendix of 1208.2701.
    Input: compos.cosmology object.
    '''
    def __init__(self, cosmology):
        matterps.__init__(self, cosmology)

        self.s8_calc = self.sigma8_calc()

    def haloparams(self, z):
        '''
        Function: Calculate all the parameters needed.
        Input: Redshift.
        '''
        self.Z = z
        self.gf = self.growfunc_z(z) / self.growfunc_z(0)
        def Sigma2(R, d=self.gf):
            '''
            Function: Calculate sigma^2(R) defined in (A4)
            Input: Radius, growth factor
            '''
            nint = 3000
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0
            i = np.arange(nint)+1
            t = (i-0.5)/float(nint)
            k = -1+1.0 / t
            dt2 = self.matterps(k) * (self.sigma8/self.s8_calc)**2 * d ** 2 * k ** 3 / (2 * np.pi ** 2)
            x = k * R
            w1 = np.exp(- x * x)
            w2 = 2 * x * x * w1
            w3 = 4 * x * x * (1 - x * x) * w1
            sum1 = np.sum(w1*dt2/k/t/t) / float(nint)
            sum2 = np.sum(w2*dt2/k/t/t) / float(nint)
            sum3 = np.sum(w3*dt2/k/t/t) / float(nint)
            d1 = -sum2 / sum1
            d2 = -sum2 * sum2 / sum1 / sum1 - sum3 / sum1
            return np.array([sum1, d1, d2])

        def diff(x, d=1.):
            return Sigma2(x, d=d)[0] - 1.0

        def solveforr(d=1):
            '''
            Function: Solve for k_sigma^(-1) such that sigma^2(R) = 1.
            '''
            self.Rnl = least_squares(diff, 1, args=(d,), bounds=(0, 1000), xtol=1e-4).x
            r = Sigma2(self.Rnl, d=d)
            self.neff = -3-r[1]
            self.C = -r[2]

        solveforr(d=self.gf)
        w = self.w_z(z)
        om0 = self.om_0_z(z)
        omq = self.om_q_z(z)
        self.an = 10 ** (1.5222 + 2.8553 * self.neff + 2.3706 * self.neff ** 2 +
                    0.9903 * self.neff ** 3 + 0.225 * self.neff ** 4 -
                    0.6038 * self.C + 0.1749 * omq * (1 + w))
        self.bn = 10 ** (-0.5642 + 0.5864 * self.neff + 0.5716 *
                    self.neff ** 2 - 1.5474 * self.C + 0.2279 * omq * (1 + w))
        self.cn = 10 ** (0.3698 + 2.0404 * self.neff + 0.8161 * self.neff ** 2 + 0.5869 * self.C)
        self.gamman = 0.1971 - 0.0843 * self.neff + 0.8460 * self.C
        self.alphan = np.abs(6.0835 + 1.3373 * self.neff - 0.1959 * self.neff ** 2 - 5.5274 * self.C)
        self.betan = 2.0379 - 0.7354 * self.neff + 0.3157 * self.neff **\
                2 + 1.249 * self.neff ** 3 + 0.3980 * self.neff ** 4 - 0.1682 * self.C
        self.mun = 0
        self.nun = 10 ** (5.2105 + 3.6902 * self.neff)
        if(abs(1-om0) > 0.01):
            f1a = om0 ** (-0.0732)
            f2a = om0 ** (-0.1423)
            f3a = om0 ** (0.0725)
            f1b = om0 ** (-0.0307)
            f2b = om0 ** (-0.0585)
            f3b = om0 ** (0.0743)
            frac = omq/(1.0-om0)
            self.f1 = frac * f1b + (1 - frac) * f1a
            self.f2 = frac * f2b + (1 - frac) * f2a
            self.f3 = frac * f3b + (1 - frac) * f3a
        else:
            self.f1 = 1.0
            self.f2 = 1.0
            self.f3 = 1.0
        return

    def deltaq(self, k):
        '''
        Function: Calculate two halo terms(A2)
        '''
        y = k * self.Rnl

        def funy(x):
            return x / 4 + x ** 2 / 8
        deltal = self.normalizedmp(k, z=0) * self.gf ** 2 * k ** 3 / (2 * np.pi ** 2)
        deltaq = deltal * ((1 + deltal) ** self.betan /
                           (1 + self.alphan * deltal)) * np.exp(- funy(y))
        return deltaq

    def deltah(self, k):
        '''
        Function: Calculate one halo terms(A3)
        '''

        y = k * self.Rnl
        deltaprime = self.an * y ** (3 * self.f1) / \
                     (1 + self.bn * y ** self.f2 + (self.cn * self.f3 * y) ** \
                      (3 - self.gamman))
        deltah = deltaprime / (1 + self.mun * y ** (-1) + self.nun * y ** (-2))
        return deltah

    def nldelta(self, k, z=0):
        '''
        Function: Calculate total perturbation Delta(k, z) (A1).
        Input: wavenumber k in Mpc^-1, redshift.
        '''
        self.haloparams(z)
        return self.deltaq(k) + self.deltah(k)

    def nlpowerspec(self, k, z=0):  # Spectrum of nonlinear perturbation
        '''
        Function: Calculate total nonlinear power spectrum P(k, z).
        Input: wavenumber k in Mpc^-1, redshift.
        '''
        return self.nldelta(k, z) * k ** (-3) * 2 * np.pi ** 2
