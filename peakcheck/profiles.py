"""
Line profiles: true Voigt (Faddeeva), Gaussian, Lorentzian.

All profiles are AREA-NORMALISED to 1, so the fitted amplitude is directly the
integrated intensity of the peak (cf. `fit_amplitudes`). The Voigt is the
exact Gaussian (x) Lorentzian convolution evaluated via the Faddeeva function
`scipy.special.wofz` — not a pseudo-Voigt.
"""
from __future__ import annotations

import numpy as np
from scipy.special import wofz



def voigt(x, center, sigma, gamma):
    r"""Area-normalised Voigt profile evaluated via the Faddeeva function.

    The Voigt profile is the *convolution* of a Gaussian (standard deviation
    ``sigma``) with a Lorentzian (half width at half maximum ``gamma``). It is
    evaluated exactly through the Faddeeva function :math:`w(z)`:

    .. math::

        V(x) = \frac{\operatorname{Re}[w(z)]}{\sigma\sqrt{2\pi}}, \qquad
        z = \frac{(x - x_0) + i\,\gamma}{\sigma\sqrt{2}}, \qquad
        w(z) = e^{-z^2}\operatorname{erfc}(-i z).

    This is **not** a pseudo-Voigt (a linear combination of a Gaussian and a
    Lorentzian); it is the true convolution and is identical to
    :func:`scipy.special.voigt_profile` to machine precision.

    Parameters
    ----------
    x : numpy.ndarray
        Positions at which to evaluate the profile.
    center : float
        Peak centre :math:`x_0`.
    sigma : float
        Gaussian standard deviation. If ``sigma <= 0`` a pure Lorentzian is
        returned (the exact ``sigma -> 0`` limit).
    gamma : float
        Lorentzian half width at half maximum. ``gamma -> 0`` gives a pure
        Gaussian (handled implicitly by ``w(z)``).

    Returns
    -------
    numpy.ndarray
        Profile values; the curve integrates to 1 over the real line.

    Notes
    -----
    The Faddeeva function is provided by :func:`scipy.special.wofz` [7]_.
    For the definition of :math:`w(z)` and its relation to the Voigt profile
    see Abramowitz & Stegun [2]_, Sec. 7.1.

    References
    ----------
    .. [2] M. Abramowitz, I. A. Stegun, "Handbook of Mathematical Functions",
       Dover (1972), Sec. 7.1.
    .. [7] P. Virtanen et al., "SciPy 1.0", Nature Methods 17 (2020) 261-272.
    """
    if sigma <= 0.0:
        return (gamma / np.pi) / ((x - center) ** 2 + gamma ** 2)
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def gaussian(x, center, sigma):
    r"""Area-normalised Gaussian profile.

    .. math:: G(x) = \frac{1}{\sigma\sqrt{2\pi}}
              \exp\!\left[-\frac{(x-x_0)^2}{2\sigma^2}\right]

    Parameters
    ----------
    x : numpy.ndarray
        Evaluation positions.
    center : float
        Peak centre :math:`x_0`.
    sigma : float
        Standard deviation (related to the FWHM by
        :math:`\mathrm{FWHM} = 2\sqrt{2\ln 2}\,\sigma`).

    Returns
    -------
    numpy.ndarray
        Profile values (integral 1).
    """
    return np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def lorentzian(x, center, gamma):
    r"""Area-normalised Lorentzian (Cauchy) profile.

    .. math:: L(x) = \frac{\gamma/\pi}{(x-x_0)^2 + \gamma^2}

    Parameters
    ----------
    x : numpy.ndarray
        Evaluation positions.
    center : float
        Peak centre :math:`x_0`.
    gamma : float
        Half width at half maximum (HWHM); the FWHM is :math:`2\gamma`.

    Returns
    -------
    numpy.ndarray
        Profile values (integral 1).
    """
    return (gamma / np.pi) / ((x - center) ** 2 + gamma ** 2)


def profile_shape(x, center, sigma, gamma, profile):
    """Single area-normalised peak of the chosen profile.
       'gauss' uses sigma (from wg), 'lorentz' uses gamma (from wl),
       'voigt' uses both. The unused width is ignored for the pure shapes."""
    p = str(profile).lower()
    if p in ("gauss", "gaussian"):
        return gaussian(x, center, sigma)
    if p in ("lorentz", "lorentzian"):
        return lorentzian(x, center, gamma)
    return voigt(x, center, sigma, gamma)
