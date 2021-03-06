# This is the setup file of compos codes#
# Run it with 'sudo python setup.py install'#
# The code should be re-installed after any change#

from setuptools import setup, find_packages

packages = find_packages()
setup(
    name="compos",
    version="1.1",
    packages=packages,
    install_requires=['numpy', 'scipy', 'matplotlib', 'astropy', 'compos'],

    #    author = "Ziang Yan",
    #    author_email = "yanza.cosmo@gmail.com",
    #    description = "Codes for Matter Power Spectrum",
    #    keywords = ("astronomy cosmology cosmological matter power spectrum"),
    #    license = "THCA","Department of Physics and Astronomy, UBC"
    #    long_description = \
    # """
    # Compos contains five parts:
    #
    #    cosmology: class including cosmological parameters
    #    transfunction: class calculating transfer functions at
    #    given cosmological parameters.
    #    matterps:class calculating linear matter power spectrum
    #    and two point correlation function.
    #    halofit: class calculating non-linear matter power spectrum
    #
    # Jupyter notebooks are provided in /examples folder as examples.
    # """
)
