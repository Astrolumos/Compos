This is a instruction document of COMPOS (COdes for Matter POwer Spectrum) written by Ziang Yan. It is written in Python.

This is the version August 2019.

Prerequisites:
=============
        
        Python
        Python packages: setuptools, numpy, matplotlib, scipy (use sudo apt-get install python-matplotlib to get matplotlib and numpy; $sudo apt-get install python-scipy to get scipy and $sudo apt-get install python-setuptools to get setuptools package)
        CAMB (if you need to call a camb routine):http://camb.info/

Installing:
===========
        
           Install: $ sudo python setup.py install
           or
           $ python setup.py install --user

Running:
========
        
        Before calling any function in the COMPOS package, you need to initialize a cosmology objoect by calling:
        cosmo = compos.cosmology(om0, omb, H, T_CMB, omq, omnu = 0, omk = 0, w_0 = -1, w_1 = 0, n_s = 1, sigma8 = 0.8 , z = 0, kmax = 20, nonlinear = 0, callcamb = 0)
        The definition and default value of those parameters are (from Planck 2013):
            om0 :    0.316,     #Total density of mass (scaled with critical density)
            omb :    0.049, 	#Density of baryon
            omq :    0.6825,    #Density of dark energy
            omnu :   0, 	#Density of neutrino
            omk :    0,         #Density of curvature
            H :      0.6711, 	#Hubble constant
            T_CMB :  2.728, 	#Temperature of CMB. Normalized to 2.7K
            w_0 :    -1, 	#DE state eq wrt z as w_0+w_1 * z / (1 + z)             
            w_1 :    0, 
            sigma8 : 0.8344, 	#Total fluctuation amplitude within 8 Mpc h^(-1)
            n_s :    0.96, 	#Power index
       
COMPOS package contains three classes:
==================================           

           transfunction: calculating transfer functions at given cosmological parameters.
           matterps:codes calculating linear matter power spectrum and two point correlation function and growthfactor.

           halofit:python version for HALOFIT (see arxiv:1208.2701) calculating nonlinear power spectrum.

           Jupyter notebook examples are provided under ./examples
         
