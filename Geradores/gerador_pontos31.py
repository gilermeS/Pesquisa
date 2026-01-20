import numpy as np
import random



omega_m = 0.315
sig_om = 0.007

h0 = 67.45
sig_h0 = 0.62



def foo(center, sig, n=3):
	return random.uniform(center- n*sig, center+ n*sig)

def friedmann(z, h, om):
	return h*np.sqrt(om*(1+z)**3. + (1.-om))



zmin = .1
zmax = 1.5

nz = 31
sig = 0.008



for i in range(1000):

	z_arr=zmin+(zmax-zmin)*np.arange(nz)/(nz-1.0)

	h0_arr = []
	om_arr = []

	# H(z) values according to the fiducial Cosmology at each z assuming a Gaussian distribution centred on hz_model

	# temp_h0 = foo(h0, sig_h0)
	temp_h0 = random.uniform(65, 80)
	temp_om = foo(omega_m, sig_om)
	

	hz_arr = np.array([friedmann(z,temp_h0,temp_om) for z in z_arr])

	h0_arr.append(temp_h0)
	om_arr.append(temp_om)
	

	sighz_arr = hz_arr*sig


	# saving the simulated hz results in a np file
	
	filename = 'input31/'+(f'data_{i+1}')
	print(f'Status: {((i+1)/10000 * 100):.2f} %', end='\r')
	

	np.save(filename, np.transpose([z_arr, hz_arr, nz*h0_arr, nz*om_arr]))
	


print(f'Status: 100.00 %')


data = np.array([
    [0.07,   69.0],
    [0.09,   69.0],
    [0.12,   68.6],
    [0.17,   83.0],
    [0.1791, 75.0],
    [0.1993, 75.0],
    [0.20,   72.9],
    [0.27,   77.0],
    [0.28,   88.8],
    [0.3519, 83.0],
    [0.3802, 83.0],
    [0.40,   95.0],
    [0.4004, 77.0],
    [0.4247, 87.1],
    [0.44497,92.8],
    [0.47,   89.0],
    [0.4783, 80.9],
    [0.48,   97.0],
    [0.5929,104.0],
    [0.6797, 92.0],
    [0.7812,105.0],
    [0.8754,125.0],
    [0.88,   90.0],
    [0.90,  117.0],
    [1.037, 154.0],
    [1.30,  168.0],
    [1.363, 160.0],
    [1.43,  177.0],
    [1.53,  140.0],
    [1.75,  202.0],
    [1.965, 186.5],
])

file = 'input31/data_real31'
np.save(file, data)

























