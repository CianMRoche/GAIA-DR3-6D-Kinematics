#####################################################
# Based on code from Lina Necib 2018 
# Parallelized via submit_coordinate_convert.batch
# Executes 1 million rows at a time by default
#
# Updated by Marianne Moore, Cian Roche (2022)
# Email: roche@mit.edu
#####################################################

import numpy as np
from tqdm import tqdm
import os
import sys
from scipy.special import erf
from zero_point import zpt
import pandas as pd
from astropy.io import fits
import astropy.coordinates as coord
import astropy.units as u
from uncertainties import unumpy


#################################
## UPDATE DIRECTORY NAMES HERE ##
#################################
input_dir = './'
input_file = 'DR3_6D_kinematics_raw.csv'

output_dir = './pickle_pieces/'
output_file = 'DR3_6D_kinematics_'+sys.argv[1]+'.pkl' # to parallelise the job

print('input_dir = %s' % input_dir)
print('output_dir = %s' % output_dir)
print('outfile = %s' % output_file)

############################################

print()
print('Loading data...')
######################
# CHECK COLUMN NAMES #
######################
start_row = int(sys.argv[1]) * 1000000 
nrows = 1000000
print('Rows '+str(start_row) +' until '+ str(start_row + nrows))


desired_columns = ['source_id', 'ra', 'dec', 'pmra', 'pmra_error', 'parallax', 'parallax_error', 'l', 'b', 'pmdec',
                    'pmdec_error', 'rv_template_fe_h', 'pmra_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr', 
                    'radial_velocity', 'radial_velocity_error', 'parallax_over_error','astrometric_params_solved',
                    'phot_g_mean_mag','nu_eff_used_in_astrometry', 'pseudocolour', 'ecl_lat', 'ruwe', 'rv_nb_transits',
                    'rv_expected_sig_to_noise']

if int(sys.argv[1]) == 0: # zeroth one requires slightly different format
    dataset = pd.read_csv(input_dir + input_file, nrows=nrows, usecols = desired_columns)
                        
else:
    dataset = pd.read_csv(input_dir + input_file, skiprows=range(1,start_row), nrows=nrows, usecols = desired_columns)


print('%i rows' % len(dataset))

print()
print('Cleaning data...')
dataset.replace('NOT_AVAILABLE', np.nan, inplace=True)
broken = np.where(dataset['pmra_error'].values < 0)[0]

print('%i rows' % len(dataset))
dataset.drop(broken, inplace=True)
print('%i rows' % len(dataset))

############################################

print()
print('Organizing data...')

# SOURCE IDS
source_id_list = np.array(dataset['source_id'], dtype=str)

# POSITIONS

# RA DEC
ra = dataset['ra'].values
dec = dataset['dec'].values

# GALACTIC COORDINATES
l = dataset['l'].values
b = dataset['b'].values

# DISTANCE
parallax_nozpcorr = dataset['parallax'].values
parallax_err = dataset['parallax_error'].values

ruwe = dataset['ruwe'].values

# zero point correction
zpt.load_tables()

try:
    zpvals = zpt.get_zpt(dataset['phot_g_mean_mag'].values, dataset['nu_eff_used_in_astrometry'].values, dataset['pseudocolour'].values, dataset['ecl_lat'].values, dataset['astrometric_params_solved'].values,_warnings=False)
except:
    mask = np.logical_and((dataset['astrometric_params_solved'].values != 31),(dataset['astrometric_params_solved'].values != 95)) 
    dataset['astrometric_params_solved'].values[mask] = 31
    logger.info(f'Encountered unrecognized soltype! {np.sum(mask)} stars affected!')
    zpvals = zpt.get_zpt(dataset['phot_g_mean_mag'].values, dataset['nu_eff_used_in_astrometry'].values, dataset['pseudocolour'].values, dataset['ecl_lat'].values, dataset['astrometric_params_solved'].values,_warnings=False)
    zpvals[mask] = np.nan

parallax = parallax_nozpcorr - zpvals
# end zero point correction calculation

# Fix uncorrected parallax
parallax_nozpcorr_cut = np.isfinite(parallax_nozpcorr) & (parallax_nozpcorr > 0) # Remove invalid parallaxes
parallax_nozpcorr[~parallax_nozpcorr_cut] = np.nan  # tilde is logical inversion

# Fix corrected parallax
parallax_cut = np.isfinite(parallax) & (parallax > 0) # Remove invalid parallaxes, cut based on CORRECTED parallax (which has some extra NaNs)
parallax[~parallax_cut] = np.nan
distance = 1.0 / parallax

distance_err = abs(parallax_err / parallax**2)
distance_all = unumpy.uarray(distance, distance_err)

# PROPER MOTIONS (AND ERRORS)
pmra = dataset['pmra'].values
pmdec = dataset['pmdec'].values
pm_ra_errors = dataset['pmra_error'].values
pm_dec_errors = dataset['pmdec_error'].values
pmra_pmdec_corr = dataset['pmra_pmdec_corr'].values
correlation_pmra_parallax = dataset['parallax_pmra_corr'].values
correlation_pmdec_parallax = dataset['parallax_pmdec_corr'].values
pmra_all = unumpy.uarray(pmra, pm_ra_errors)
pmdec_all = unumpy.uarray(pmdec, pm_dec_errors)

# RADIAL VELOCITY
vrad = np.array(dataset['radial_velocity'].values, dtype=float)
vrad_errors = np.array(dataset['radial_velocity_error'].values)
vr_all = unumpy.uarray(vrad, vrad_errors)

# METALLICITY
feH = dataset['rv_template_fe_h'].values


############################################

print()
print('Converting coordinates...')

# Sun

# 2018
# rSun = 8.0  # kpc
# x_shift = rSun
# y_shift = 0.  # kpc
# z_shift = 0.015  # kpc

# astropy 4.0 values
rSun =-8.122  # kpc
x_shift = rSun
y_shift = 0.  # kpc
z_shift = 0.0208  # kpc
print('Using solar parameters rsun = %.1f, x_shift = %.1f, y_shift = %.1f, z_shift = %.1f' % (rSun, x_shift, y_shift, z_shift))

# Converting...
x_list_error = distance_all * np.cos(np.radians(b)) * np.cos(np.radians(l)) + x_shift
y_list_error = distance_all * np.cos(np.radians(b)) * np.sin(np.radians(l)) + y_shift
z_list_error = distance_all * np.sin(np.radians(b)) + z_shift

z_list_values = unumpy.nominal_values(z_list_error)
z_list_std = unumpy.std_devs(z_list_error)

x = unumpy.nominal_values(x_list_error)
y = unumpy.nominal_values(y_list_error)
z = unumpy.nominal_values(z_list_error)

x_err = unumpy.std_devs(x_list_error)
y_err = unumpy.std_devs(y_list_error)
z_err = unumpy.std_devs(z_list_error)

radial_dis = np.linalg.norm([x, y, z], axis=0)
radial_err = (np.abs(x * x_err) + np.abs(y * y_err) + np.abs(z * z_err)) / radial_dis

############################################

# vx, vy, vz
print()
print('Calculating vx, vy, vz...')

# CHECK NUMBERS
# factor = 4.74  # *1e-3 factor off here
factor = 4.74047
print('Using factor = %.5f' % factor)


d_pm_ra = distance_all * pmra_all * factor
d_pm_de = distance_all * pmdec_all * factor
print(d_pm_ra[0:5])

# Convert ra, dec to radians
ra = np.radians(ra)
dec = np.radians(dec)

a_matrix = np.zeros((len(ra), 3, 3))
a1_matrix = np.zeros((len(ra), 3, 3))
a2_matrix = np.zeros((len(ra), 3, 3))

for i in tqdm(range(len(a1_matrix))):
    a1_matrix[i] = [[np.cos(ra[i]), -np.sin(ra[i]), 0], [np.sin(ra[i]), np.cos(ra[i]), 0], [0, 0, 1]]
    a2_matrix[i] = [[np.cos(dec[i]), 0, -np.sin(dec[i])], [0, 1, 0], [np.sin(dec[i]), 0, np.cos(dec[i])]]
    a_matrix[i] = np.dot(a1_matrix[i], a2_matrix[i])


# CHECK NUMBERS
theta = np.radians(123)
delta = np.radians(27.4)
alpha = np.radians(192.25)
print('Using theta = %.0f, detla = %.1f, alpha = %.2f' % (theta, delta, alpha))

t1_matrix = [[np.cos(theta), np.sin(theta), 0], [np.sin(theta), -np.cos(theta), 0], [0, 0, 1]]
t2_matrix = [[-np.sin(delta), 0, np.cos(delta)], [0, 1, 0], [np.cos(delta), 0, np.sin(delta)]]
t3_matrix = [[np.cos(alpha), np.sin(alpha), 0], [-np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]]

t_matrix = np.dot(np.dot(t1_matrix, t2_matrix), t3_matrix)

vx = np.zeros(len(dec))
vy = np.zeros(len(dec))
vz = np.zeros(len(dec))

vx_error = np.zeros(len(dec))
vy_error = np.zeros(len(dec))
vz_error = np.zeros(len(dec))

covariance_xyz = np.zeros((len(dec), 3, 3))

for i in tqdm(range(len(dec))):
    matrix = np.dot(t_matrix, a_matrix[i])
    matrix_UVA = np.dot(matrix, [vr_all[i], d_pm_ra[i], d_pm_de[i]])
    vx[i] = matrix_UVA[0].nominal_value
    vy[i] = matrix_UVA[1].nominal_value
    vz[i] = matrix_UVA[2].nominal_value

    # Getting the correlation.... which is a nightmare
    pmra_val = pmra[i]
    pmra_s = pm_ra_errors[i]

    pmde_val = pmdec[i]
    pmde_s = pm_dec_errors[i]

    dis = distance[i]
    dis_s = distance_err[i]

    corr_mu_mu = pmra_pmdec_corr[i] * pmra_s * pmde_s

    # if parallax:
    corr_d_alpha = correlation_pmra_parallax[i] * dis_s * pmra_s
    corr_d_delta = correlation_pmdec_parallax[i] * dis_s * pmde_s
    # else:
    #     corr_d_alpha = 0
    #     corr_d_delta = 0

    vr_s = vrad_errors[i]

    # Now let's multiply the matrices for the error propagation
    covariance_d_alpha_delta = np.array([[dis_s**2, corr_d_alpha, corr_d_delta], [corr_d_alpha, pmra_s**2, corr_mu_mu], [corr_d_delta, corr_mu_mu, pmde_s**2]])
    bmatrix = factor * np.array([[pmra_val, dis, 0], [pmde_val, 0, dis]])

    covariance_valpha_vdelta = np.dot(np.dot(bmatrix, covariance_d_alpha_delta), np.transpose(bmatrix))

    covariance_vr_valpha_vdelta = np.array([[vr_s**2, 0, 0], [0, covariance_valpha_vdelta[0][0], covariance_valpha_vdelta[0][1]], [0, covariance_valpha_vdelta[1][0], covariance_valpha_vdelta[1][1]]])

    covariance_xyz[i] = np.dot(np.dot(matrix, covariance_vr_valpha_vdelta), np.transpose(matrix))


for i in tqdm(range(len(covariance_xyz))):
    if covariance_xyz[i][0][0] >= 0:
        vx_error[i] = np.sqrt(covariance_xyz[i][0][0])
    else:
        vx_error[i] = float('NaN')

    if covariance_xyz[i][1][1] >= 0:
        vy_error[i] = np.sqrt(covariance_xyz[i][1][1])
    else:
        vy_error[i] = float('NaN')

    if covariance_xyz[i][2][2] >= 0:
        vz_error[i] = np.sqrt(covariance_xyz[i][2][2])
    else:
        vz_error[i] = float('NaN')


# solar_velocities = np.array([11.1, 239.08, 7.25]) # 2018
solar_velocities = np.array([12.9, 245.6, 7.78])  # astropy 4.0
print('Using solar velocities %.2f, %.2f, %.2f' % (solar_velocities[0], solar_velocities[1], solar_velocities[2]))

vx_all = unumpy.uarray(vx, vx_error)
vy_all = unumpy.uarray(vy, vy_error)
vz_all = unumpy.uarray(vz, vz_error)

vx_shifted = vx_all + solar_velocities[0]
vy_shifted = vy_all + solar_velocities[1]
vz_shifted = vz_all + solar_velocities[2]

# absolute magnitude of velocity and its error
vabs = np.zeros_like(vx)
vabs_err = np.zeros_like(vx)
vabs_err_nocov = np.zeros_like(vx)

# using full xyz covariance matrix, compute vabs and error on vabs.
for i in tqdm(range(len(dec))):
    vmean = np.array([vx[i] + solar_velocities[0], vy[i] + solar_velocities[1], vz[i] + solar_velocities[2]])
    Nvmean = np.linalg.norm(vmean)
    dv = np.array([vmean[0], vmean[1], vmean[2]]) / Nvmean
    vabs[i] = Nvmean
    vabs_err[i] = np.sqrt(np.dot(dv, np.dot(covariance_xyz[i], dv)))
    vabs_err_nocov[i] = np.sqrt(dv[0]**2 * vx_error[i]**2 +
                                dv[1]**2 * vy_error[i]**2 + dv[2]**2 * vz_error[i]**2)

print('Max absolute velocity = %.1f km/s' % (np.max(vabs)))

############################################

print()
print('Calculating v_r, v_theta, v_phi...')

r = radial_dis
phi = np.arctan2(y, x)
theta = np.arccos(z / r)

r_helio = np.linalg.norm(np.transpose([x + rSun, y, z]), axis=1)
print('r_helio min, max = %.1f, %.1f kpc' % (np.min(r_helio), np.max(r_helio)))

# Velocities
vr = np.array(vx_shifted * np.cos(phi) * np.sin(theta) + vy_shifted * np.sin(phi) * np.sin(theta) + vz_shifted * np.cos(theta))
vphi = np.array(-vx_shifted * np.sin(phi) + vy_shifted * np.cos(phi))
vtheta = np.array(vx_shifted * np.cos(phi) * np.cos(theta) + vy_shifted * np.sin(phi) * np.cos(theta) - vz_shifted * np.sin(theta))

vr_nom = unumpy.nominal_values(vr)
vtheta_nom = unumpy.nominal_values(vtheta)
vphi_nom = unumpy.nominal_values(vphi)

vr_std = unumpy.std_devs(vr)
vtheta_std = unumpy.std_devs(vtheta)
vphi_std = unumpy.std_devs(vphi)

############################################

print()
print('Saving data...')

dictionary = {}

# Basic Information and angular positions
dictionary["source_id"] = source_id_list  # .asstr()
dictionary["ra"] = dataset['ra'].values
dictionary["dec"] = dataset['dec'].values
dictionary["l"] = dataset['l'].values
dictionary["b"] = dataset['b'].values

# Parallaxes and distances
dictionary["parallax"] = parallax
dictionary["parallax_nozpcorr"] = parallax_nozpcorr
dictionary["plx_over_error"] = np.divide(parallax,parallax_err)

dictionary["r"] = radial_dis
dictionary["r_err"] = radial_err
dictionary["r_helio"] = r_helio

# Positions in Cartesian galactocentric coordinates (right-handed, solar position (-8.122,0,0.0308))
dictionary["x"] = x
dictionary["y"] = y
dictionary["z"] = z
dictionary["x_err"] = x_err
dictionary["y_err"] = y_err
dictionary["z_err"] = z_list_std

# Derived Velocities
dictionary["v_radial"] = vrad
dictionary["v_radial_error"] = vrad_errors

dictionary["pmra"] = pmra
dictionary["pmdec"] = pmdec
dictionary["pmra_err"] = pm_ra_errors
dictionary["pmdec_err"] = pm_dec_errors

# Velocities in Cartesian galactocentric coords
dictionary["vx"] = unumpy.nominal_values(vx_shifted) 
dictionary["vy"] = unumpy.nominal_values(vy_shifted) 
dictionary["vz"] = unumpy.nominal_values(vz_shifted)                                       
dictionary["vx_err"] = vx_error
dictionary["vy_err"] = vy_error
dictionary["vz_err"] = vz_error

# Velocities in Spherical galactocentric coords
dictionary["vr"] = vr_nom
dictionary["vtheta"] = vtheta_nom
dictionary["vphi"] = vphi_nom
dictionary["vr_err"] = vr_std
dictionary["vtheta_err"] = vtheta_std
dictionary["vphi_err"] = vphi_std

# Speed and associated errors
dictionary["vabs"] = vabs
dictionary["vabs_err"] = vabs_err
dictionary["vabs_err_nocov"] = vabs_err_nocov

# Quality cuts and metallicity
dictionary["ruwe"] = ruwe
dictionary["rv_nb_transits"] = dataset['rv_nb_transits'].values
dictionary["rv_expected_sig_to_noise"] = dataset['rv_expected_sig_to_noise'].values
dictionary["feH"] = feH

############################################

filename = output_dir + output_file
print()
print('Writing data to %s...' % filename)

# Construct a pandas dataframe
df = pd.DataFrame(data=dictionary)
df.to_pickle(filename)


print()
print('Done!')

