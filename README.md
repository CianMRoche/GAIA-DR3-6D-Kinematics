# GAIA DR3 6D Kinematics
This repo contains the code for a download and processing pipeline whose output is a file with columns containing the 6D kinematics of GAIA's data release 3 (DR3). The full sample of stars with radial velocities is roughly 33 million stars, but some columns are included for making quality cuts which reduce the sample to something closer to 20 million stars. By default, the only quality cut which is pre-applied is that the (uncorrected) parallax over parallax error is greater than 5.

### Where to get the processed data
Download available from [this link](https://mitprod-my.sharepoint.com/:u:/g/personal/roche_mit_edu/EQZ9Y-_csntIkb-VO-PuZZQBP0xjH86xBLAJHxjsW3ZqOQ?e=iyZF15)
  
Comes in a "pickle" file format, readable by pandas via  
```
import pandas
dataframe = pandas.read_pickle("filename")
```

### What information is available?
In the file linked above, the following columns can be obtained via a line of python like `dataframe["v_radial"].values`. Note that a column label such as "Q_err" or "Q_error" corresponds to the standard deviation of quantity Q. 

**Basic**:  
| Column Name | Description | Units |
| --- | --- | --- |
| "source_id" | Source IDs | n/a |
| "feH" | Metallicity | dex[^1] |

[^1]: Unitless, logarithm base 10

**Positions**:  
| Column Name | Description | Units |
| --- | --- | --- |
| "ra", "dec" | Right ascension, Declination | deg |
| "l", "b" | Star positions in galactic coordinates | deg |
| "x", "y", "z" | Star positions in galactocentric Cartesian coordinates | kpc |
| "x_err", "y_err", "z_err" | Errors on the above positions | kpc |
| "r" | Distance from galactic center | kpc |
| "r_helio" | Distance from Sun | kpc |
| "parallax_nozpcorr" | Parallax without zero point correction | mas |
| "parallax" | Parallax with an applied zero point correction[^2] | mas |
| "plx_over_error" | Parallax (corrected) over error | deg |

[^2]: Calculations are done with this corrected value, introduces nans to many columns, strip with  
    `dataframe = pandas.read_pickle("/DR3_6D_kinematics.pkl")`  
    `dataframe = dataframe[numpy.isfinite(df["parallax"])]`  
    Correction package here: https://pypi.org/project/gaiadr3-zeropoint/

**Velocities**  
| Column Name | Description | Units |
| --- | --- | --- |
| "pmra", "pmdec" | Proper motion in RA and DEC | mas yr$^{-1}$ |
| "pmra_err", "pmdec_err" | Errors on proper motions in RA and DEC | mas yr$^{-1}$ |
| "vx","vy","vz" | Velocities of the stars in galactocentric cartesian coordinates | km s$^{-1}$ |
| "vx_err", "vy_err", "vz_err" | Errors on the above velocities | km s$^{-1}$ |
| "vr","vtheta","vphi" | Velocities of the stars in galactocentric spherical coordinates | km s$^{-1}$ |
| "vr_err", "vtheta_err", "vphi_err" | Errors on the above velocities | km s$^{-1}$ |
| "v_radial" | Radial velocity as observed by GAIA | km s$^{-1}$ |
| "v_radial_error" | Associated error | km s$^{-1}$ |
| "vabs" | Speed of star | km s$^{-1}$ |
| "vabs_error" | Speed errors | km s$^{-1}$ |
| "vabs_error_nocov" | Speed error without covariances taken into account | km s$^{-1}$ |


**quality cuts**  
| Column Name | Description | Units |
| --- | --- | --- |
| "rv_nb_transits" | Number of transits | n/a |
| "rv_expected_sig_to_noise" | Radial velocity expected signal to noise | n/a |
| "ruwe" | Renormalised Unit Weight Error[^3] | km s$^{-1}$ |

[^3]: See for example the DR2 ruwe info [here](https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_ruwe.html).

### Directory Structure
Not all these directories are present in the repo, but would be necessary to recreate entirely the data pipeline. They are listed here for completeness.
```
.
└── DR3_downloading/
    └── download_DR3_6D_kinematics.py
    └── submit_DR3_6D_kinematics_download.batch
└── DR3_pieces/
    └── (untracked)
└── pickle_pieces/
    └── (untracked)
└── submitted_job_outputs/
    └── (untracked)
└── concatenate_csvs.py
└── concatenate_dataframes.py
└── converting_code.py
└── DR3_6D_kinematics_raw.csv (untracked)
└── DR3_6D_kinematics.pkl (most important data, untracked)
└── submit_concatenate_dataframes.batch
└── submit_converting_code.batch
└── README.md
```

### The subfolders in this folder are as follows:

1. "DR3_downloading"  
    Script to download 6D kinematic data from GAIA DR3 and a parallel submission script so it doesnt hit download limits

2. "DR3_pieces"  
    DR3 data is downloaded in chunks, via 
    "DR3_downloading/submit_DR3_6D_kinematics_download.batch"
    and the resulting chunks are saved here. Chunks can be concatenated and saved by concatenate_csvs.py

3. "pickle_pieces"  
    Outputs of converting_code.py which is run on chunks of the raw GAIA data.
    Can be concatenated by concatenate_csvs.py (see below)

4. "submitted_job_outputs"  
    Output files from the parallel data conversion from submit_converting_code.batch and submit_concatenate_dataframes.batch
  
  
# The files in this folder are as follows:   

1. "DR3_6D_kinematics_raw.csv"   
    This is velocity information from GAIA DR3 downloaded via the following ADQL script:  
  
    SELECT  
        source_id, ra, dec, l,b, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, 
        parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr, ruwe, radial_velocity, 
        radial_velocity_error, rv_template_fe_h, parallax_over_error, phot_g_mean_mag,
        nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved, 
        rv_nb_transits, rv_expected_sig_to_noise  
    FROM gaiadr3.gaia_source  
    WHERE rv_nb_transits > 0 AND parallax/parallax_error > 5.0  
  
2. "converting_code.py"  
    This code reads raw data from "DR3_6D_kinematics.csv" and creates a pandas "pickle" file
    with the columns outined above.  
    
    
3. "submit_converting_code.batch"  
    Parallel submission script for converting_code, to improve speed of conversion. Requires editing to be used on a different cluster and for different users


4. "DR3_6D_kinematics.pkl"  
    This is the (concatenated, see point 5.) output of "converting_code.py" which contains the columns described in point 2.


5. "concatenate_dataframes.py"  
    Script to concatenate the dataframe chunk outputs from converting_code. which is submitted to a job system via submit_concatenate_dataframes.batch


6. "submit_concatenate_dataframes.batch"  
    Submits concatenate_dataframes.py to a job scheduler. Requires editing to be used on a different cluster and for different users
