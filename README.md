# Where to get the data
Download available from `https://mitprod-my.sharepoint.com/:u:/g/personal/roche_mit_edu/EQZ9Y-_csntIkb-VO-PuZZQBP0xjH86xBLAJHxjsW3ZqOQ?e=iyZF15`
  
Comes in a "pickle" file format, readable by pandas via  
```
import pandas
dataframe = pandas.read_pickle("filename")
```

# Directory Structure
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

# The subfolders in this folder are as follows:

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
    with the following columns:  
    
    Basic:  
    - Source IDs ("source_id")

    Positions:  
    - RA ("ra") and DEC ("dec")
    - Positions in galactic coordinates ("l","b")
    - Positions of stars in galactocentric Cartesian coordinates ("x","y","z")
    - Distance from galactic center ("r") and distance from the Sun ("r_helio")
    - Errors on those positions ("x_err","y_err","z_err","r_err")
    - Parallax with an applied zero point correction ("parallax") 
        Calculations are done with this corrected value, introduces nans to many columns, strip with  
        dataframe = pandas.read_pickle("/DR3_6D_kinematics.pkl")
        dataframe = dataframe[numpy.isfinite(df["parallax"])] 
        Correction package here: https://pypi.org/project/gaiadr3-zeropoint/
    - Parallax without zero point correction ("parallax_nozpcorr")
    - Parallax over error ("plx_over_error")

    Velocities  
    - Proper motion in RA ("pmra") and DEC ("pmdec") and associated errors ("pmra_err", "pmdec_err")
    - Velocities of the stars in galactocentric cartesian coordinates ("vx","vy","vz") and their errors ("vx_err",...)
    - Velocities of the stars in galactocentric spherical coordinates ("vr","vtheta","vphi") and their errors ("vr_err",...)
    - Radial velocity ("v_radial") and associated error ("v_radial_error")
    - Speed ("vabs") and associated properly propagated error ("vabs_error")
    - Speed error without covariances taken into account for errors ("vabs_error_nocov")

    Misc / quality cuts  
    - Metallicity ("feH")
    - Number of transits ("rv_nb_transits")
    - Radial velocity signal to noise ("rv_expected_sig_to_noise")
    - RUWE ("ruwe") see for example https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_ruwe.html


3. "submit_converting_code.batch"  
    Parallel submission script for converting_code, to improve speed of conversion. Requires editing to be used on a different cluster and for different users


4. "DR3_6D_kinematics.pkl"  
    This is the (concatenated, see point 5.) output of "converting_code.py" which contains the columns described in point 2.


5. "concatenate_dataframes.py"  
    Script to concatenate the dataframe chunk outputs from converting_code. which is submitted to a job system via submit_concatenate_dataframes.batch


6. "submit_concatenate_dataframes.batch"  
    Submits concatenate_dataframes.py to a job scheduler. Requires editing to be used on a different cluster and for different users
