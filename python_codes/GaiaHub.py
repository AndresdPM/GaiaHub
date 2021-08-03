#!/usr/bin/env python

from __future__ import print_function

import sys
import os

import numpy as np
import pandas as pd

import argparse

import GaiaHubmod as gh

import warnings
warnings.filterwarnings("ignore")


def gaiahub(argv):  
   """
   Inputs
   """
   
   examples = '''Examples:
   
   gaiahub --name "Sculptor dSph"

   gaiahub --name "NGC 5053" --use_members --use_sat --quiet
   
   gaiahub --ra 201.405 --dec -47.667 --search_radius 0.1 --hst_filters "F814W" "F606W" --use_members --use_sat --preselect_cmd

   '''

   parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, usage='%(prog)s [options]', description='GaiaHub computes proper motions (PM) combining HST and Gaia data.', epilog=examples)
   
   # Search options
   parser.add_argument('--name', type=str, default = 'Output', help='Name for the Output table.')
   parser.add_argument('--ra', type=float, default = None, help='Central R.A.')
   parser.add_argument('--dec', type=float, default = None, help='Central Dec.')
   parser.add_argument('--search_radius', type=float, default = None, help='Radius of search in degrees.')
   parser.add_argument('--search_width', type=float, default = None, help='Width of the search rectangule in degrees.')
   parser.add_argument('--search_height', type=float, default = None, help='Height of the search rectangule in degrees.')
   parser.add_argument('--min_gmag', type=float, default = 16.0, help='Brighter G magnitude')
   parser.add_argument('--max_gmag', type=float, default = 21.5, help='Fainter G magnitude')

   # Gaia options
   parser.add_argument('--source_table', type = str, default = 'gaiaedr3.gaia_source', help='Gaia source table. Default is gaiaedr3.gaia_source.')
   parser.add_argument('--save_individual_queries', action='store_true', help='If True, the code will save the individual queries.')
   parser.add_argument('--sigma_flux_excess_factor', type=float, default=3., help='Sigma used for clipping in flux_excess_factor. Default is 3.')
   parser.add_argument('--only_5p_solutions', action='store_true', help='If True, only 5p solution stars will be considered as "good" sources.')
   parser.add_argument('--date_second_epoch', type=int, nargs='+', default= [5, 28, 2017], help='Second epoch adquisition date. Default is Gaia EDR3 (05-28-2017).')
   parser.add_argument('--date_reference_second_epoch', type=str, default= 'J2016.0', help='Second epoch reference date. Default is Gaia EDR3 J2016.0.')

   # HST options
   parser.add_argument('--hst1pass', action='store_true', help='Force the program to perform the search for sources in the HST images. Default is False, which will use existing files if any.')
   parser.add_argument('--fmin', type=int, default= None, help='Minimum flux above the sky to substract a source in the HST image. Default is automatic, computed from the HST integration time.')
   parser.add_argument('--pixel_scale', type=float, default= None, help='Pixel scale in arcsec/pixel used to compute the tangential plane during the match between epochs. Default is automatic, and will take the value from the HST images.')
   parser.add_argument('--hst_filters', type=str, nargs='+', default = ['any'], help='Required filter for the HST images. Default all filters. They can be added as a list, e.g. "F814W" "F606W".')
   parser.add_argument('--hst_integration_time_min', type=float, default = 50, help='Required minimum average integration time for a set of HST images. This quantity is a limit on the average exposure time of an entire set of observations. Therefore, longer and shorter exposures may be available in an specific data set.')
   parser.add_argument('--hst_integration_time_max', type=float, default = 2000, help='Required maximum average integration time for a set of HST images. This quantity is a limit on the average exposure time of an entire set of observations. Therefore, longer and shorter exposures may be available in an specific data set. Expossures with less that 500 seconds of integration time are preferred. The default value is 2000 seconds, which is far more than the optimal value, but allow datasets with combinations of short and long expossures to be considered.')
   parser.add_argument('--time_baseline', type=float, default = 2190, help='Minimum time baseline with respect to Gaia EDR3 in days. Default 2190.')
   parser.add_argument('--project', type=str, nargs='+', default = ['HST'], help='Processing project. E.g. HST, HLA, EUVE, hlsp_legus. Default HST. They can be added as a list, e.g. "HST", "HLA".')
   parser.add_argument('--field_id', type=str, nargs='+', default = None, help='Specify the Ids of the fields to download. This is an internal id created by GaiaHub (field_id). The default value, "y", will download all the available HST observations fulfiling the required conditions. The user can also especify "n" for none, or the specific ids separated by spaces.')

   # HST-Gaia match options
   parser.add_argument('--use_only_good_gaia', action='store_true', help = 'Force GaiaHub to use all the Gaia stars to make the alignment with HST. Otherwise, GaiaHub will use only good measurements based on Gaia EDR3 quality flags. Useful when not enough good stars are available.')
   parser.add_argument('--use_members', action='store_true', help='Whether to use only member stars for the epochs alignment or to use all available stars.')
   parser.add_argument('--preselect_cmd', action='store_true', help='If "--use_members" is in use, it enables the user to manually select member stars in the color-magnitude diagram prior to the automatic selection in the PM space. It helps when the method does not converge due to contaminanation from non-member stars.')
   parser.add_argument('--preselect_pm', action='store_true', help='If "--use_members" is in use, it enables the user to manually select member stars in the vector-point diagram prior to the automatic selection in the PM space. It helps when the method does not converge due to contaminanation from non-member stars, or when there are a significant amount of contaminants.')
   parser.add_argument('--clipping_prob_pm', type=float, default=3, help='Ratio used for clipping in PM when selecting members. Default is 0.95.')
   parser.add_argument('--ask_user_stop', action='store_true', help='It ask the user whether to continue with the next iteration instead of continuing until convergence is reached. It only works when "--use_only_good_gaia" or "--rewind_stars" are in use. It can be useful when convergence fails.')
   parser.add_argument('--max_iterations', type=int, default = 10, help='Maximum number of allowed iterations before convergence. Default 10.')
   parser.add_argument('--pm_n_components', type=int, default=1, help='Number of Gaussian componnents for pm and parallax clustering. Default is 1.')
   parser.add_argument('--previous_xym2pm', action='store_true', help='Force the program to perform the match between Gaia and HST sources. If False, the code will use existing files if any.')
   parser.add_argument('--rewind_stars', action='store_true', help='Force the program to use the PMs to rewind the stars from their second epoch to the first one before the matching.')
   parser.add_argument('--fix_mat', action='store_true', help='Force the program to use only the best stars in the MAT files to perform the alignment in the next iteration.')
   parser.add_argument('--max_separation', type=float, default= None, help='Maximum allowed separation in pixels during the match between epochs. Default 10 pixels.')
   parser.add_argument('--use_sat', action='store_true', help='Force the program to use saturated stars during the match between epochs. Default is True.')
   parser.add_argument('--wcs_search_radius', type=float, default= None, help='When set to a radius (in arcsec), the program search the closest Gaia star to each bright star in the HST image within that distance to perform a pre-alignment between the two frames. Useful when not many stars are available. Default is None.')
   parser.add_argument('--min_stars_alignment', type=int, default = 10, help='Minimum number of stars per HST image to be used for the epochs alignment. Default 10.')
   parser.add_argument('--no_amplifier_based', action='store_true', help='Force the program to use only one channel instead of amplifier-based transformations.')
   parser.add_argument('--min_stars_amp', type=int, default = 10, help='Minimum number of stars per HST amplifier to compute transformations. Default is 10.')

   #Miscellaneus options
   parser.add_argument('--n_processes', type = int, default = -1, help='The number of jobs to run in parallel. Default is -1, which uses all the available processors. For single-threaded execution use 1.')
   parser.add_argument('--load_existing', action='store_true', help='With this flag, the code will try to resume the previous Gaia search and load previous individual queries. Useful when a specific search is failing due to connection problems.')
   parser.add_argument('--no_plots', action='store_true', help='This flag prevents the code from making any plot. Useful when using distributed computing or in situations when python is not able to open an plotting device.')
   parser.add_argument('--no_error_weighted', action='store_true', help = 'The program will use non-weighted arithmetic mean to compute the PMs. By default, the code will try to make use of the errors obtained for each individual HST image in order to compute a final error-weighted mean for the PMs. This flag forces the code to use a normal arithmetic mean instead. Useful if you think all HST images should have exactly the same weight.')
   parser.add_argument('--remove_previous_files', action='store_true', help='Remove previous intermediate files.')
   parser.add_argument('--verbose', action='store_true', help='It controls the program verbosity. Default True.')
   parser.add_argument('--quiet', action='store_true', help='This flag deactivate the interactivity of GaiaHub. When used, GaiaHub will use all the default values without asking the user. This flag unables the "--preselect_cmd" option.')

   if len(argv)==0:
      parser.print_help(sys.stderr)
      sys.exit(1)

   args = parser.parse_args(argv)
   args = gh.get_object_properties(args)

   """
   The script creates directories and set files names
   """   
   gh.create_dir(args.base_path)
   gh.create_dir(args.HST_path)
   gh.create_dir(args.Gaia_path)
   if args.save_individual_queries:
      gh.create_dir(args.Gaia_ind_queries_path)

   """
   The script tries to load an existing Gaia table, otherwise it will download it from the Gaia archive.
   """
   installation_path = ''
   gh.remove_file(args.exec_path)
   os.symlink(installation_path, args.exec_path)

   astrometric_cols = 'l, b, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, dr2_radial_velocity, dr2_radial_velocity_error, ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr'

   photometric_cols = 'phot_g_mean_flux, phot_g_mean_mag AS gmag, (1.086*phot_g_mean_flux_error/phot_g_mean_flux) AS gmag_error, phot_bp_mean_mag AS bpmag, (1.086*phot_bp_mean_flux_error/phot_bp_mean_flux) AS bpmag_error, phot_rp_mean_mag AS rpmag, (1.086*phot_rp_mean_flux_error/phot_rp_mean_flux) AS rpmag_error, bp_rp, sqrt( power( (1.086*phot_bp_mean_flux_error/phot_bp_mean_flux), 2) + power( (1.086*phot_rp_mean_flux_error/phot_rp_mean_flux), 2) ) as bp_rp_error'

   quality_cols = 'ecl_lat, pseudocolour, nu_eff_used_in_astrometry, visibility_periods_used, astrometric_excess_noise_sig, astrometric_params_solved, astrometric_n_good_obs_al, astrometric_chi2_al, phot_bp_rp_excess_factor, ruwe, (phot_bp_n_blended_transits+phot_rp_n_blended_transits) *1.0 / (phot_bp_n_obs + phot_rp_n_obs) AS beta, ipd_gof_harmonic_amplitude, phot_bp_n_contaminated_transits, phot_rp_n_contaminated_transits'

   query, quality_cols = gh.columns_n_conditions(args.source_table, astrometric_cols, photometric_cols, quality_cols, args.ra, args.dec, args.search_width, args.search_height)

   try:
      Gaia_table = pd.read_csv(args.Gaia_clean_table_filename)
   except:
      Gaia_table, Gaia_queries = gh.incremental_query(query, args.area, min_gmag = args.min_gmag, max_gmag = args.max_gmag, n_processes = args.n_processes,
                                                   save_individual_queries = args.save_individual_queries, name = args.name)

      f = open(args.queries, 'w+')
      if type(Gaia_queries) is list:
         for Gaia_query in Gaia_queries:
            f.write('%s\n'%Gaia_query)
            f.write('\n')
      else:
         f.write('%s\n'%Gaia_queries)
      f.write('\n')
      f.close()

      # We fix some of the variables using the codes published with EDR3
      Gaia_table['corrected_flux_excess_factor'] = gh.correct_flux_excess_factor(Gaia_table['bp_rp'], Gaia_table['phot_bp_rp_excess_factor'])

      clean_label = gh.pre_clean_data(Gaia_table['gmag'], Gaia_table['corrected_flux_excess_factor'], Gaia_table['ruwe'], Gaia_table['ipd_gof_harmonic_amplitude'], Gaia_table['visibility_periods_used'], Gaia_table['astrometric_excess_noise_sig'], Gaia_table['astrometric_params_solved'], sigma_flux_excess_factor = args.sigma_flux_excess_factor, use_5p = args.only_5p_solutions)

      Gaia_table['clean_label'] = clean_label

      Gaia_table.to_csv(args.Gaia_clean_table_filename)

   """
   The script tries to load an existing HST table, otherwise it will download it from the MAST archive.
   """
   obs_table, data_products_by_obs = gh.search_mast(args.ra, args.dec, search_width = args.search_width, search_height = args.search_height, filters = args.hst_filters, project = args.project, t_exptime_min = args.hst_integration_time_min, t_exptime_max = args.hst_integration_time_max, date_second_epoch = args.date_second_epoch, time_baseline = args.time_baseline)

   obs_table.to_csv(args.HST_obs_table_filename, index = False)
   data_products_by_obs.to_csv(args.HST_data_table_products_filename, index = False)

   """
   Plot results and find Gaia stars within HST fields
   """
   obs_table = gh.plot_fields(Gaia_table, obs_table, args.HST_path, use_only_good_gaia = args.use_only_good_gaia,  min_stars_alignment = args.min_stars_alignment, no_plots = args.no_plots, name = args.base_path+args.base_file_name+'_search_footprint.pdf')

   if len(obs_table) > 0:

      """
      Ask whether the user wish to download the available HST images 
      """

      print(obs_table.loc[:, ['obsid', 'filters', 'n_exp', 'i_exptime', 'obs_time', 't_baseline', 'gaia_stars_per_obs', 'proposal_id', 's_ra', 's_dec', 'field_id']].to_string(index=False), '\n')

      if (args.quiet is True) and (args.field_id is None):
         print('GaiaHub will use the above sets of observations.\n')
         HST_obs_to_use = 'y'
      elif args.field_id is not None:
         print('GaiaHub will use the observations sets %s.'%(' '.join(str(p) for p in args.field_id) ))
         HST_obs_to_use = ' '.join([str(p) for p in args.field_id])
      else:
         print('Would you like to use above HST observations?\n')
         print("Type 'y' or just press enter for all observations, 'n' for none. Type the id within parentheses at the right (field_id) if you wish to use that specific set of observations. You can enter several ids separated by space. \n")
         HST_obs_to_use = input('Please type your answer and press enter: ') or 'y'
         print('\n')
      
      try:
         HST_obs_to_use = gh.str2bool(HST_obs_to_use)
      except:
         try:
            HST_obs_to_use = list(set([obsid for obsid in obs_table.obsid[[int(obsid)-1 for obsid in HST_obs_to_use.split()]] if np.isfinite(obsid)]))
         except:
            print('No valid input. Not downloading observations.')
            HST_obs_to_use = False

      if HST_obs_to_use is not False:
         if HST_obs_to_use is True:
            HST_obs_to_use = list(obs_table['obsid'].values)
         hst_images = gh.download_HST_images(data_products_by_obs.loc[data_products_by_obs['parent_obsid'].isin(HST_obs_to_use), :], path = args.HST_path)
      else:
         print('\nExiting now.\n')
         gh.remove_file(args.exec_path)
         sys.exit(1)

      """
      Select only flc
      """
      drz_images = data_products_by_obs[(data_products_by_obs['productSubGroupDescription'] == 'DRZ') & (data_products_by_obs['parent_obsid'].isin(HST_obs_to_use))]
      flc_images = data_products_by_obs[(data_products_by_obs['productSubGroupDescription'] == 'FLC') & (data_products_by_obs['parent_obsid'].isin(HST_obs_to_use))]

      """
      Call hst1pass
      """
      gh.launch_hst1pass(flc_images, HST_obs_to_use, args.HST_path, args.exec_path, force_fmin = args.fmin, force_hst1pass = args.hst1pass, n_processes = args.n_processes)

      """
      Call xym2pm_Gaia
      """
      Gaia_table_hst, lnks = gh.launch_xym2pm_Gaia(Gaia_table.copy(), flc_images, HST_obs_to_use, args.HST_path, args.exec_path, args.date_reference_second_epoch, only_use_members = args.use_members, preselect_cmd = args.preselect_cmd, preselect_pm = args.preselect_pm, rewind_stars = args.rewind_stars, force_pixel_scale = args.pixel_scale, force_max_separation = args.max_separation, force_use_sat = args.use_sat, fix_mat = args.fix_mat, no_amplifier_based = args.no_amplifier_based, min_stars_amp = args.min_stars_amp, force_wcs_search_radius = args.wcs_search_radius, n_components = args.pm_n_components, clipping_prob = args.clipping_prob_pm, use_only_good_gaia = args.use_only_good_gaia, min_stars_alignment = args.min_stars_alignment, use_mean = args.use_mean, no_plots = args.no_plots, verbose = args.verbose, quiet = args.quiet, ask_user_stop = args.ask_user_stop, max_iterations = args.max_iterations, previous_xym2pm = args.previous_xym2pm, remove_previous_files = args.remove_previous_files, n_processes = args.n_processes, plot_name = args.base_path+'PM_Sel')

      """
      Save Gaia and HST tables
      """
      obs_table.to_csv(args.HST_obs_table_filename, index = False)
      data_products_by_obs.to_csv(args.HST_data_table_products_filename, index = False)

      flc_images.to_csv(args.used_HST_obs_table_filename, index = False)
      Gaia_table_hst.to_csv(args.HST_Gaia_table_filename, index = False)
      lnks.to_csv(args.lnks_summary_filename)
      
      avg_pm = gh.weighted_avg_err(Gaia_table_hst.loc[Gaia_table_hst.use_for_alignment, ['hst_gaia_pmra_%s'%args.use_mean, 'hst_gaia_pmdec_%s'%args.use_mean, 'hst_gaia_pmra_%s_error'%args.use_mean, 'hst_gaia_pmdec_%s_error'%args.use_mean]])

      """
      Plot the results
      """
      if args.no_plots == False:
         gh.plot_results(Gaia_table_hst, lnks, drz_images, args.HST_path, avg_pm, use_mean = args.use_mean, plot_name_1 = args.base_path+args.base_file_name+'_vpd', plot_name_2 = args.base_path+args.base_file_name+'_diff', plot_name_3 = args.base_path+args.base_file_name+'_cmd', plot_name_4 = args.base_path+args.base_file_name+'_footprint', plot_name_5 = args.base_path+args.base_file_name+'_errors_xy', plot_name_6 = args.base_path+args.base_file_name+'_errors_mag', plot_name_7 = args.base_path+args.base_file_name+'_errors_color', ext = '.pdf')

      """
      Print a summary with the location of files
      """
      logresults = ' RESULTS '.center(82, '-')+'\n - Final table: %s'%args.HST_Gaia_table_filename+'\n - Used HST observations: %s'%args.used_HST_obs_table_filename+'\n'+'-'*82+'\n - A total of %i stars were used.\n'%Gaia_table_hst.use_for_alignment.sum() +' - Average absolute PM of used stars: \n   pmra = %s+-%s \n'%(gh.round_significant(avg_pm['hst_gaia_pmra_%s_%s'%(args.use_mean, args.use_mean)], avg_pm['hst_gaia_pmra_%s_%s_error'%(args.use_mean, args.use_mean)]))+'   pmdec = %s+-%s \n '%(gh.round_significant(avg_pm['hst_gaia_pmdec_%s_%s'%(args.use_mean, args.use_mean)], avg_pm['hst_gaia_pmdec_%s_%s_error'%(args.use_mean, args.use_mean)]))+'-'*82 + '\n \n Execution ended.\n'

      print('\n')
      print(logresults)

      f = open(args.logfile, 'w+')
      f.write(logresults)
      f.close()

   else:
      if args.quiet:
         print('No suitable HST observations were found. Please try with different parameters. Exiting now.')
      else:
         input('No suitable HST observations were found. Please try with different parameters.\nPress enter to exit.\n')

   # Remove temporary links
   gh.remove_file(args.exec_path)

if __name__ == '__main__':
    gaiahub(sys.argv[1:])
    sys.exit(0)

"""
Andres del Pino Molina
"""

