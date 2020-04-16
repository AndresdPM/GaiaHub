#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import pandas as pd
import argparse
import sys
from astropy.time import Time
from astropy.coordinates import match_coordinates_sky
import os
from astropy.table import Table
import matplotlib.pyplot as plt

from sklearn.utils.validation import check_is_fitted


def get_uwe(phot_g_mean_mag, bp_rp, astrometric_chi2_al, astrometric_n_good_obs_al, norm_uwe = True):
   """
   Calculates the corresponding RUWE for Gaia stars.
   """

   uwe = np.sqrt(astrometric_chi2_al/(astrometric_n_good_obs_al - 5.))

   if norm_uwe:
      #We make use of the normalization array from files table_u0_g_col.txt, table_u0_g.txt
      print('Normalizing uwe index...')

      has_color = np.isfinite(bp_rp) & (np.isfinite(phot_g_mean_mag))
      u0gc = pd.read_csv('./Auxiliary/DR2_RUWE_V1/table_u0_g_col.txt', header =0)

      #histogram
      dx = 0.01
      dy = 0.1
      bins = [np.arange(np.amin(u0gc['g_mag'])-0.5*dx, np.amax(u0gc['g_mag'])+dx, dx), np.arange(np.amin(u0gc[' bp_rp'])-0.5*dy, np.amax(u0gc[' bp_rp'])+dy, dy)]
      
      posx = np.digitize(phot_g_mean_mag[has_color], bins[0])
      posy = np.digitize(bp_rp[has_color], bins[1])
      
      u0_gc = np.reshape(np.array(u0gc[' u0']), (len(bins[0])-1, len(bins[1])-1))[posx, posy]
      uwe[has_color] /= np.array(u0_gc)

      if not all(has_color):
         u0g = pd.read_csv('./Auxiliary/DR2_RUWE_V1/table_u0_g.txt', header =0)

         posx = np.digitize(phot_g_mean_mag[~has_color], bins[0])

         u0_c = u0g[' u0'][posx]
         uwe[~has_color] /= np.array(u0_c)

   return uwe


def clean_astrometry(phot_g_mean_mag, bp_rp, astrometric_chi2_al, astrometric_n_good_obs_al, norm_uwe = True):
   """
   Select stars with good astrometry in Gaia
   """
   
   b = 1.2 * np.maximum(np.ones_like(phot_g_mean_mag), np.exp(-0.2*(phot_g_mean_mag-19.5)))
   
   uwe = get_uwe(phot_g_mean_mag, bp_rp, astrometric_chi2_al, astrometric_n_good_obs_al, norm_uwe = norm_uwe)
   
   if norm_uwe:
      labels_uwe = uwe < 1.5
   else:
      labels_uwe = uwe < 1.95
 
   labels_astrometric = (uwe < b) & labels_uwe

   return labels_astrometric, uwe


def clean_photometry(bp_rp, phot_bp_rp_excess_factor):
   """
   Select stars with good photometry in Gaia
   """

   labels_photometric = (1.0 + 0.015*bp_rp**2 < phot_bp_rp_excess_factor) & (1.5*(1.3 + 0.06*bp_rp**2) > phot_bp_rp_excess_factor)

   return labels_photometric


def pre_clean_data(phot_g_mean_mag, bp_rp, astrometric_chi2_al, astrometric_n_good_obs_al, phot_bp_rp_excess_factor, norm_uwe = True):
   """
   This routine cleans the Gaia data from astrometrically and photometric bad measured stars.
   """

   labels_photometric = clean_photometry(bp_rp, phot_bp_rp_excess_factor)
   
   labels_astrometric, uwe = clean_astrometry(phot_g_mean_mag, bp_rp, astrometric_chi2_al, astrometric_n_good_obs_al, norm_uwe = norm_uwe)
   
   return labels_photometric & labels_astrometric, uwe


def remove_jobs():
   """
   This routine removes jobs from the Gaia archive server
   """

   list_jobs = []
   for job in Gaia.list_async_jobs():
      list_jobs.append(job.get_jobid())
   
   Gaia.remove_jobs(list_jobs)


def gaia_log_in():
   """
   This routine log in to the Gaia archive
   """

   from astroquery.gaia import Gaia
   import getpass
   
   while True:
      user_loging = input("Gaia username: ")
      user_paswd = getpass.getpass(prompt='Gaia password: ') 
      try:
         Gaia.login(user=user_loging, password=user_paswd)
         print("Welcome to the Gaia server!")
         break

      except:
         print("Incorrect username or password!")

   return Gaia


def gaia_query(Gaia, ra, dec, width, height, gmag_min, gmag_max, clean_data = False, norm_uwe=True, save_individual_queries = False, output_name = 'output'):
   """
   This routine launch the query to the Gaia archive
   """

   if clean_data:
      query = "SELECT source_id, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, radial_velocity, radial_velocity_error, ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr, astrometric_n_obs_al, astrometric_n_obs_ac, astrometric_n_good_obs_al, astrometric_n_bad_obs_al, astrometric_gof_al, astrometric_chi2_al, astrometric_excess_noise, astrometric_excess_noise_sig, astrometric_weight_al, astrometric_pseudo_colour, astrometric_pseudo_colour_error, mean_varpi_factor_al, astrometric_matched_observations, visibility_periods_used, astrometric_sigma5d_max, matched_observations, phot_g_n_obs, phot_g_mean_mag AS gmag, (1.09*phot_g_mean_flux_error/phot_g_mean_flux) AS gmag_error, phot_bp_mean_mag AS bpmag, (1.09*phot_bp_mean_flux_error/phot_bp_mean_flux) AS bpmag_error, phot_rp_mean_mag AS rpmag, (1.09*phot_rp_mean_flux_error/phot_rp_mean_flux) AS rpmag_error, phot_g_mean_flux, phot_g_mean_flux_error, phot_bp_n_obs, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_rp_n_obs, phot_rp_mean_flux, phot_rp_mean_flux_error,phot_bp_rp_excess_factor, bp_rp, bp_g, g_rp FROM gaiadr2.gaia_source WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),BOX('ICRS',%.8f,%.8f,%.8f,%.8f))=1 AND (phot_g_mean_mag > %.4f) AND (phot_g_mean_mag <= %.4f) AND (1.0 + 0.015*(phot_bp_mean_mag - phot_rp_mean_mag)*(phot_bp_mean_mag - phot_rp_mean_mag) < phot_bp_rp_excess_factor) AND (1.5*(1.3 + 0.06*(phot_bp_mean_mag - phot_rp_mean_mag)*(phot_bp_mean_mag - phot_rp_mean_mag)) > phot_bp_rp_excess_factor)"%(ra, dec, width, height, gmag_max, gmag_min)

   else:
      query = "SELECT source_id, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, radial_velocity, radial_velocity_error, ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr, astrometric_n_obs_al, astrometric_n_obs_ac, astrometric_n_good_obs_al, astrometric_n_bad_obs_al, astrometric_gof_al, astrometric_chi2_al, astrometric_excess_noise, astrometric_excess_noise_sig, astrometric_weight_al, astrometric_pseudo_colour, astrometric_pseudo_colour_error, mean_varpi_factor_al, astrometric_matched_observations, visibility_periods_used, astrometric_sigma5d_max, matched_observations, phot_g_n_obs, phot_g_mean_mag AS gmag, (1.09*phot_g_mean_flux_error/phot_g_mean_flux) AS gmag_error, phot_bp_mean_mag AS bpmag, (1.09*phot_bp_mean_flux_error/phot_bp_mean_flux) AS bpmag_error, phot_rp_mean_mag AS rpmag, (1.09*phot_rp_mean_flux_error/phot_rp_mean_flux) AS rpmag_error, phot_g_mean_flux, phot_g_mean_flux_error, phot_bp_n_obs, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_rp_n_obs, phot_rp_mean_flux, phot_rp_mean_flux_error,phot_bp_rp_excess_factor, bp_rp, bp_g, g_rp FROM gaiadr2.gaia_source WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),BOX('ICRS',%.8f,%.8f,%.8f,%.8f))=1 AND (phot_g_mean_mag > %.4f) AND (phot_g_mean_mag <= %.4f)"%(ra, dec, width, height, gmag_max, gmag_min)

   try:
      job = Gaia.launch_job_async(query)
   except:
      Gaia = gaia_log_in()
      job = Gaia.launch_job_async(query)

   try:
      result = job.get_results()
   except:
      Gaia = gaia_log_in()
      result = job.get_results()

   try:
      removejob = Gaia.remove_jobs([job.jobid])
   except:
      Gaia = gaia_log_in()
      removejob = Gaia.remove_jobs([job.jobid])
   
   if clean_data:
      pre_clean_labels, uwe = pre_clean_data(result['gmag'], result['bpmag'] - result['rpmag'], result['astrometric_chi2_al'], result['astrometric_n_good_obs_al'], result['phot_bp_rp_excess_factor'], norm_uwe = norm_uwe)

      result = result[pre_clean_labels]

   if save_individual_queries:
      result.write('./%s/Individual_queries/%s_%.3f_%.3f_%.3fG_%.4f_%.4f.dat'%(output_name,output_name,ra,dec,r,gmag_min,gmag_max), format="ascii.csv", overwrite=True)
   
   print('Table with %i stars'%len(result))
   return result


def get_mag_bins(min_mag, max_mag, n_bins_mag, mag = None):
   """
   This routine generates logarithmic spaced bins for G magnitude
   """

   bins_mag = (1.0 + max_mag - np.logspace(np.log10(1.), np.log10(1. + max_mag - min_mag), num = n_bins_mag, endpoint = True))
   
   try:
      pos_mag = np.digitize(mag, bins_mag) - 1
      return bins_mag, pos_mag
   except:
      return bins_mag


def gaia_multi_query_run(args):
   return gaia_query(*args)


def incremental_query(ra, dec, width, height, gmag_min = 10.0, gmag_max = 19.5, clean_data = True, norm_uwe = True, save_individual_queries = False, output_name = 'output'):
   """
   This routine search the Gaia archive and downloads the stars using parallel workers.
   """

   from multiprocessing import Pool, cpu_count
   from astropy.table import vstack

   if save_individual_queries:
      path = './%s/Individual_queries'%(output_name)
      try:
         os.mkdir(path)
      except OSError:  
         print ("Creation of the directory %s failed" % path)
      else:  
         print ("Successfully created the directory %s " % path)

   Gaia = gaia_log_in()

   num_nodes = np.max((1, np.round(np.sqrt((gmag_max - gmag_min) * (gmag_max + gmag_min) * (0.5*width**2+0.5*height**2)) / 3 )))

   mag_nodes = get_mag_bins(gmag_min, gmag_max, num_nodes)[0]

   if num_nodes > 1:

      print("Executing %s jobs."%(num_nodes-1))

      nproc = int(np.min((num_nodes, 20, cpu_count()*2)))

      pool = Pool(nproc)

      args = []
      for ii, node in enumerate(range(len(mag_nodes)-1)):

         args.append((Gaia, ra, dec, width, height, mag_nodes[ii], mag_nodes[ii+1], clean_data, norm_uwe, save_individual_queries, output_name))
         
      tables_gaia = pool.map(gaia_multi_query_run, args)

      result_gaia = vstack(tables_gaia).to_pandas()
   
      pool.close()

   else:
      result_gaia = gaia_query(Gaia, ra, dec, width, height, gmag_max, gmag_min, clean_data, norm_uwe, save_individual_queries, output_name)
      
      result_gaia = result_gaia.to_pandas()
      
   Gaia.logout()
   
   return result_gaia


def str2bool(v):
   """
   This routine converts ascii input to boolean
   """
 
   if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
   elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
   else:
      raise argparse.ArgumentTypeError('Boolean value expected.')


def cli_progress_test(current, end_val, bar_length=50):
   """
   Just a progress bar
   """

   percent = float(current) / end_val
   hashes = '#' * int(round(percent * bar_length))
   spaces = ' ' * (bar_length - len(hashes))
   sys.stdout.write("\rProcessing: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
   sys.stdout.flush()


def plot_fields(Gaia_stars, obs_table, name = 'test.png'):
   """
   This routine plots the fields and select Gaia stars within them
   """

   import matplotlib.pyplot as plt
   from matplotlib.patches import Polygon
   from matplotlib.collections import PatchCollection
   from shapely.geometry.polygon import Polygon as shap_polygon
   from shapely.geometry import Point
   
   fig, ax = plt.subplots(1,1, figsize = (5.5, 5.5))
   patches = []
   gaia_stars_per_field = []
   for ii, footprint_str in enumerate(obs_table['s_region']):
      cli_progress_test(ii+1, len(obs_table))

      try:
         list_coo = footprint_str.split('POLYGON')[1::]
         if len(list_coo) > 1:
            for poly in list_coo:
               try:
                  poly = list(map(float, poly.split()))
               except:
                  poly = list(map(float, poly.split('J2000')[1::][0].split()))

               tuples_list = [(ra % 360, dec) for ra, dec in zip(poly[0::2], poly[1::2])]

               polygon = Polygon(tuples_list, True, ec = 'k', fc = 'None', antialiased = True, lw = 0.75, zorder = 4)
               footprint =  shap_polygon(tuples_list)
               star_counts = 0
               for ra, dec in zip(Gaia_stars.ra, Gaia_stars.dec):
                  if Point(ra, dec).within(footprint):
                     star_counts += 1

               gaia_stars_per_field.append(star_counts)
               patches.append(polygon)
         else:

            try:
               poly = list(map(float, list_coo[0].split()))
            except:
               poly = list(map(float, list_coo[0].split('J2000')[1::][0].split()))

            tuples_list = [(ra % 360, dec) for ra, dec in zip(poly[0::2], poly[1::2])]

            polygon = Polygon(tuples_list, True, ec = 'k', fc = 'None', antialiased = True, lw = 0.75, zorder = 4)
            footprint =  shap_polygon(tuples_list)

            star_counts = 0
            for ra, dec in zip(Gaia_stars.ra, Gaia_stars.dec):
               if Point(ra, dec).within(footprint):
                  star_counts += 1
            
            gaia_stars_per_field.append(star_counts)
            patches.append(polygon)
      except:
         pass
   
   print('\n')

   obs_table['gaia_stars_per_field'] = gaia_stars_per_field
   
   p = PatchCollection(patches, alpha=1, ec = 'k', fc = 'None', antialiased = True, lw = 0.75, zorder = 4)
   ax.add_collection(p)
   
   ax.plot(Gaia_stars.ra, Gaia_stars.dec, '.', color = '0.4', ms = 0.75, zorder = 1)

   max_ptp = np.max([np.ptp(Gaia_stars.ra.values), np.ptp(Gaia_stars.dec.values)])

   ax.set_xlim(Gaia_stars.ra.max(), Gaia_stars.ra.min())
   ax.set_ylim(Gaia_stars.dec.min(), Gaia_stars.dec.max())
   ax.grid()

   ax.set_xlabel(r'$\alpha$ [deg]')
   ax.set_ylabel(r'$\delta$ [deg]')

   plt.savefig(name, bbox_inches='tight')
   plt.show()
   
   return obs_table


def search_mast(ra, dec, width, height, filters = 'any', t_exptime_min = 50, t_exptime_max = 2500, time_baseline = 3650):
   """
   This routine search for HST observations in MAST at a given position
   """

   from astroquery.mast import Catalogs
   from astroquery.mast import Observations

   ra1 = ra - width / 2 - 0.056*np.cos(np.deg2rad(dec))
   ra2 = ra + width / 2 + 0.056*np.cos(np.deg2rad(dec))
   dec1 = dec - height / 2 - 0.056
   dec2 = dec + height / 2 + 0.056
   
   tgaia = Time('%4i-%02i-%02iT00:00:00.000'%(2016, 5, 23))
   
   t_max = tgaia.mjd - time_baseline
   
   if filters == ['any']:
      filters=['F555W','F606W','F775W','F814W','F850LP']
   elif type(filters) is not list:
      filters = [filters]

   obs_table = Observations.query_criteria(dataproduct_type=["image"], obs_collection=["HST"], s_ra=[ra1, ra2], s_dec=[dec1, dec2], instrument_name=['ACS/WFC', 'WFC3/UVIS'], t_max=[0, t_max], filters = filters)

   data_products_by_obs = search_data_products_by_obs(obs_table)
   
   #Pandas is easier:
   obs_table = obs_table.to_pandas()
   data_products_by_obs = data_products_by_obs.to_pandas()
   
   obs_table = obs_table.merge(data_products_by_obs.groupby(['parent_obsid'])['parent_obsid'].count().rename_axis('obsid').rename('n_exp'), on = ['obsid'])
   obs_table['i_exptime'] = obs_table['t_exptime'] / obs_table['n_exp']
   
   data_products_by_obs = data_products_by_obs.merge(obs_table.loc[:, ['obsid', 'i_exptime']].rename(columns={'obsid':'parent_obsid'}), on = ['parent_obsid'])
   
   #We select by individual exp time:
   obs_table = obs_table.loc[(obs_table.i_exptime > t_exptime_min) & (obs_table.i_exptime < t_exptime_max)]
   data_products_by_obs = data_products_by_obs.loc[(data_products_by_obs.i_exptime > t_exptime_min) & (data_products_by_obs.i_exptime < t_exptime_max)]

   return Table.from_pandas(obs_table), Table.from_pandas(data_products_by_obs)


def search_data_products_by_obs(obs_table):
   """
   This routine search for images in MAST related to the given observations table
   """
   
   from astroquery.mast import Observations

   data_products_by_obs = Observations.get_product_list(obs_table)

   return data_products_by_obs[data_products_by_obs['productSubGroupDescription'] == 'FLC']


def download_HST_images(data_products_by_obs, path = './'):
   """
   This routine downloads the selected HST images from MAST
   """
   
   from astroquery.mast import Observations

   flc_images = Observations.download_products(data_products_by_obs[data_products_by_obs['productSubGroupDescription'] == 'FLC'], download_dir=path)

   return flc_images


def read_isochrones(Ages, Zs, Gmag_max = -3.5):
   """
   This routine will read PARSEC isochrones and return their extinction and distance modulus corrected Gaia magnitudes.
   """

   print('Reading isochrone(s).')

   Z_models_list = np.array([0.00015, 0.00024, 0.00038, 0.00061, 0.00096, 0.00152, 0.00241, 0.00382, 0.00605, 0.0096, 0.0152, 0.0241])

   try:
      Z_models_lim = [min(Z_models_list, key=lambda x:abs(x-min(Zs))), min(Z_models_list, key=lambda x:abs(x-max(Zs)))]
      Z_models = Z_models_list[(Z_models_list >= Z_models_lim[0]) & (Z_models_list <= Z_models_lim[1])]
   except:
      Z_models = [min(Z_models_list, key=lambda x:abs(x-Zs))]

   isochrones = []
   for Z_model in Z_models:
      print('Readding Z:', Z_model)
      try:
         isochrone = np.loadtxt('./Auxiliary/PARSEC_Tracks/%s.dat'%Z_model, usecols = [7, 1, 3, 23, 24, 25])
      except:
         print("Could not find isochrones in ./Auxiliary/PARSEC_Tracks/")
         sys.exit(1)
 
      for Age_model in Ages:
         Age_models = (np.abs(isochrone[:,1] - Age_model*1e9) == np.amin(np.abs(isochrone[:,1] - Age_model*1e9)))
         Max_mag = isochrone[:,3] <= Gmag_max
         try:
            isochrone_age_maxg = isochrone[Age_models & Max_mag]        
            print('Readding Age:', isochrone_age_maxg[0, 1]*1e-9)
            for label in set(isochrone_age_maxg[:, 0]):
               evolutionary_state = isochrone_age_maxg[:, 0] == label
               isochrones.append(pd.DataFrame(data = {'evolutionary_state': isochrone_age_maxg[evolutionary_state, 0], 'Mass': isochrone_age_maxg[evolutionary_state, 2], 'gmag_0': isochrone_age_maxg[evolutionary_state, 3], 'bpmag_0': isochrone_age_maxg[evolutionary_state, 4], 'rpmag_0': isochrone_age_maxg[evolutionary_state, 5]}))
         except:
            pass

   return isochrones


def combine_isochrones(isochrones, cmd_broadening = 0.05, extended_HB = False):
   """
   This routine will combine isochrones prior to to select stars in the CMD.
   """
   
   from shapely.geometry import LineString
   from shapely.geometry.point import Point   
   from shapely.ops import unary_union

   print('Unary union of isochrones.')

   N_isochornes = len(isochrones)
   isochrones_cmd = [None]*N_isochornes
   for ii, isochrone in enumerate(isochrones):
      cli_progress_test(ii+1, N_isochornes)
      if len(isochrone) >=2:
         if (extended_HB == True) & (isochrone.evolutionary_state == 4).all():
            isochrones_cmd[ii] = LineString([(isochrone_color-0.001/isochrone_color**2, isochrone_mag+0.0001/isochrone_color**3.4) for isochrone_color, isochrone_mag in zip(isochrone.bpmag_0-isochrone.rpmag_0, isochrone.gmag_0)]).buffer(cmd_broadening)
         else:
            isochrones_cmd[ii] = LineString([(isochrone_color, isochrone_mag) for isochrone_color, isochrone_mag in zip(isochrone.bpmag_0-isochrone.rpmag_0, isochrone.gmag_0)]).buffer(cmd_broadening)            
      else:
         isochrones_cmd[ii] = Point(isochrone.bpmag_0-isochrone.rpmag_0, isochrone.gmag_0).buffer(cmd_broadening)
   
   print('\n')
   isochrones_cmd = unary_union(isochrones_cmd)
   
   return isochrones_cmd


def cmd_cleaning(stars, isochrones_cmd, clipping_sigma = 3., plots = True, plot_name = ''):
   """
   This routine will clean the CMD by rejecting stars more than intrinsic_broadening + clipping_sigma away from the used isochrone(s).
   """

   from shapely.geometry.point import Point
   from shapely.affinity import scale
   from descartes import PolygonPatch
   
   print('Selecting stars in the cmd.')

   stars_cmd = [scale(Point((stars_color, stars_mag)).buffer(1), xfact=stars_color_error, yfact=stars_mag_error) for stars_color, stars_mag, stars_color_error, stars_mag_error in zip(stars.bpmag_0-stars.rpmag_0, stars.gmag_0, clipping_sigma*np.sqrt(stars.bpmag_error**2+stars.rpmag_error**2), clipping_sigma*stars.gmag_error)]

   labels_cmd = np.zeros_like(stars_cmd, dtype=bool)
   for ii, star in enumerate(stars_cmd):
      cli_progress_test(ii+1, len(stars_cmd))
      labels_cmd[ii] = star.intersects(isochrones_cmd)

   if plots == True:
      plt.close('all')

      fig = plt.figure(1)
      ax = fig.add_subplot(111)
      try:
         patch = PolygonPatch(isochrones_cmd, facecolor='orange', lw=0, alpha = 0.5, zorder = 2)
         ax.add_patch(patch)
      except:
         pass
      ax.plot((stars.bpmag_0-stars.rpmag_0)[labels_cmd], stars.gmag_0[labels_cmd], 'b.', label = 'selected', ms = 1., zorder = 1)
      ax.plot((stars.bpmag_0-stars.rpmag_0)[~labels_cmd], stars.gmag_0[~labels_cmd], 'k.', label = 'rejected', ms = 0.5, zorder = 0, alpha = 0.5)
      ax.set_ylim([stars.gmag_0[stars.gmag_0 < 50].max(), stars.gmag_0[stars.gmag_0 < 50].min()-1.])
      ax.set_xlim([min(stars.bpmag_0-stars.rpmag_0)-1., max(stars.bpmag_0-stars.rpmag_0)+1.])      
      ax.set_xlabel(r'$G_{BP}-G_{RP}$')
      ax.set_ylabel(r'$G$')
      plt.legend()
      plt.savefig(plot_name)

   print('\n')

   return labels_cmd


def create_dir(path):
   """
   This routine creates the pertinent directories
   """
   
   try:
      os.mkdir(path)
   except OSError:  
      print ("Creation of the directory %s failed" % path)
   else:  
      print ("Successfully created the directory %s " % path)


def get_AV_map(table):
   """
   This routine downloads the reddening maps
   """

   from dustmaps.sfd import SFDQuery
   from astropy.coordinates import SkyCoord
   import astropy.units as u

   print('Obtaining AV map.')

   icrs = SkyCoord(ra = np.array(table.ra)*u.deg, dec = np.array(table.dec)*u.deg, frame = 'icrs')

   sfd = SFDQuery()
   
   AV = 0.86*(3.1*sfd(icrs))  #0.86 is for Schlafly & Finkbeiner 2011 (ApJ 737, 103)

   return AV


def simple_reddening_correction(AV):

   Ax = np.tile([0.85926, 1.06794, 0.65199], (len(AV),1)) * np.expand_dims(AV, axis=1)

   return Ax


def get_object_properties(args):

   #We try to get coordinates:
   if (args.ra is None) or (args.dec is None):
      try:
         from astroquery.simbad import Simbad
         from astropy.coordinates import SkyCoord
         from astropy import units as u
         customSimbad = Simbad()
         customSimbad.add_votable_fields('distance', 'propermotions', 'dim', 'fe_h')

         object_table = customSimbad.query_object(args.name)
         print(object_table['MAIN_ID', 'RA', 'DEC'])

         coo = SkyCoord(ra = object_table['RA'], dec = object_table['DEC'], unit=(u.hourangle, u.deg))

         args.ra = float(coo.ra.deg)
         args.dec = float(coo.dec.deg)
      except:
         if args.ra is None:
            args.ra = float(input('R.A. not defined, please enter R.A. in degrees:'))
         if args.dec is None:
            args.dec = float(input('Dec not defined, please enter Dec in degrees:'))

   #We try to get distances:
   if args.distance is None:
      try:
         if (object_table['Distance_distance'].mask == False):
            if object_table['Distance_unit'] == 'Mpc':
               args.distance = float(object_table['Distance_distance']*1e3)
            elif object_table['Distance_unit'] == 'kpc':
               args.distance = float(object_table['Distance_distance'])
            elif object_table['Distance_unit'] == 'pc':
               args.distance = float(object_table['Distance_distance']*1e-3)
      except:
         try:
            args.distance = float(input('Distance to the object not found, please enter distance in kpc (Press enter to skip: No CMD selection will be performed):'))
         except:
            args.distance = None

   #Try to get radius
   if args.r is None:
      try:
         if (object_table['GALDIM_MAJAXIS'].mask == False):
            args.r = max(np.round(float(2. * object_table['GALDIM_MAJAXIS'] / 60.), 2), 0.1)
      except:
         try:
            args.r = float(input('Search radius not defined, please enter the search radius in degrees (Press enter to adopt the default value of 1 deg):'))
         except:
            args.r = 1.0

   args.width = np.abs(args.r/np.cos(np.deg2rad(args.dec)))
   args.height = args.r

   #Try to get metallicity
   if args.feh is None:
      try:
         if (object_table['Fe_H_Fe_H'].mask == False):
            args.feh = float(object_table['Fe_H_Fe_H'])
      except:
         try:
            print('Metallicity [Fe/H] not defined, please enter [Fe/H] as a range from min to max (Press enter to adopt the default values [-3, 0]):')
            min_feh = float(input('Min [Fe/H]:'))
            max_feh = float(input('Min [Fe/H]:'))
            args.feh = [min_feh, max_feh]
         except:
            args.feh = [-3., 0.]
      args.z = np.round(0.019*10**np.array(args.feh), 6)

   #We try to get PMs:
   if args.pmra is None:
      try:
         if (object_table['PMRA'].mask == False):      
            args.pmra = float(object_table['PMRA'])
            args.pmdec = float(object_table['PMDEC'])
      except:
         try:
            args.pmra = float(input('PMRA not defined, please enter pmra in mas (Press enter to ignore):'))
         except:
            args.pmra = 0.0
 
   if args.pmdec is None:
      try:
         if (object_table['PMDEC'].mask == False):      
            args.pmdec = float(object_table['PMDEC'])
      except:
         try:
            args.pmdec = float(input('PMDEC not defined, please enter pmdec in mas (Press enter to ignore):'))
         except:
            args.pmdec = 0.0

   if args.parallax is None:
      args.parallax = 0.0
   
   name_coo = 'ra%.3f_%.3f_dec%.3f_%.3f'%(args.ra-args.width/2, args.ra+args.width/2, args.dec-args.height/2, args.dec+args.height/2)

   if args.name is not None:
      args.name = args.name.replace(" ", "_")+'_'+name_coo
   else:
      args.name = name_coo

   return args


def members_prob(table, clf, vars, clipping_prob = 3, data_0 = None):
   """
   This routine will find probable members through scoring of a passed model (clf)
   """

   clustering_data = table.clustering_data == 1

   if (clustering_data.sum() > 1):

      if data_0 is None:
         data_0 = table.loc[clustering_data, vars].median().values

      data = table.loc[:,vars] - data_0

      data_std = data.loc[clustering_data, :].std().values

      clf.fit(data.loc[clustering_data, :] / data_std)

      log_prob = clf.score_samples(data / data_std)

      prob_clf = np.abs(np.median(log_prob[clustering_data]) - log_prob)
      label_clf = prob_clf <= clipping_prob*np.std(log_prob[clustering_data])

      results = pd.DataFrame(data={'member_prob': prob_clf, 'member_label': label_clf}, index = table.index)

   else:

      results = pd.DataFrame(data={'member_prob': 1.0, 'member_label': True}, index = table.index)

   return results


def pm_cleaning_GMM_recursive(table, vars, alt_table = None, data_0 = None, n_components = 1, clipping_prob = 3, plots = True, verbose = False, plot_name = ''):
   """
   This routine iteratively find members using a Gaussian mixture model.
   """

   from sklearn import mixture

   table['clustering_data'] = 1
   if alt_table is not None:
      alt_table['clustering_data'] = 0
      table = pd.concat([table, alt_table], ignore_index = True, sort=True)

   clf = mixture.GaussianMixture(n_components = n_components, covariance_type='full', means_init = np.zeros((n_components, len(vars))))

   convergence = False
   iteration = 0
   while not convergence:
      if verbose:
         print("\rIteration %i, %i objects remain."%(iteration, len(table)))

      clustering_data = table.clustering_data == 1

      clust = table.loc[:,vars]
      clust['clustering_data'] = table.clustering_data

      if iteration > 3:
         data_0 = None

      fitting = members_prob(clust, clf, vars, clipping_prob = clipping_prob,  data_0 = data_0)

      if (iteration > 999):
         convergence = True
      else:
         convergence = table.equals(table[fitting.member_label])

      if not convergence:
         try:
            rejected = pd.concat([rejected, table[~fitting.member_label]])
         except:
            rejected = table[~fitting.member_label]
         table = table[fitting.member_label]

      iteration += 1

   if plots == True:

      plt.close('all')
      fig, ax1 = plt.subplots(1, 1)
      ax1.plot(rejected.pmra, rejected.pmdec, 'k.', ms = 0.5, zorder = 0)
      ax1.scatter(table.pmra, table.pmdec, c = fitting.member_prob[fitting.member_label], s = 1, zorder = 1)
      ax1.set_xlabel(r'$PM_{RA}$')
      ax1.set_ylabel(r'$PM_{Dec}$')
      ax1.set_xlim(table.pmra.mean()-3*table.pmra.std(), table.pmra.mean()+3*table.pmra.std())
      ax1.set_ylim(table.pmdec.mean()-3*table.pmdec.std(), table.pmdec.mean()+3*table.pmdec.std())      
      ax1.grid()
      plt.savefig(plot_name)
      plt.close('all')

   if alt_table is not None:
      if "rejected" in locals():
         return table.loc[table.clustering_data == 1, :], rejected.loc[rejected.clustering_data == 1, :], table.loc[table.clustering_data == 0, :], rejected.loc[rejected.clustering_data == 0, :]
      else:
         return table.loc[table.clustering_data == 1, :], pd.DataFrame(columns = table.columns), table.loc[table.clustering_data == 0, :], pd.DataFrame(columns = table.columns)
   else:
      if "rejected" in locals():
         return table.loc[table.clustering_data == 1, :], rejected.loc[rejected.clustering_data == 1, :]
      else:
         return table.loc[table.clustering_data == 1, :], pd.DataFrame(columns = table.columns)


def main(argv):  
   """
   Inputs
   """

   parser = argparse.ArgumentParser(description="This script asynchronously download Gaia DR2 data, cleans it from poorly measured sources.")
   parser.add_argument('--name', type=str, default = None, help='Name for the Output table.')
   parser.add_argument('--ra', type=float, default = None, help='Central R.A.')
   parser.add_argument('--dec', type=float, default = None, help='Central Dec.')
   parser.add_argument('--pmra', type=float, default=None, help='pmra in mas.')
   parser.add_argument('--pmdec', type=float, default=None, help='pmdec in mas.')
   parser.add_argument('--parallax', type=float, default=None, help='parallax in mas.')
   parser.add_argument('--r', type=float, default = None, help='Radius of search in degrees (in R.A.).')
   parser.add_argument('--distance', type=float, default = None, help='Distance in kpc.')
   parser.add_argument('--age', type=float, nargs='+', default= [12.], help='Age of the system in Gyr. Both, a single value or a range can be provided. Default is age within [8., 13.7].')
   parser.add_argument('--age_step', type=float, default= 0.1, help='Age resolution.')
   parser.add_argument('--age_mode', type=str, default= "discrete", help="If 'discrete', only the ages specified will be used. If 'continuous', ages between the max and min of --age will be used every --age_step")
   parser.add_argument('--feh', type=float, nargs='+', default= None, help='Metallicity ([Fe/H]) of the system. Both, a single value or a range can be provided. Default is range [Fe/H] within [-2.5, -0.5].')
   parser.add_argument('--cmd_broadening', type=float, default=0.05, help='CMD intrinsic color broadening in magnitudes. It is used to compute the maximum distance in color to a star as to consider it as possible bember of an isochrone population. Default is 0.01.')
   parser.add_argument('--clipping_sigma_cmd', type=float, default=3., help='Sigma used for clipping in the cmd. i.e. distance to the isochrone. Default is 3.')
   parser.add_argument('--clipping_prob_pm', type=float, default=3., help='Sigma used for clipping pm and parallax. Default is 3.')
   parser.add_argument('--pm_n_components', type=int, default=1, help='Number of Gaussian componnents for pm and parallax clustering. Default is 1.')
   parser.add_argument('--hst_filters', type=str, nargs='+', default = ['any'], help='Required filter for the HST images.')
   parser.add_argument('--hst_integration_time_min', type=float, default = 50, help='Required integration time for the HST images.')
   parser.add_argument('--hst_integration_time_max', type=float, default = 500, help='Required integration time for the HST images.')
   parser.add_argument('--time_baseline', type=float, default = 4., help='Minimum time baseline with respect to Gaia DR2.')
   parser.add_argument('--gmag_max', type=float, default = 22.0, help='Fainter G magnitude')
   parser.add_argument('--gmag_min', type=float, default = 17.0, help='Brighter G magnitude')
   parser.add_argument('--clean_uwe', type = str2bool, default = True)
   parser.add_argument('--norm_uwe', type = str2bool, default = True)
   parser.add_argument('--clean_data', type = str2bool, default = False)
   parser.add_argument('--previous_PMs_table', type=str, default = "Filename.csv", help='In case previous PMs are available.')
   parser.add_argument('--plots', type=str2bool, default=True, help='Create sanity plots. Default is "True".')

   args = parser.parse_args(argv)

   args = get_object_properties(args)

   """
   The script creates directories and set files names
   """
   base_path = './%s/'%(args.name)
   HST_path = base_path+'HST/'
   Gaia_path = base_path+'Gaia/'
   
   Gaia_raw_table_filename = Gaia_path+args.name+'.csv'
   Gaia_members_table_filename = Gaia_path+args.name+'_members.csv'
   Gaia_not_members_table_filename = Gaia_path+args.name+'_not_members.csv'
   Gaia_faint_table_filename = Gaia_path+args.name+'_faint.csv'
   
   HST_obs_table_filename = HST_path+args.name+'_obs.csv'
   HST_data_table_products_filename = HST_path+args.name+'_data_products.csv'

   create_dir(base_path)
   create_dir(HST_path)
   create_dir(Gaia_path)

   """
   The script tries to load an existing Gaia table, otherwise it will download it from the Gaia archive.
   """
   try:
      Gaia_table = pd.read_csv(Gaia_raw_table_filename)
   except:
      Gaia_table = incremental_query(args.ra, args.dec, args.width, args.height, gmag_min = args.gmag_min, gmag_max = args.gmag_max, clean_data = args.clean_data, norm_uwe = args.norm_uwe, output_name = args.name)
      Gaia_table.to_csv(Gaia_raw_table_filename, index = False)

   Gaia_table['AV'] = get_AV_map(Gaia_table.loc[:, ['ra','dec']])

   if args.distance is not None:
      """
      If distance is defined, the code will attempt to first select stars using isochrones.
      """
      has_color_pm = Gaia_table.loc[:, ['bpmag', 'rpmag', 'pmra', 'pmdec', 'parallax']].notnull().all(axis = 1)
      too_faint = Gaia_table[~has_color_pm]
      Gaia_table = Gaia_table[has_color_pm]

      Gaia_table[['gmag_0','bpmag_0','rpmag_0']] = Gaia_table[['gmag','bpmag','rpmag']] - (simple_reddening_correction(Gaia_table['AV']) + (5.*np.log10((args.distance)*1e3)-5.))

      isochrones = read_isochrones(args.age, args.z, Gmag_max = args.gmag_max)
      isochrones_cmd = combine_isochrones(isochrones, cmd_broadening = args.cmd_broadening)

      labels_cmd = cmd_cleaning(Gaia_table, isochrones_cmd, clipping_sigma = args.clipping_sigma_cmd, plots = args.plots, plot_name = Gaia_path+'CMD_selection.png')

      CMD_rejected = Gaia_table[~labels_cmd]
      Gaia_table = Gaia_table[labels_cmd]

      has_pm_too_faint = too_faint.loc[:, ['pmra', 'pmdec', 'parallax']].notnull().all(axis = 1)

      too_faint_wpm = too_faint[has_pm_too_faint]
      too_faint = too_faint[~has_pm_too_faint]

      Gaia_table, Gaia_table_rejected_pm, too_faint_wpm, too_faint_wpm_rejected_pm = pm_cleaning_GMM_recursive(Gaia_table, ['pmra', 'pmdec', 'parallax'], alt_table = too_faint_wpm, data_0 = [args.pmra, args.pmdec, args.parallax], n_components = args.pm_n_components, clipping_prob = args.clipping_prob_pm, plots = args.plots, plot_name = Gaia_path+'PM_selection')

      Gaia_members = Gaia_table.append(too_faint_wpm)
      Gaia_not_members = CMD_rejected.append(too_faint_wpm_rejected_pm, sort=True).append(Gaia_table_rejected_pm, sort=True)

   else:
      """
      If distance is not defined, the code just perform the selection in the PM-parallax space.
      """
      has_pm = Gaia_table.loc[:, ['pmra', 'pmdec', 'parallax']].notnull().all(axis = 1)
      too_faint = Gaia_table[~has_pm]
      Gaia_table = Gaia_table[has_pm]

      Gaia_members, Gaia_not_members = pm_cleaning_GMM_recursive(Gaia_table, ['pmra', 'pmdec', 'parallax'], data_0 = [args.pmra, args.pmdec, args.parallax], n_components = args.pm_n_components, clipping_prob = args.clipping_prob_pm, plots = args.plots, plot_name = Gaia_path+'PM_selection')

   """
   Save Gaia tables
   """ 
   Gaia_members.to_csv(Gaia_members_table_filename, index = False)
   Gaia_not_members.to_csv(Gaia_not_members_table_filename, index = False)
   too_faint.to_csv(Gaia_faint_table_filename, index = False)


   """
   The script tries to load an existing HST table, otherwise it will download it from the MAST archive.
   """
   try:
      obs_table = Table.from_pandas(pd.read_csv(HST_obs_table_filename))
      data_products_by_obs = Table.from_pandas(pd.read_csv(HST_data_table_products_filename))
   except:
      obs_table, data_products_by_obs = search_mast(args.ra, args.dec, args.width, args.height, filters = args.hst_filters, t_exptime_min = args.hst_integration_time_min, t_exptime_max = args.hst_integration_time_max, time_baseline = args.time_baseline*365)
      obs_table.to_pandas().to_csv(HST_obs_table_filename, index = False)
      data_products_by_obs.to_pandas().to_csv(HST_data_table_products_filename, index = False)

   """
   Plot results and find Gaia stars within HST fields
   """
   obs_table_gaia = plot_fields(Gaia_members, obs_table.copy(), name = base_path+args.name+'_footprint.png')

   """
   Save HST tables
   """
   obs_table.to_pandas().to_csv(HST_obs_table_filename, index = False)
   data_products_by_obs.to_pandas().to_csv(HST_data_table_products_filename, index = False)

   print('\n')

   print(obs_table)

   print('\n')

   print(data_products_by_obs)

   print('\n')

   """
   Ask whether the user wish to download the available HST images 
   """
   download = str2bool(input('Do you want to download these products?\n'))

   if download:
      hst_images = download_HST_images(data_products_by_obs, path = HST_path)

   """
   Print a summary with the location of files
   """
   summary_gaia = 'Gaia tables can be found in %s'%Gaia_path
   summary_hst = 'HST data can be found in %s'%HST_path
   
   print('\n')
   print(' SUMMARY '.center(len(summary_gaia), '*'))
   print(summary_gaia)
   print('-Gaia table: %s'%args.name+'.csv')
   print('-Gaia members: %s'%args.name+'_members.csv')
   print('-Gaia not members: %s'%args.name+'_not_members.csv')
   print('-Gaia not classified: %s'%args.name+'_faint.csv')
   print('-'*len(summary_gaia))

   print(summary_hst)
   print('-HST observations: %s'%args.name+'_obs.csv')
   print('-HST products: %s'%args.name+'_data_products.csv')
   
   print('*'*len(summary_gaia)+'\n')

   print('\n Bye!\n')

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)

"""
Andres del Pino Molina
"""
