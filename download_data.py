#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys
import os
import subprocess

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from sklearn.utils.validation import check_is_fitted

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.time import Time
from astropy.coordinates import match_coordinates_sky
from astropy.wcs.utils import proj_plane_pixel_scales

from scipy import stats


def manual_select_from_cmd(color, mag):

   class SelectFromCollection(object):
      """
      Select indices from a matplotlib collection using `LassoSelector`.
      """
      def __init__(self, ax, collection, alpha_other=0.2):
         self.canvas = ax.figure.canvas
         self.collection = collection
         self.alpha_other = alpha_other

         self.xys = collection.get_offsets()
         self.Npts = len(self.xys)

         # Ensure that we have separate colors for each object
         self.fc = collection.get_facecolors()
         if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
         elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

         lineprops = {'color': 'k', 'linewidth': 1, 'alpha': 0.8}
         self.lasso = LassoSelector(ax, onselect=self.onselect, lineprops=lineprops)
         self.ind = []

      def onselect(self, verts):
         path = Path(verts)
         self.ind = np.nonzero(path.contains_points(self.xys))[0]
         self.selection = path.contains_points(self.xys)
         self.fc[:, -1] = self.alpha_other
         self.fc[self.ind, -1] = 1
         self.collection.set_facecolors(self.fc)
         self.canvas.draw_idle()

      def disconnect(self):
         self.lasso.disconnect_events()
         self.fc[:, -1] = 1
         self.collection.set_facecolors(self.fc)
         self.canvas.draw_idle()


   #subplot_kw = dict(xlim=(color.min()-0.1, color.max()+0.1), ylim=(mag.max()+0.05, mag.min()-0.05), autoscale_on=False)
   #fig, ax = plt.subplots(subplot_kw=subplot_kw)

   subplot_kw = dict(autoscale_on = False)
   fig, ax = plt.subplots(subplot_kw = subplot_kw)

   pts = ax.scatter(color, mag, s=2)
   try:
      ax.set_xlim(np.nanmin(color)-0.1, np.nanmax(color)+0.1)
      ax.set_ylim(np.nanmax(mag)+0.05, np.nanmin(mag)-0.05)
   except:
      pass

   ax.grid()
   ax.set_xlabel(color.name)
   ax.set_ylabel(mag.name)

   selector = SelectFromCollection(ax, pts)

   def accept(event):
      if event.key == "enter":
         selector.disconnect()
         plt.close('all')

   fig.canvas.mpl_connect("key_press_event", accept)
   ax.set_title("Press enter to accept selected points.")

   plt.show()

   input("Once your selection is made, please press enter to continue.")
   try:
      return selector.selection
   except:
      return [True]*len(mag)


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
   Select stars with good astrometry in Gaia.
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
   Select stars with good photometry in Gaia.
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
   This routine removes jobs from the Gaia archive server.
   """

   list_jobs = []
   for job in Gaia.list_async_jobs():
      list_jobs.append(job.get_jobid())
   
   Gaia.remove_jobs(list_jobs)


def gaia_log_in():
   """
   This routine log in to the Gaia archive.
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
   This routine launch the query to the Gaia archive.
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
   This routine generates logarithmic spaced bins for G magnitude.
   """

   bins_mag = (1.0 + max_mag - np.logspace(np.log10(1.), np.log10(1. + max_mag - min_mag), num = int(n_bins_mag), endpoint = True))

   return bins_mag


def gaia_multi_query_run(args):
   """
   This routine pipes gaia_query into multiple threads.
   """

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

   mag_nodes = get_mag_bins(gmag_min, gmag_max, num_nodes)

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


def plot_fields(Gaia_stars, obs_table, name = 'test.png'):
   """
   This routine plots the fields and select Gaia stars within them.
   """

   from matplotlib.patches import Polygon
   from matplotlib.collections import PatchCollection
   from shapely.geometry.polygon import Polygon as shap_polygon
   from shapely.geometry import Point

   def deg_to_hms(lat, even = True):
      from astropy.coordinates import Angle
      from astropy import units as u
      angle = Angle(lat, unit = u.deg)
      if even:
         if lat%30:
            string =''
         else:
            string = angle.to_string(unit=u.hour)
      else:
         string = angle.to_string(unit=u.hour)
      return string

   def deg_to_dms(lon):
      from astropy.coordinates import Angle
      from astropy import units as u
      angle = Angle(lat, unit = u.deg)
      string = angle.to_string(unit=u.degree)
      return string

   def coolwarm(filter, alpha):
      color = [rgb for rgb in plt.cm.coolwarm(255 * (float(filter) - 555) / (850-555))]
      color[-1] = alpha
      return color

   Gaia_stars['parent_obsid'] = ""
   
   fig, ax = plt.subplots(1,1, figsize = (5.5, 5.5))
   patches = []
   ecs = []
   previous_obsid = []
   gaia_stars_per_obs = []
   
   filter_range = [float(s.replace('F', '').replace('W', '').replace('LP', '')) for s in obs_table.filters]
   
   for index_obs, (footprint_str, obsid, filter) in obs_table.loc[:, ['s_region', 'obsid', 'filters']].iterrows():
      cli_progress_test(index_obs+1, len(obs_table))

      idx_Gaia_in_field = []
      gaia_stars_per_poly = [] 
      list_coo = footprint_str.split('POLYGON')[1::]

      for poly in list_coo:
         try:
            poly = list(map(float, poly.split()))
         except:
            poly = list(map(float, poly.split('J2000')[1::][0].split()))

         tuples_list = [(ra % 360, dec) for ra, dec in zip(poly[0::2], poly[1::2]) if (ra%360 < Gaia_stars.ra.max()) & (ra%360 > Gaia_stars.ra.min()) & (dec < Gaia_stars.dec.max()) & (dec > Gaia_stars.dec.min())]

         if len(tuples_list) > 0:
            polygon = Polygon(tuples_list, True)
            ecs.append(coolwarm(filter.replace(r'F', '').replace(r'W', '').replace('LP', ''), 1))

            footprint =  shap_polygon(tuples_list)

            star_counts = 0
            for idx, ra, dec in zip(Gaia_stars.index, Gaia_stars.ra, Gaia_stars.dec):
               if Point(ra, dec).within(footprint):
                  idx_Gaia_in_field.append(idx)
                  star_counts += 1

            gaia_stars_per_poly.append(star_counts)
            patches.append(polygon)

            annotation_coo = [round(max(tuples_list)[0], 2), round(max(tuples_list)[1], 2)]

            if annotation_coo in previous_obsid:
               annotation_coo[1] += 0.01

            ax.annotate(obsid, xy=(annotation_coo[0], annotation_coo[1]), xycoords='data', color = coolwarm(filter.replace(r'F', '').replace(r'W', '').replace('LP', ''), 1))

            previous_obsid.append(annotation_coo)

      gaia_stars_per_obs.append(sum(gaia_stars_per_poly))

      Gaia_stars.loc[idx_Gaia_in_field, 'parent_obsid'] = Gaia_stars.loc[idx_Gaia_in_field, 'parent_obsid'].astype(str) + '%s '%obsid

   print('\n')

   obs_table['gaia_stars_per_obs'] = gaia_stars_per_obs
   
   p = PatchCollection(patches, alpha = 1, ec = ecs, fc = 'None', antialiased = True, lw = 1, zorder = 4)
   ax.add_collection(p)
   ax.plot(Gaia_stars.ra, Gaia_stars.dec, '.', color = '0.4', ms = 0.75, zorder = 1)
   ax.set_xlim(Gaia_stars.ra.max(), Gaia_stars.ra.min())
   ax.set_ylim(Gaia_stars.dec.min(), Gaia_stars.dec.max())
   ax.grid()

   ax.set_xlabel(r'$\alpha$ [deg]')
   ax.set_ylabel(r'$\delta$ [deg]')

   plt.savefig(name, bbox_inches='tight')

   with plt.rc_context(rc={'interactive': False}):
      plt.gcf().show()

   return Gaia_stars, obs_table


def search_mast(ra, dec, width, height, filters = 'any', t_exptime_min = 50, t_exptime_max = 2500, date_second_epoch = 57531.0, time_baseline = 3650):
   """
   This routine search for HST observations in MAST at a given position.
   """

   from astroquery.mast import Catalogs
   from astroquery.mast import Observations

   ra1 = ra - width / 2 - 0.056*np.cos(np.deg2rad(dec))
   ra2 = ra + width / 2 + 0.056*np.cos(np.deg2rad(dec))
   dec1 = dec - height / 2 - 0.056
   dec2 = dec + height / 2 + 0.056

   t_max = date_second_epoch - time_baseline
   
   if filters == ['any']:
      filters=['F555W','F606W','F775W','F814W','F850LP']
   elif type(filters) is not list:
      filters = [filters]

   obs_table = Observations.query_criteria(dataproduct_type=["image"], obs_collection=["HST"], s_ra=[ra1, ra2], s_dec=[dec1, dec2], instrument_name=['ACS/WFC', 'WFC3/UVIS'], t_max=[0, t_max], filters = filters)

   data_products_by_obs = search_data_products_by_obs(obs_table)

   #Pandas is easier:
   obs_table = obs_table.to_pandas()
   data_products_by_obs = data_products_by_obs.to_pandas()

   obs_table.to_csv('obs_table.csv', index = False)
   data_products_by_obs.to_csv('data_products_by_obs.csv', index = False)

   #obs_table = obs_table.merge(data_products_by_obs.groupby(['parent_obsid'])['parent_obsid'].count().rename_axis('obsid').rename('n_exp'), on = ['obsid'])
   obs_table = obs_table.merge(data_products_by_obs.groupby(['obs_id'])['obs_id'].count().rename('n_exp'), on = ['obs_id'])

   obs_table['i_exptime'] = obs_table['t_exptime'] / obs_table['n_exp']

   #For convenience we add an extra column with the time baseline
   obs_time = Time(obs_table['t_max'], format='mjd')
   obs_time.format = 'iso'
   obs_time.out_subfmt = 'date'
   obs_table['obs_time'] = obs_time
   
   obs_table['t_baseline'] = round((date_second_epoch - obs_table['t_max']) / 365.2422, 2)
   obs_table['filters'] = obs_table['filters'].str.strip('; CLEAR2L CLEAR1L')

   #data_products_by_obs = data_products_by_obs.merge(obs_table.loc[:, ['obsid', 'i_exptime', 'filters', 't_baseline', 's_ra', 's_dec']].rename(columns={'obsid':'parent_obsid'}), on = ['parent_obsid'])
   data_products_by_obs = data_products_by_obs.merge(obs_table.loc[:, ['obs_id', 'i_exptime', 'filters', 't_baseline', 's_ra', 's_dec']], on = ['obs_id'])

   #We select by individual exp time:
   obs_table = obs_table.loc[(obs_table.i_exptime > t_exptime_min) & (obs_table.i_exptime < t_exptime_max)]
   data_products_by_obs = data_products_by_obs.loc[(data_products_by_obs.i_exptime > t_exptime_min) & (data_products_by_obs.i_exptime < t_exptime_max)]
   
   return obs_table.astype({'obsid': 'int64'}).reset_index(drop = False), data_products_by_obs.astype({'parent_obsid': 'int64'}).reset_index(drop = False)


def search_data_products_by_obs(obs_table):
   """
   This routine search for images in MAST related to the given observations table.
   """
   
   from astroquery.mast import Observations

   data_products_by_obs = Observations.get_product_list(obs_table)

   return data_products_by_obs[data_products_by_obs['productSubGroupDescription'] == 'FLC']


def download_HST_images(data_products_by_obs, path = './'):
   """
   This routine downloads the selected HST images from MAST.
   """
   
   from astroquery.mast import Observations

   flc_images = Observations.download_products(data_products_by_obs[data_products_by_obs['productSubGroupDescription'] == 'FLC'], download_dir=path)

   return flc_images


def read_isochrones(Ages, Zs, gmag_max = -3.5):
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
         Max_mag = isochrone[:,3] <= gmag_max
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


def cmd_cleaning(table, isochrones_cmd, distance = None, AV = None, clipping_sigma = 3., plots = True, plot_name = ''):
   """
   This routine will clean the CMD by rejecting stars more than intrinsic_broadening + clipping_sigma away from the used isochrone(s).
   """

   from shapely.geometry.point import Point
   from shapely.affinity import scale
   from descartes import PolygonPatch
   
   if AV is None:
      table['AV'] = get_AV_map(table.loc[:, ['ra','dec']])

   try:
      table.distance
   except:
      table['distance'] = distance

   member_cmd = pd.Series(index = table.index)

   table[['gmag_0','bpmag_0','rpmag_0']] = table.loc[:, ['gmag','bpmag','rpmag']] - (simple_reddening_correction(table.loc[:, 'AV']) + np.expand_dims((5.*np.log10((1.5+table.loc[:, 'distance'])*1e3)-5.), 1))

   has_cmd = table.loc[:, ['gmag', 'bpmag', 'rpmag']].notnull().all(axis = 1)

   print('Selecting stars in the cmd.')

   stars_cmd = [scale(Point((stars_color, stars_mag)).buffer(1), xfact=stars_color_error, yfact=stars_mag_error) for stars_color, stars_mag, stars_color_error, stars_mag_error in zip(table.loc[has_cmd, 'bpmag_0']-table.loc[has_cmd, 'rpmag_0'], table.loc[has_cmd, 'gmag_0'], clipping_sigma*np.sqrt(table.loc[has_cmd, 'bpmag_error']**2+table.loc[has_cmd, 'rpmag_error']**2), clipping_sigma*table.loc[has_cmd, 'gmag_error'])]

   labels_cmd = np.zeros_like(stars_cmd, dtype=bool)
   for ii, star in enumerate(stars_cmd):
      cli_progress_test(ii+1, len(stars_cmd))
      labels_cmd[ii] = star.intersects(isochrones_cmd)
   
   member_cmd.loc[has_cmd] = labels_cmd

   if plots == True:
      plt.close('all')

      fig = plt.figure(1)
      ax = fig.add_subplot(111)
      try:
         patch = PolygonPatch(isochrones_cmd, facecolor='orange', lw=0, alpha = 0.5, zorder = 2)
         ax.add_patch(patch)
      except:
         pass
      ax.plot((table.bpmag_0-table.rpmag_0).loc[member_cmd == True] , table.gmag_0.loc[member_cmd == True] , 'b.', label = 'selected', ms = 1., zorder = 1)
      ax.plot((table.bpmag_0-table.rpmag_0).loc[member_cmd == False] , table.gmag_0.loc[member_cmd == False] , 'k.', label = 'rejected', ms = 0.5, zorder = 0, alpha = 0.5)
      ax.set_ylim([table.gmag_0.loc[member_cmd == True].max()+1, table.gmag_0.loc[member_cmd == True].min()-1.])
      ax.set_xlim([(table.bpmag_0-table.rpmag_0).min()-1., (table.bpmag_0-table.rpmag_0).max()+1.])      
      ax.set_xlabel(r'$G_{BP}-G_{RP}$')
      ax.set_ylabel(r'$G$')
      plt.legend()
      plt.savefig(plot_name, bbox_inches='tight')

   return member_cmd


def create_dir(path):
   """
   This routine creates directories.
   """
   
   try:
      os.mkdir(path)
   except OSError:  
      print ("Creation of the directory %s failed" % path)
   else:  
      print ("Successfully created the directory %s " % path)


def get_AV_map(table):
   """
   This routine downloads the reddening maps.
   """

   from dustmaps.sfd import SFDQuery
   from astropy.coordinates import SkyCoord
   import astropy.units as u

   print('Obtaining AV map.')

   icrs = SkyCoord(ra = np.array(table.ra)*u.deg, dec = np.array(table.dec)*u.deg, frame = 'icrs')

   sfd = SFDQuery()
   
   AV = 3.1*sfd(icrs)  # multiply by 0.86 if you want to use Schlafly & Finkbeiner 2011 (ApJ 737, 103)

   return AV


def simple_reddening_correction(AV):
   """
   This routine returns Delta(G), Delta(BP), Delta(RP) in Gaia filters from Delta(V).
   """

   Ax = np.tile([0.85926, 1.06794, 0.65199], (len(AV),1)) * np.expand_dims(AV, axis=1)

   return Ax


def members_prob(table, clf, vars, clipping_prob = 3, data_0 = None):
   """
   This routine will find probable members through scoring of a passed model (clf).
   """

   has_vars = table.loc[:, vars].notnull().all(axis = 1)

   data = table.loc[has_vars, vars]
   
   clustering_data = table.loc[has_vars, 'clustering_data'] == 1

   results = pd.DataFrame(columns = ['member_clustering_prob', 'member_clustering'], index = table.index)

   if (clustering_data.sum() > 1):

      if data_0 is None:
         data_0 = data.loc[clustering_data, vars].median().values

      data -= data_0

      data_std = data.loc[clustering_data, :].std().values

      clf.fit(data.loc[clustering_data, :] / data_std)

      log_prob = clf.score_samples(data / data_std)
      prob_clf = np.abs(np.median(log_prob[clustering_data]) - log_prob)
      label_clf = prob_clf <= clipping_prob*np.std(log_prob[clustering_data])

      results.loc[has_vars, 'member_clustering_prob'] = prob_clf
      results.loc[has_vars, 'member_clustering'] = label_clf

   return results


def pm_cleaning_GMM_recursive(table, vars, alt_table = None, data_0 = None, n_components = 1, covariance_type = 'full', clipping_prob = 3, plots = True, verbose = False, plot_name = ''):
   """
   This routine iteratively find members using a Gaussian mixture model.
   """

   from sklearn import mixture
   
   table['real_data'] = True
   
   try:
      table['clustering_data']
   except:
      table['clustering_data'] = 1

   if alt_table is not None:
      alt_table['real_data'] = False
      alt_table['clustering_data'] = 0
      table = pd.concat([table, alt_table], ignore_index = True, sort=True)

   clf = mixture.GaussianMixture(n_components = n_components, covariance_type = covariance_type, means_init = np.zeros((n_components, len(vars))))

   convergence = False
   iteration = 0
   while not convergence:
      if verbose:
         print("\rIteration %i, %i objects remain."%(iteration, table.clustering_data.sum()))

      clust = table.loc[:, vars+['clustering_data']]

      if iteration > 3:
         data_0 = None

      fitting = members_prob(clust, clf, vars, clipping_prob = clipping_prob,  data_0 = data_0)
      
      table['member_clustering'] = fitting.member_clustering
      table['member_clustering_prob'] = fitting.member_clustering_prob
      table['clustering_data'] = (table.clustering_data == 1) & (fitting.member_clustering == 1)  & (table.real_data == 1)
      
      if (iteration > 999):
         convergence = True
      elif iteration > 0:
         convergence = fitting.equals(previous_fitting)

      previous_fitting = fitting.copy()
      iteration += 1

   if plots == True:
      plt.close('all')
      fig, ax1 = plt.subplots(1, 1)
      ax1.plot(table.loc[table.real_data == 1 ,vars[0]], table.loc[table.real_data == 1 , vars[1]], 'k.', ms = 0.5, zorder = 0)
      ax1.scatter(table.loc[table.clustering_data == 1 ,vars[0]].values, table.loc[table.clustering_data == 1 ,vars[1]].values, c = table.loc[table.clustering_data == 1, 'member_clustering_prob'].values, s = 1, zorder = 1)
      ax1.set_xlabel(r'$\mu_{\alpha*}$')
      ax1.set_ylabel(r'$\mu_{\delta}$')
      ax1.set_xlim(table.loc[table.real_data == 1 , vars[0]].mean()-5*table.loc[table.real_data == 1 , vars[0]].std(), table.loc[table.real_data == 1 , vars[0]].mean()+5*table.loc[table.real_data == 1 , vars[0]].std())
      ax1.set_ylim(table.loc[table.real_data == 1 , vars[1]].mean()-5*table.loc[table.real_data == 1 , vars[1]].std(), table.loc[table.real_data == 1 , vars[1]].mean()+5*table.loc[table.real_data == 1 , vars[1]].std())      
      ax1.grid()
      plt.savefig(plot_name, bbox_inches='tight')
      plt.close('all')

   if alt_table is not None:
      return fitting.loc[table.real_data == 1, 'member_clustering'], fitting.loc[table.real_data == 0, 'member_clustering']
   else:
      return fitting.member_clustering


def remove_file(file_name):
   """
   This routine removes files
   """

   try:
      os.remove(file_name)
   except:
      pass


def applied_pert(XYmqxy_filename):
   """
   This routine will read the XYmqxy file and find whether the PSF perturbation worked . If file not found then False.
   """

   try:
      f = open(XYmqxy_filename, 'r')
      perts = []
      for index, line in enumerate(f):
         if '# CENTRAL PERT PSF' in line:
            for index, line in enumerate(f):
               if len(line[10:-1].split()) > 0:
                  perts.append([float(pert) for pert in line[10:-1].split()])
               else:
                  break
            f.close()
            break
      if len(perts) > 0:
         return np.array(perts).ptp() != 0
      else:
         print('CAUTION: No information about PSF perturbation found in %s!'%XYmqxy_filename)
         return True
   except:
      return False


def get_fmin(i_exptime):
   """
   This routine returns the best value for FMIN to be used in hst1pass execution based on the integration time
   """
   
   # These are the values for fmin at the given lower_exptime and upper_exptime
   lower_fmin, upper_fmin = 1000, 10000
   lower_exptime, upper_exptime = 50, 500

   return max((int(lower_fmin + (upper_fmin - lower_fmin) * (i_exptime - lower_exptime) / (upper_exptime - lower_exptime)), 1000))


def hst1pass(data_products_by_obs, HST_obs_to_use, HST_path, force_fmin = None, force_hst1pass = True, remove_previous_XYmqxy = True, verbose = True):
   """
   This routine will execute hst1pass Fortran routine using the correct arguments.
   """
   
   #Define HST servicing missions times
   t_sm3 = Time('2002-03-12').mjd
   t_sm4 = Time('2009-05-24').mjd

   for HST_image_obsid in [HST_obs_to_use] if not isinstance(HST_obs_to_use, list) else HST_obs_to_use:
      for index_image, (obs_id, HST_image, HST_image_obsid) in data_products_by_obs.loc[data_products_by_obs['parent_obsid'] == HST_image_obsid, ['obs_id', 'productFilename', 'parent_obsid']].iterrows():

         if verbose:
            print('Finding sources in',HST_image)

         output_dir = HST_path+'mastDownload/HST/'+obs_id+'/'
         HST_image_filename = output_dir+HST_image
         XYmqxy_filename = HST_image_filename.split('.fits')[0]+'.XYmqxy'
         
         if not os.path.isfile(XYmqxy_filename) or force_hst1pass:
            hdul = fits.open(HST_image_filename)
            instrument = hdul[0].header['INSTRUME']
            detector = hdul[0].header['DETECTOR']
            try:
               filter = [filter for filter in [hdul[0].header['FILTER1'], hdul[0].header['FILTER2']] if 'CLEAR' not in filter][0]
            except:
               filter = hdul[0].header['FILTER']
            t_max = hdul[0].header['EXPEND']
            sm = ''
            if instrument == 'ACS':
               if (t_max > t_sm3) & (t_max < t_sm4):
                  sm = '_SM3'
               elif (t_max > t_sm4):
                  sm = '_SM4'

            if force_fmin is None:
               fmin = get_fmin(hdul[0].header['EXPTIME'])
            else:
               fmin = force_fmin
            
            if verbose:
               print('  Using FMIN =', fmin)

            psf_filename = './Auxiliary/STDPSFs/%s%s/PSFSTD_%s%s_%s%s.fits'%(instrument, detector, instrument, detector, filter, sm)
            gdc_filename = './Auxiliary/STDGDCs/%s%s/STDGDC_%s%s_%s.fits'%(instrument, detector, instrument, detector, filter)

            if remove_previous_XYmqxy:
               #Try removing the XYmqxy?
               remove_file(XYmqxy_filename)

            pert_grid = 5
            while not applied_pert(XYmqxy_filename) and (pert_grid > 0):
               if verbose:
                  print('    Trying PERT PSF = %sx%s'%(pert_grid, pert_grid))

               # In principle, we should be abble to fine-tune FMIN knowing the exptime and the filter. Maybe for the next version.
               bashCommand = "./Auxiliary/hst1pass.e HMIN=5 FMIN=%s PMAX=999999 GDC=%s PSF=%s PERT%i=AUTO OUT=XYmqxy OUTDIR=%s %s"%(fmin, gdc_filename, psf_filename, pert_grid, output_dir, HST_image_filename)

               process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
               output, error = process.communicate()
               pert_grid -= 1


def find_stars_to_align(stars_catalog, HST_image_filename):
   """
   This routine will find which stars from stars_catalog within and HST image.
   """

   from shapely.geometry.polygon import Polygon as shap_polygon
   from shapely.geometry import Point
   from shapely.ops import unary_union

   HST_image = HST_image_filename.split('/')[-1].split('.fits')[0]
   
   hdu = fits.open(HST_image_filename)
   
   if 'HST_image' not in stars_catalog.columns:
      stars_catalog['HST_image'] = ""

   idx_Gaia_in_field = []
   footprint = []
   for ii in [2, 5]:
      wcs = WCS(hdu[ii].header)
      footprint_chip = wcs.calc_footprint()
      
      #We add 10 arcsec of HST pointing error to the footprint to ensure we have all the stars.
      center_chip = np.mean(footprint_chip, axis = 0)

      footprint_chip[np.where(footprint_chip[:,0] < center_chip[0]),0] -= 0.0028*np.cos(np.deg2rad(center_chip[1]))
      footprint_chip[np.where(footprint_chip[:,0] > center_chip[0]),0] += 0.0028*np.cos(np.deg2rad(center_chip[1]))

      footprint_chip[np.where(footprint_chip[:,1] < center_chip[1]),1] -= 0.0028
      footprint_chip[np.where(footprint_chip[:,1] > center_chip[1]),1] += 0.0028

      tuples_coo = [(ra % 360, dec) for ra, dec in zip(footprint_chip[:, 0], footprint_chip[:, 1])]
      footprint.append(shap_polygon(tuples_coo))
   
   footprint = unary_union(footprint)

   for idx, ra, dec in zip(stars_catalog.index, stars_catalog.ra, stars_catalog.dec):
      if Point(ra, dec).within(footprint):
         idx_Gaia_in_field.append(idx)

   stars_catalog.loc[idx_Gaia_in_field, 'HST_image'] = stars_catalog.loc[idx_Gaia_in_field, 'HST_image'].astype(str) + '%s '%HST_image

   return stars_catalog


def check_mat(mat_filename, iteration, alpha = 0.01, plots = True):
   """
   This routine will read the transformation file MAT and provide a quality flag based on how Gaussian the transformation is.
   """

   mat = np.loadtxt(mat_filename, usecols = [6, 7])

   # Obtain the Shapiro–Wilk statistics
   stat, p = stats.shapiro(mat)
   
   valid = (p > alpha) & (mat.mean(axis = 0) < 1e-5).all() & (len(mat) > 10)

   if plots:
      plt.close()
      fig, ax1 = plt.subplots(1, 1)
      ax1.plot(mat[:,0], mat[:,1], '.', ms = 2, label = 'Gaia')
      ax1.axvline(x=0, linewidth = 0.75, color = 'k')
      ax1.axhline(y=0, linewidth = 0.75, color = 'k')
      ax1.set_xlabel('X_Gaia - X_HST [pixels]')
      ax1.set_ylabel('Y_Gaia - Y_HST [pixels]')
      ax1.grid()
      
      add_inner_title(ax1, 'Valid = %s'%valid, 1)
      
      plt.savefig(mat_filename.split('.MAT')[0]+'_MAT_%i.png'%iteration, bbox_inches='tight')
      plt.close()

   return valid


def xym2pm_Gaia(Gaia_HST_table, data_products_by_obs, HST_obs_to_use, HST_path, date_reference_second_epoch, only_use_members = False, force_pixel_scale = None, force_max_separation = None, force_use_sat = True, n_components = 1, clipping_prob = 6, min_stars_alignment = 100, plots = True, verbose = True, remove_previous_files = True, plot_name = ''):
   """
   This routine will execute xym2pm_Gaia Fortran routine using the correct arguments.
   """
   # TODO:
   # - Parallel execution

   convergence = False
   iteration = 0
   while not convergence:
      if verbose:
         print("\n-----------")
         print("Iteration %i"%(iteration))
         print("-----------")
      
      print(data_products_by_obs)
      print(HST_obs_to_use)
      lnks = []      
      for index_image, (obs_id, HST_image) in data_products_by_obs.loc[data_products_by_obs['parent_obsid'].isin([HST_obs_to_use] if not isinstance(HST_obs_to_use, list) else HST_obs_to_use), ['obs_id', 'productFilename']].iterrows():
         HST_image_filename = HST_path+'mastDownload/HST/'+obs_id+'/'+HST_image
         Gaia_HST_catalog_filename = HST_path+'Gaia_%s.ascii'%HST_image.split('.fits')[0]
         lnk_filename = HST_image_filename.split('.fits')[0]+'.LNK'
         mat_filename = HST_image_filename.split('.fits')[0]+'.MAT'

         if iteration == 0:
            Gaia_HST_table = find_stars_to_align(Gaia_HST_table, HST_image_filename)
            if (Gaia_HST_table.loc[Gaia_HST_table['HST_image'].str.contains(str(obs_id)), 'use_for_alignment'].sum() < min_stars_alignment):
               Gaia_HST_table.loc[Gaia_HST_table['HST_image'].str.contains(str(obs_id)), 'use_for_alignment'] = True

         Gaia_HST_table_field = Gaia_HST_table.loc[Gaia_HST_table['HST_image'].str.contains(str(obs_id)), :]

         if verbose:
            print("%s: Matching %i stars."%(HST_image, Gaia_HST_table_field.use_for_alignment.sum()))

         if iteration == 0:
            if remove_previous_files:
               remove_file(mat_filename)
               remove_file(lnk_filename)

         hdul = fits.open(HST_image_filename)
         t_max = hdul[0].header['EXPEND']
         ra_cent = hdul[0].header['RA_TARG']
         dec_cent = hdul[0].header['DEC_TARG']
         try:
            filter = [filter for filter in [hdul[0].header['FILTER1'], hdul[0].header['FILTER2']] if 'CLEAR' not in filter][0]
         except:
            filter = hdul[0].header['FILTER']

         t_baseline = (date_reference_second_epoch - t_max) / 365.2422

         if force_pixel_scale is None:
            # Infer pixel scale from the header
            pixel_scale = round(np.mean([proj_plane_pixel_scales(WCS(hdul[3].header)).mean(), proj_plane_pixel_scales(WCS(hdul[5].header)).mean()])*3600, 3)
         else:
            pixel_scale = force_pixel_scale

         pixel_scale_mas = 1e3 * pixel_scale

         f = open(Gaia_HST_catalog_filename, 'w+')
         f.write('# ')
         Gaia_HST_table_field.loc[:, ['ra', 'ra_error', 'dec', 'dec_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'gmag', 'use_for_alignment']].astype({'use_for_alignment': 'int32'}).to_csv(f, index = False, sep = ' ', na_rep = 0)
         f.close()

         # Here it goes the executable line. Input values can be fine-tuned here
         if force_use_sat:
            use_sat = ' USESAT+'
         else:
            use_sat = ''
            
         if force_max_separation is None:
            if iteration == 0:
               max_separation = 5.0
            else:
               max_separation = 5.0
               # Improvement: Use max_separation to do the match, but derive PM for all the stars.
               #max_separation = clipping_prob * Gaia_HST_table_field.loc[Gaia_HST_table_field.use_for_alignment == True, ['relative_hst_gaia_pmra', 'relative_hst_gaia_pmdec']].std().mean() / pixel_scale_mas * t_baseline

         bashCommand = "./Auxiliary/xym2pm_Gaia.e %s %s RACEN=%f DECEN=%f XCEN=5000.0 YCEN=5000.0 PSCL=%s SIZE=10000 DISP=%.1f TIME=2015.5%s"%(Gaia_HST_catalog_filename, HST_image_filename, ra_cent, dec_cent, pixel_scale, max_separation, use_sat)

         process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
         output, error = process.communicate()

         try:
            # Next are threshold rejection values for the MAT files.
            if (iteration > 0) & only_use_members:
               alpha = 1e-4
            else:
               alpha = 1e-32

            if check_mat(mat_filename, iteration, alpha = alpha):

               f = open(lnk_filename, 'r')
               header = [w.replace('m_hst', filter) for w in f.readline().rstrip().strip("# ").split(' ')]
               lnk = pd.read_csv(f, names=header, sep = '\s+', comment='#', na_values = 0.0).set_index(Gaia_HST_table_field.index)
               f.close()

               # Assign positional and mag error to each star based on QFIT parameter. 0.85 seems reasonable, although this may be tuned through an empirical function
               eradec_hst = lnk.q_hst.replace({0:lnk.q_hst.max()}) * pixel_scale_mas * 0.85

               lnk['relative_hst_gaia_pmra'] = -(lnk.x_gaia - lnk.xhst_gaia) * pixel_scale_mas / t_baseline
               lnk['relative_hst_gaia_pmdec'] = (lnk.y_gaia - lnk.yhst_gaia) * pixel_scale_mas / t_baseline

               # Notice the 1e3. xym2pm_Gaia.e takes the error in mas/yr but returns it in arcsec.
               lnk['relative_hst_gaia_pmra_error'] = np.sqrt(eradec_hst**2 + (1e3 * lnk.era_gaia)**2) / t_baseline
               lnk['relative_hst_gaia_pmdec_error'] = np.sqrt(eradec_hst**2 + (1e3 * lnk.edec_gaia)**2) / t_baseline

               lnk['%s_error'%filter] = lnk.q_hst.replace({0:lnk.q_hst.max()})

               lnks.append(lnk.loc[:, [filter, '%s_error'%filter, 'relative_hst_gaia_pmra', 'relative_hst_gaia_pmra_error', 'relative_hst_gaia_pmdec', 'relative_hst_gaia_pmdec_error']])

               if plots:
                  plt.close()
                  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex = False, sharey = False, figsize = (12, 3.5))
                  ax1.plot(lnk.pmra_gaia, lnk.relative_hst_gaia_pmra, '.', ms = 1)
                  ax1.plot(lnk.pmra_gaia, lnk.pmra_gaia, 'k-', linewidth = 0.75)
                  ax1.grid()

                  ax2.plot(lnk.pmdec_gaia, lnk.relative_hst_gaia_pmdec, '.', ms = 1)
                  ax2.plot(lnk.pmdec_gaia, lnk.pmdec_gaia, 'k-', linewidth = 0.75)
                  ax2.grid()

                  ax3.plot(lnk.pmra_gaia, lnk.pmdec_gaia, '.', ms = 1, label = 'Gaia')
                  ax3.plot(lnk.relative_hst_gaia_pmra, lnk.relative_hst_gaia_pmdec, '.', ms = 1, label = 'HST+Gaia')
                  ax3.set_xlabel('PMRA [mas/yr]')
                  ax3.set_ylabel('PMDEC [mas/yr]')
                  ax3.grid()
                  ax3.legend()
                  plt.subplots_adjust(wspace=0.25, hspace=0.1)
                  plt.savefig(lnk_filename.split('.LNK')[0]+'_LNK_%i.png'%iteration, bbox_inches='tight')
                  plt.close()
            else:
               print('   WARNING: Bad quality match. Skipping image.')
         except:
            print('  WARNING: No match found for %s.'%(HST_image))

      try:
         lnks = pd.concat(lnks, sort=True)
      except:
         print('WARNING: No match could be found for any of the images. Please try with other parameters.\nExiting now.')
         sys.exit(1)

      lnks_averaged = lnks.groupby(lnks.index).apply(weighted_avg_err)

      try:
         Gaia_HST_table.drop(columns = lnks_averaged.columns, inplace = True)
      except:
         pass

      Gaia_HST_table = Gaia_HST_table.join(lnks_averaged)

      # Membership selection
      if only_use_members:
         if iteration == 0:
            # Select stars in theCMD
            hst_filters = [col for col in lnks_averaged.columns if ('F' in col) & ('error' not in col) & ('std' not in col)]
            hst_filters.sort()
            if len(hst_filters) == 2:
               Gaia_HST_table['clustering_data'] =  manual_select_from_cmd((Gaia_HST_table[hst_filters[0]]-Gaia_HST_table[hst_filters[1]]).rename('%s - %s'%(hst_filters[0], hst_filters[1])), Gaia_HST_table[hst_filters[1]])
            else:
               for cmd_filter in hst_filters:
                  Gaia_HST_table['%s_clustering_data_cmd'%cmd_filter] =  manual_select_from_cmd((Gaia_HST_table['gmag']-Gaia_HST_table[cmd_filter]).rename('Gmag - %s'%cmd_filter), Gaia_HST_table['gmag'].rename('Gmag'))

               cmd_clustering_filters = [col for col in Gaia_HST_table.columns if '_clustering_data_cmd' in col]
               Gaia_HST_table['clustering_data'] = (Gaia_HST_table.loc[:, cmd_clustering_filters] == True).any(axis = 1)
               Gaia_HST_table.drop(columns = cmd_clustering_filters, inplace = True)
         else:
            Gaia_HST_table['clustering_data'] = True
         
         # Select stars in the PM space asuming spherical covariance (Reasonable for dSphs and globular clusters) 
         pm_clustering = pm_cleaning_GMM_recursive(Gaia_HST_table.copy(), ['relative_hst_gaia_pmra', 'relative_hst_gaia_pmdec'], data_0 = [0, 0], n_components = n_components, covariance_type = 'spherical', clipping_prob = clipping_prob, plots = plots, plot_name = '%s_%i'%(plot_name, iteration))
         new_use_for_alignment = pm_clustering & Gaia_HST_table.clustering_data

         if iteration > 4:
            print('\nWARNING: Max number of iterations reached: Something might have gone wrong!\nPlease check the results carefully.')
            convergence = True
         elif iteration > 0:
            convergence = np.array_equal(Gaia_HST_table['use_for_alignment'].values, new_use_for_alignment.values)

         if not convergence:
            Gaia_HST_table['use_for_alignment'] = new_use_for_alignment

         iteration += 1
      else:
         convergence = True
   
   Gaia_HST_table = Gaia_HST_table[Gaia_HST_table.relative_hst_gaia_pmdec.notnull() & Gaia_HST_table.relative_hst_gaia_pmra.notnull()]

   if plots:
      fig, (ax1, ax2) = plt.subplots(1, 2)
      ax1.plot(Gaia_HST_table.pmra, Gaia_HST_table.pmdec, '.', ms = 1, label = 'Gaia')
      ax1.plot(Gaia_HST_table.relative_hst_gaia_pmra, Gaia_HST_table.relative_hst_gaia_pmdec, '.', ms = 1, label = 'HST+Gaia')
      ax1.set_xlabel('PMRA [mas/yr]')
      ax1.set_ylabel('PMDEC [mas/yr]')
      ax1.grid()

      ax2.plot(Gaia_HST_table.gmag, np.sqrt(Gaia_HST_table.pmra_error**2+Gaia_HST_table.pmdec_error**2), '.', ms = 1, label = 'Gaia')
      ax2.plot(Gaia_HST_table.gmag, np.sqrt(Gaia_HST_table.relative_hst_gaia_pmra_error**2+Gaia_HST_table.relative_hst_gaia_pmdec_error**2), '.', ms = 1, label = 'HST+Gaia error')
      ax2.plot(Gaia_HST_table.gmag, np.sqrt(Gaia_HST_table.relative_hst_gaia_pmra_std**2+Gaia_HST_table.relative_hst_gaia_pmdec_std**2), '.', ms = 1, label = 'HST+Gaia std')
      ax2.set_ylim(0, np.sqrt(Gaia_HST_table.pmra_error**2+Gaia_HST_table.pmdec_error**2).max())
      ax2.set_xlabel('Gmag')
      ax2.set_ylabel('PM random errors [mas/yr]')
      ax1.grid()
      ax2.legend()
      plt.savefig('%s_random_errors.png'%(plot_name), bbox_inches='tight')

   return Gaia_HST_table


def weighted_avg_err(table):
   """
   Weighted average its error and the standard deviation.
   """

   var_cols = [x for x in table.columns if not '_error' in x]
   
   x_i = table.loc[:, var_cols]
   ex_i = get_errors(table, used_cols = var_cols)

   var_variance = 1./(1./ex_i**2).sum(axis = 0)

   var_avg = (x_i.div(ex_i.values**2)).sum(axis = 0) * var_variance.values
   
   var_std = x_i.std().add_suffix('_std')

   return pd.concat([var_avg, np.sqrt(var_variance[~var_variance.index.duplicated()]), var_std])


def get_errors(data, used_cols = None):
   """
   Obtain corresponding errors for the cols. If the error is not available it will assume the std of the entire distribution.
   """

   errors = pd.DataFrame(index = data.index)

   cols = data.columns
   if not used_cols:
      used_cols = [x for x in cols if not '_error' in x]

   for col in used_cols:
      if '%s_error'%col in cols:
         errors['%s_error'%col] = data['%s_error'%col]
      else:
         errors['%s_error'%col] = np.std(data[col])

   return errors


def absolute_pm(table, use = 'all'):
   """
   This routine computes the absolute PM just adding the absolute differences between Gaia and HST PMs.
   """
   
   pm_differences = table.loc[:, ['pmra', 'pmdec']] - table.loc[:, ['relative_hst_gaia_pmra', 'relative_hst_gaia_pmdec']].values
   pm_differences_error = np.sqrt(table.loc[:, ['pmra_error', 'pmdec_error']]**2 + table.loc[:, ['relative_hst_gaia_pmra', 'relative_hst_gaia_pmdec']].values**2)
   
   pm_differences_weighted = weighted_avg_err(pm_differences.join(pm_differences_error))

   table['hst_gaia_pmra'], table['hst_gaia_pmdec'] = table.relative_hst_gaia_pmra + pm_differences_weighted.pmra, table.relative_hst_gaia_pmdec + pm_differences_weighted.pmdec
   table['hst_gaia_pmra_error'], table['hst_gaia_pmdec_error'] = np.sqrt(table.relative_hst_gaia_pmra_error**2 + pm_differences_weighted.pmra_error**2), np.sqrt(table.relative_hst_gaia_pmdec_error**2 + pm_differences_weighted.pmdec_error**2)

   return table


def str2bool(v):
   """
   This routine converts ascii input to boolean.
   """
 
   if v.lower() in ('yes', 'true', 't', 'y'):
      return True
   elif v.lower() in ('no', 'false', 'f', 'n'):
      return False
   else:
      raise argparse.ArgumentTypeError('Boolean value expected.')


def cli_progress_test(current, end_val, bar_length=50):
   """
   Just a progress bar.
   """

   percent = float(current) / end_val
   hashes = '#' * int(round(percent * bar_length))
   spaces = ' ' * (bar_length - len(hashes))
   sys.stdout.write("\rProcessing: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
   sys.stdout.flush()


def add_inner_title(ax, title, loc, size=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke

    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at


def get_object_properties(args):
   """
   This routine will try to obtain all the required object properties from Simbad or from the user.
   """

   #Try to get object:
   if (args.ra is None) or (args.dec is None):
      try:
         from astroquery.simbad import Simbad
         from astropy.coordinates import SkyCoord
         from astropy import units as u
         customSimbad = Simbad()
         customSimbad.add_votable_fields('distance', 'propermotions', 'dim', 'fe_h')

         object_table = customSimbad.query_object(args.name)
         print('Object found:', object_table['MAIN_ID'])

         coo = SkyCoord(ra = object_table['RA'], dec = object_table['DEC'], unit=(u.hourangle, u.deg))

         args.ra = float(coo.ra.deg)
         args.dec = float(coo.dec.deg)

         #Try to get distances:
         if (args.distance is None) and args.use_members and not args.force_manual_cmd_cleaning:
            if (object_table['Distance_distance'].mask == False):
               if object_table['Distance_unit'] == 'Mpc':
                  args.distance = float(object_table['Distance_distance']*1e3)
               elif object_table['Distance_unit'] == 'kpc':
                  args.distance = float(object_table['Distance_distance'])
               elif object_table['Distance_unit'] == 'pc':
                  args.distance = float(object_table['Distance_distance']*1e-3)
            else:
               try:
                  args.distance = float(input('Distance to the object not found, please enter distance in kpc (Press enter to skip): '))
               except:
                  args.distance = None

         #Try to get radius
         if args.r is None:
            if (object_table['GALDIM_MAJAXIS'].mask == False):
               args.r = max(np.round(float(2. * object_table['GALDIM_MAJAXIS'] / 60.), 2), 0.1)
            else:
               try:
                  args.r = float(input('Search radius not defined, please enter the search radius in degrees (Press enter to adopt the default value of 1 deg): '))
               except:
                  args.r = 1.0

         #Try to get metallicity
         if (args.feh is None) and args.use_members and not args.force_manual_cmd_cleaning:
            if (object_table['Fe_H_Fe_H'].mask == False):
               args.feh = float(object_table['Fe_H_Fe_H'])
            else:
               try:
                  print('Metallicity [Fe/H] not defined, please enter one or more values for [Fe/H] separated by spaces (Press enter to adopt the default values [-3, 0]): ')
                  feh_s = input('[Fe/H]: ') or '-3 0'
                  args.feh = list(set([float(feh) for feh in feh_s.split() if np.isfinite(float(feh))]))
               except:
                  print("Non-numerical values in '%s'. Adopting default value [-3, 0]."%feh_s)
                  args.feh = [-3., 0.]
            args.z = np.round(0.019*10**np.array(args.feh), 6)

         #We try to get PMs:
         if args.pmra is None:
            if (object_table['PMRA'].mask == False):
               args.pmra = float(object_table['PMRA'])
            else:
               try:
                  args.pmra = float(input('PMRA not defined, please enter pmra in mas/yr (Press enter to ignore): '))
               except:
                  args.pmra = 0.0
      
         if args.pmdec is None:
            if (object_table['PMDEC'].mask == False):
               args.pmdec = float(object_table['PMDEC'])
            else:
               try:
                  args.pmdec = float(input('PMDEC not defined, please enter pmdec in mas/yr (Press enter to ignore): '))
               except:
                  args.pmdec = 0.0

         if args.parallax is None:
            args.parallax = 0.0

      except:
         if args.ra is None:
            args.ra = float(input('R.A. not defined, please enter R.A. in degrees: '))
         if args.dec is None:
            args.dec = float(input('Dec not defined, please enter Dec in degrees: '))
         if args.r is None:
            try:
               args.r = float(input('Search radius not defined, please enter the search radius in degrees (Press enter to adopt the default value of 1 deg): '))
            except:
               args.r = 1.0
   else:
      if args.pmra is None:
         args.pmra = 0.0
      if args.pmdec is None:
         args.pmdec = 0.0
      if args.parallax is None:
         args.parallax = 0.0
      if args.r is None:
         args.r = 1.0

   args.width = np.abs(args.r/np.cos(np.deg2rad(args.dec)))
   args.height = args.r

   name_coo = 'ra_%.3f_dec_%.3f_r_%.1f'%(args.ra, args.dec, args.r)

   if args.name is not None:
      args.name = args.name.replace(" ", "_")
      args.base_file_name = args.name+'_'+name_coo
   else:
      args.name = name_coo
      args.base_file_name = name_coo

   #The script creates directories and set files names
   args.base_path = './%s/'%(args.name)
   args.HST_path = args.base_path+'HST/'
   args.Gaia_path = args.base_path+'Gaia/'

   args.Gaia_raw_table_filename = args.Gaia_path + args.base_file_name+'_raw.csv'
   args.Gaia_table_filename = args.Gaia_path + args.base_file_name+'.csv'

   args.HST_obs_table_filename = args.HST_path + args.base_file_name+'_obs.csv'
   args.HST_data_table_products_filename = args.HST_path + args.base_file_name+'_data_products.csv'

   args.date_second_epoch = Time('%4i-%02i-%02iT00:00:00.000'%(args.date_second_epoch[2], args.date_second_epoch[0], args.date_second_epoch[1])).mjd
   args.date_reference_second_epoch = Time(args.date_reference_second_epoch).mjd

   print('\n')
   print(' USED PARAMETERS '.center(42, '*'))
   print('- (RA, Dec) = (%s, %s) deg.'%(round(args.ra, 5), round(args.dec, 5)))
   print('- (PM_RA, PM_Dec) = (%s, %s) m.a.s.'%(args.pmra, args.pmdec))
   print('- Distance = %s.'%('%s kpc'%args.distance if args.distance is not None else 'Not defined'))
   print('- [Fe/H] = %s.'%('%s dex'%([round(feh, 3) for feh in args.feh] if isinstance(args.feh, list) else round(args.feh, 3)) if args.feh is not None else 'Not defined'))
   print('- Radius = %s deg.'%args.r)
   print('*'*42+'\n')

   return args


def plot_results(table, plot_name_1 = 'output1.pdf', plot_name_2 = 'output2.pdf', plot_name_3 = 'output3.pdf'):
   """
   Plot results
   """

   pmra_lims = [table.hst_gaia_pmra.mean()-5*table.hst_gaia_pmra.std(), table.hst_gaia_pmra.mean()+5*table.hst_gaia_pmra.std()]
   pmdec_lims = [table.hst_gaia_pmdec.mean()-5*table.hst_gaia_pmdec.std(), table.hst_gaia_pmdec.mean()+5*table.hst_gaia_pmdec.std()]
   
   plt.close('all')
   
   fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex = False, sharey = False, figsize = (12, 3.5))

   ax1.plot(table.pmra[table.use_for_alignment == False], table.pmdec[table.use_for_alignment == False], 'k.', ms = 1, alpha = 0.35)
   ax1.plot(table.pmra[table.use_for_alignment == True], table.pmdec[table.use_for_alignment == True], 'k.', ms = 1)
   ax1.grid()
   ax1.set_xlabel(r'$\mu_{\alpha*}$ [m.a.s./yr.]')
   ax1.set_ylabel(r'$\mu_{\delta}$ [m.a.s./yr.]')
   try:
      ax1.set_xlim(pmra_lims)
      ax1.set_ylim(pmdec_lims)
   except:
      pass
   add_inner_title(ax1, 'Gaia' , loc=1)


   ax2.plot(table.hst_gaia_pmra[table.use_for_alignment == False], table.hst_gaia_pmdec[table.use_for_alignment == False], 'r.', ms =1, alpha = 0.35)
   ax2.plot(table.hst_gaia_pmra[table.use_for_alignment == True], table.hst_gaia_pmdec[table.use_for_alignment == True], 'r.', ms =1)
   ax2.grid()
   ax2.set_xlabel(r'$\mu_{\alpha*}$ [m.a.s./yr.]')
   ax2.set_ylabel(r'$\mu_{\delta}$ [m.a.s./yr.]')
   try:
      ax2.set_xlim(pmra_lims)
      ax2.set_ylim(pmdec_lims)
   except:
      pass
   add_inner_title(ax2, 'HST+Gaia' , loc=1)

   ax3.plot(table.gmag[table.use_for_alignment == False], np.sqrt(table.pmra_error[table.use_for_alignment == False]**2+table.pmdec_error[table.use_for_alignment == False]**2), 'k.', ms = 1, alpha = 0.35)
   ax3.plot(table.gmag[table.use_for_alignment == True], np.sqrt(table.pmra_error[table.use_for_alignment == True]**2+table.pmdec_error[table.use_for_alignment == True]**2), 'k.', ms = 1)

   ax3.plot(table.gmag[table.use_for_alignment == False], np.sqrt(table.hst_gaia_pmra_error[table.use_for_alignment == False]**2+table.hst_gaia_pmdec_error[table.use_for_alignment == False]**2), 'r.', ms = 1, alpha = 0.35)
   ax3.plot(table.gmag[table.use_for_alignment == True], np.sqrt(table.hst_gaia_pmra_error[table.use_for_alignment == True]**2+table.hst_gaia_pmdec_error[table.use_for_alignment == True]**2), 'r.', ms = 1)
   ax3.grid()
   try:
      ax3.set_ylim(0, np.sqrt(table.pmra_error**2+table.pmdec_error**2).max())
   except:
      pass
   ax3.set_xlabel('Gmag')
   ax3.set_ylabel(r'$\sqrt{\sigma(\mu_{\alpha*})^2 + \sigma(\mu_{\delta})^2 }$ [m.a.s./yr.]')

   plt.subplots_adjust(wspace=0.25, hspace=0.1)

   plt.savefig(plot_name_1, bbox_inches='tight')

   plt.close('all')

   fig2, (ax1, ax2) = plt.subplots(1,2, sharex = False, sharey = False, figsize = (12, 3.5))
   ax1.errorbar(table.pmra, table.hst_gaia_pmra, xerr=table.pmra_error, yerr=table.hst_gaia_pmra_error, fmt = '.', ms=1, color = '0.2', zorder = 1, alpha = 0.35, elinewidth = 0.5)
   ax1.plot([pmra_lims[0], pmra_lims[1]], [pmra_lims[0], pmra_lims[1]], 'r-', linewidth = 0.5)
   ax1.grid()
   ax1.set_xlabel(r'Gaia $\mu_{\alpha*}$ [m.a.s./yr.]')
   ax1.set_ylabel(r'HST + Gaia $\mu_{\alpha*}$ [m.a.s./yr.]')
   try:
      ax1.set_xlim(pmra_lims)
      ax1.set_ylim(pmra_lims)      
   except:
      pass

   ax2.errorbar(table.pmdec, table.hst_gaia_pmdec, xerr=table.pmdec_error, yerr=table.hst_gaia_pmdec_error, fmt = '.', ms=1, color = '0.2', zorder = 1, alpha = 0.35, elinewidth = 0.5)
   ax2.plot([pmdec_lims[0], pmdec_lims[1]], [pmdec_lims[0], pmdec_lims[1]], 'r-', linewidth = 0.5)
   ax2.grid()
   ax2.set_xlabel(r'Gaia $\mu_{\delta}$ [m.a.s./yr.]')
   ax2.set_ylabel(r'HST + Gaia $\mu_{\delta}$ [m.a.s./yr.]')
   try:
      ax2.set_xlim(pmdec_lims)
      ax2.set_ylim(pmdec_lims)      
   except:
      pass

   plt.savefig(plot_name_2, bbox_inches='tight')

   plt.close('all')


   plt.close('all')

   fig2, ax = plt.subplots(1,1, sharex = False, sharey = False, figsize = (5.5, 5.5))

   hst_filters = [col for col in table.columns if ('F' in col) & ('error' not in col) & ('std' not in col)]
   hst_filters.sort()
   if len(hst_filters) >= 2:
      color = (table[hst_filters[0]]-table[hst_filters[1]]).rename('%s - %s'%(hst_filters[0], hst_filters[1]))
      mag = table[hst_filters[1]]
   else:
      color = (table['gmag']-table[hst_filters[0]]).rename('Gmag - %s'%hst_filters[0])
      mag = table[hst_filters[0]]

   ax.plot(color[table.use_for_alignment == False], mag[table.use_for_alignment == False], 'k.', ms=1, alpha = 0.35)
   ax.plot(color[table.use_for_alignment == True], mag[table.use_for_alignment == True], 'k.', ms=1)

   ax.set_xlabel(color.name)
   ax.set_ylabel(mag.name)

   try:
      ax.set_xlim(np.nanmin(color)-0.1, np.nanmax(color)+0.1)
      ax.set_ylim(np.nanmax(mag)+0.25, np.nanmin(mag)-0.25)
   except:
      pass

   plt.savefig(plot_name_3, bbox_inches='tight')


def main(argv):  
   """
   Inputs
   """
   parser = argparse.ArgumentParser(description="This script derives proper motions (PM) combining HST and Gaia data.")
   parser.add_argument('name', type=str, default = None, help='Name for the Output table.')
   parser.add_argument('--use_members', type=str2bool, default=True, help='Whether to use only member stars for the epochs alignment or to use all available stars.')
   parser.add_argument('--date_second_epoch', type=int, nargs='+', default= [5, 23, 2016], help='Second epoch adquisition date. Default is Gaia DR2 (05-23-2016).')
   parser.add_argument('--date_reference_second_epoch', type=str, default= 'J2015.5', help='Second epoch reference date. Default is Gaia DR2 J2015.5.')
   parser.add_argument('--force_hst1pass', type=str2bool, default=False, help='Force the program to perform the sources search in the HST images. Default is False, which will use existing files if any.')
   parser.add_argument('--fmin', type=int, default= None, help='Minimum flux above the sky to substract a source in the HST image. Default is automatic from HST integration time.')
   parser.add_argument('--max_separation', type=float, default= None, help='Maximum allowed separation in pixels during the match between epochs. Default 5 pixels.')
   parser.add_argument('--pixel_scale', type=float, default= None, help='Pixel scale in arcsec/pixel used to compute the tangential plane during the match between epochs. Default is automatic from HST images.')
   parser.add_argument('--force_use_sat', type=str2bool, default=False, help='Force the program to use saturated stars during the match between epochs. Default is False.')
   parser.add_argument('--force_manual_cmd_cleaning', type=str2bool, default=False, help='Force the program to use manually selected stars from the CMD. Default False.')
   parser.add_argument('--min_stars_alignment', type=int, default = 100, help='Minimum number of stars per HST image to be used for the epochs alignment. Default 100.')
   parser.add_argument('--r', type=float, default = None, help='Radius of search in degrees (in R.A.).')
   parser.add_argument('--ra', type=float, default = None, help='Central R.A.')
   parser.add_argument('--dec', type=float, default = None, help='Central Dec.')
   parser.add_argument('--pmra', type=float, default=None, help='pmra in mas.')
   parser.add_argument('--pmdec', type=float, default=None, help='pmdec in mas.')
   parser.add_argument('--parallax', type=float, default=None, help='parallax in mas.')
   parser.add_argument('--distance', type=float, default = None, help='Distance in kpc.')
   parser.add_argument('--AV', type=float, default = None, help='Reddening in mag (AV).')
   parser.add_argument('--age', type=float, nargs='+', default= [12.], help='Age of the system in Gyr. Both, a single value or a range can be provided. Default is age within [8., 13.7].')
   parser.add_argument('--age_step', type=float, default= 0.1, help='Age resolution.')
   parser.add_argument('--age_mode', type=str, default= "discrete", help="If 'discrete', only the ages specified will be used. If 'continuous', ages between the max and min of --age will be used every --age_step")
   parser.add_argument('--feh', type=float, nargs='+', default= None, help='Metallicity ([Fe/H]) of the system. Both, a single value or a range can be provided. Default is range [Fe/H] within [-2.5, -0.5].')
   parser.add_argument('--cmd_broadening', type=float, default=0.1, help='CMD intrinsic color broadening in magnitudes. It is used to compute the maximum distance in color to a star as to consider it as possible bember of an isochrone population. Default is 0.1.')
   parser.add_argument('--clipping_sigma_cmd', type=float, default=6., help='Sigma used for clipping in the cmd. i.e. distance to the isochrone. Default is 3.')
   parser.add_argument('--extend_HB', type=str2bool, default=False, help='Whether to extend the HB of the isochrones in order to cover extremely low-metallicity populations.')
   parser.add_argument('--clipping_prob_pm', type=float, default=5., help='Sigma used for clipping pm and parallax. Default is 4.')
   parser.add_argument('--pm_n_components', type=int, default=1, help='Number of Gaussian componnents for pm and parallax clustering. Default is 1.')
   parser.add_argument('--hst_filters', type=str, nargs='+', default = ['any'], help='Required filter for the HST images.')
   parser.add_argument('--hst_integration_time_min', type=float, default = 50, help='Required integration time for the HST images.')
   parser.add_argument('--hst_integration_time_max', type=float, default = 450, help='Required integration time for the HST images.')
   parser.add_argument('--time_baseline', type=float, default = 1460, help='Minimum time baseline with respect to Gaia DR2 in days. Default 1460.')
   parser.add_argument('--gmag_max', type=float, default = 22.0, help='Fainter G magnitude')
   parser.add_argument('--gmag_min', type=float, default = 16.5, help='Brighter G magnitude')
   parser.add_argument('--clean_uwe', type = str2bool, default = True)
   parser.add_argument('--norm_uwe', type = str2bool, default = True)
   parser.add_argument('--clean_data', type = str2bool, default = False)
   parser.add_argument('--plots', type=str2bool, default=True, help='Create sanity plots. Default is True.')
   parser.add_argument('--verbose', type=str2bool, default=True, help='Program verbosity. Default True')
   args = parser.parse_args(argv)
   
   args = get_object_properties(args)

   """
   The script creates directories and set files names
   """

   create_dir(args.base_path)
   create_dir(args.HST_path)
   create_dir(args.Gaia_path)

   """
   The script tries to load an existing Gaia table, otherwise it will download it from the Gaia archive.
   """
   try:
      Gaia_table = pd.read_csv(args.Gaia_raw_table_filename)
   except:
      Gaia_table = incremental_query(args.ra, args.dec, args.width, args.height, gmag_min = args.gmag_min, gmag_max = args.gmag_max, clean_data = args.clean_data, norm_uwe = args.norm_uwe, output_name = args.name)
      Gaia_table.to_csv(args.Gaia_raw_table_filename, index = False)
   
   if args.use_members:
      if (args.distance is not None) and (args.force_manual_cmd_cleaning is False):
         """
         If distance is defined, the code will attempt to first select stars using isochrones.
         """
         isochrones = read_isochrones(args.age, args.z, gmag_max = args.gmag_max)
         isochrones_cmd = combine_isochrones(isochrones, cmd_broadening = args.cmd_broadening, extended_HB = args.extend_HB)
         Gaia_table['member_cmd_gaia'] = cmd_cleaning(Gaia_table.copy(), isochrones_cmd, distance = args.distance, AV = args.AV, clipping_sigma = args.clipping_sigma_cmd, plots = args.plots, plot_name = args.Gaia_path+'CMD_selection.png')

      else:
         Gaia_table['member_cmd_gaia'] = manual_select_from_cmd(Gaia_table.bp_rp, Gaia_table.gmag)

      """
      Perform the selection in the PM-parallax space.
      """
      Gaia_table['clustering_data'] = Gaia_table['member_cmd_gaia']
      Gaia_table['member_pm_gaia'] = pm_cleaning_GMM_recursive(Gaia_table.copy(), ['pmra', 'pmdec', 'parallax'], data_0 = [args.pmra, args.pmdec, args.parallax], n_components = args.pm_n_components, clipping_prob = args.clipping_prob_pm, plots = args.plots, plot_name = args.Gaia_path+'PM_selection')

      Gaia_table['use_for_alignment'] = (Gaia_table.loc[:, ['member_cmd_gaia', 'member_pm_gaia']] == True).all(axis = 1)
   else:
      Gaia_table['use_for_alignment'] = True

   """
   The script tries to load an existing HST table, otherwise it will download it from the MAST archive.
   """
   obs_table, data_products_by_obs = search_mast(args.ra, args.dec, args.width, args.height, filters = args.hst_filters, t_exptime_min = args.hst_integration_time_min, t_exptime_max = args.hst_integration_time_max, date_second_epoch = args.date_second_epoch, time_baseline = args.time_baseline)

   obs_table.to_csv(args.HST_obs_table_filename, index = False)
   data_products_by_obs.to_csv(args.HST_data_table_products_filename, index = False)

   """
   Plot results and find Gaia stars within HST fields
   """
   Gaia_table, obs_table = plot_fields(Gaia_table, obs_table, name = args.base_path+args.base_file_name+'_footprint.png')

   if len(obs_table) > 0:

      """
      Ask whether the user wish to download the available HST images 
      """
      print('Would you like to use the following HST observations?\n')
      print(obs_table.loc[:, ['obsid', 'filters', 'i_exptime', 'obs_time', 't_baseline', 'gaia_stars_per_obs']].to_string(index=False), '\n')

      print(Table.from_pandas(data_products_by_obs.loc[:, ['obsID', 'obs_id', 'description', 'i_exptime', 'filters']]))
      print('\n')

      print("Type 'y' for all observations, 'n' for none. Type the number within parentheses if you wish to use that specific set of observations. You can enter several numbers separated by space:\n")

      for ii, index_obs in enumerate(obs_table['obsid']):
         print('%s (%i)'%(index_obs, ii+1))

      print('\n')
      HST_obs_to_use = input('Please type your answer and press enter: ')

      try:
         HST_obs_to_use = str2bool(HST_obs_to_use)
      except:
         try:
            HST_obs_to_use = list(set([obsid for obsid in obs_table.obsid[[int(obsid)-1 for obsid in HST_obs_to_use.split()]] if np.isfinite(obsid)]))
         except:
            print('No valid input. Not downloading observations.')
            HST_obs_to_use = False

      if HST_obs_to_use is not False:
         if HST_obs_to_use is not True:
            hst_images = download_HST_images(Table.from_pandas(data_products_by_obs.loc[data_products_by_obs['parent_obsid'].isin(HST_obs_to_use), :]), path = args.HST_path)
         else:
            HST_obs_to_use = list(obs_table['obsid'].values)
            hst_images = download_HST_images(Table.from_pandas(data_products_by_obs), path = args.HST_path)

      """
      Call hst1pass
      """
      hst1pass(data_products_by_obs, HST_obs_to_use, args.HST_path, force_fmin = args.fmin, remove_previous_XYmqxy = True, force_hst1pass = args.force_hst1pass)

      """
      Call xym2pm_Gaia
      """
      Gaia_table_hst = xym2pm_Gaia(Gaia_table.copy(), data_products_by_obs, HST_obs_to_use, args.HST_path, args.date_second_epoch, only_use_members = args.use_members, force_pixel_scale = args.pixel_scale, force_max_separation = args.max_separation, force_use_sat = args.force_use_sat, n_components = args.pm_n_components, clipping_prob = args.clipping_prob_pm, min_stars_alignment = args.min_stars_alignment, plots = args.plots, verbose = args.verbose, remove_previous_files = True, plot_name = args.base_path+'PM_selection')

      """
      Obtain absolute PMs
      """
      Gaia_table_hst = absolute_pm(Gaia_table_hst.copy(), use = 'members')

      """
      Save Gaia and HST tables
      """
      obs_table.to_csv(args.HST_obs_table_filename, index = False)
      data_products_by_obs.to_csv(args.HST_data_table_products_filename, index = False)
      Gaia_table_hst.to_csv(args.Gaia_table_filename, index = False)

      """
      Print a summary with the location of files and plot the results
      """
      plot_results(Gaia_table_hst, plot_name_1 = args.base_path+args.base_file_name+'_vpd.pdf', plot_name_2 = args.base_path+args.base_file_name+'_diff.pdf', plot_name_3 = args.base_path+args.base_file_name+'_cmd.pdf')

      print('\n')
      print(' RESULTS '.center(80, '*'))
      print('- Table: %s'%args.Gaia_table_filename)
      print('-'*80)
      print('- HST observations: %s'%args.HST_obs_table_filename)
      print('- HST products: %s'%args.HST_data_table_products_filename)
      print('*'*80)
      print('\nExecution ended.\n')
   else:
      input('No suitable HST observations were found. Please try with different parameters.\nPress enter to exit.\n')
if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)

"""
Andres del Pino Molina
"""
