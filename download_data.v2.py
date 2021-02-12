#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys
import os
import subprocess
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from sklearn.utils.validation import check_is_fitted
from sklearn import mixture

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.time import Time
from astropy.coordinates import match_coordinates_sky
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import (MinMaxInterval, SqrtStretch, ImageNormalize, ManualInterval)
import astropy.units as u
from astropy.coordinates import SkyCoord

from scipy import stats
from math import log10, floor

from zero_point import zpt


def correct_gband(bp_rp, astrometric_params_solved, phot_g_mean_mag, phot_g_mean_flux):
   """
   Correct the G-band fluxes and magnitudes for the input list of Gaia EDR3 data.
   
   Parameters
   ----------
   
   bp_rp: float, array_like
       The (BP-RP) colour listed in the Gaia EDR3 archive.
   astrometric_params_solved: int, array_like
       The astrometric solution type listed in the Gaia EDR3 archive.
   phot_g_mean_mag: float, array_like
       The G-band magnitude as listed in the Gaia EDR3 archive.
   phot_g_mean_flux: float, array_like
       The G-band flux as listed in the Gaia EDR3 archive.
       
   Returns
   -------
   
   The corrected G-band magnitudes and fluxes. The corrections are only applied to
   sources with a 6-parameter astrometric solution fainter than G=13, for which a
   (BP-RP) colour is available.
   
   Example
   
   gmag_corr, gflux_corr = correct_gband(bp_rp, astrometric_params_solved, 
                                         phot_g_mean_mag, phot_g_mean_flux)
   """
   if np.isscalar(bp_rp) or np.isscalar(astrometric_params_solved) or \
                   np.isscalar(phot_g_mean_mag) or np.isscalar(phot_g_mean_flux):
       bp_rp = np.float64(bp_rp)
       astrometric_params_solved = np.int64(astrometric_params_solved)
       phot_g_mean_mag = np.float64(phot_g_mean_mag)
       phot_g_mean_flux = np.float64(phot_g_mean_flux)
   
   if not (bp_rp.shape == astrometric_params_solved.shape \
                       == phot_g_mean_mag.shape == phot_g_mean_flux.shape):
       raise ValueError('Function parameters must be of the same shape!')
   
   do_not_correct = np.isnan(bp_rp) | (phot_g_mean_mag<=13) | \
                                      (astrometric_params_solved != 95)
   bright_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag>13) & \
                                  (phot_g_mean_mag<=16)
   faint_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag>16)
   bp_rp_c = np.clip(bp_rp, 0.25, 3.0)
   
   correction_factor = np.ones_like(phot_g_mean_mag)
   correction_factor[faint_correct] = 1.00525 - 0.02323*bp_rp_c[faint_correct] + 0.01740*np.power(bp_rp_c[faint_correct],2) - 0.00253*np.power(bp_rp_c[faint_correct],3)
   correction_factor[bright_correct] = 1.00876 - 0.02540*bp_rp_c[bright_correct] + 0.01747*np.power(bp_rp_c[bright_correct],2) - 0.00277*np.power(bp_rp_c[bright_correct],3)
   
   gmag_corrected = phot_g_mean_mag - 2.5*np.log10(correction_factor)
   gflux_corrected = phot_g_mean_flux * correction_factor
   
   return gmag_corrected, gflux_corrected
 

def correct_flux_excess_factor(bp_rp, phot_bp_rp_excess_factor):
   """
   Calculate the corrected flux excess factor for the input Gaia EDR3 data.
   
   Parameters
   ----------
   
   bp_rp: float, array_like
       The (BP-RP) colour listed in the Gaia EDR3 archive.
   phot_bp_rp_flux_excess_factor: float, array_like
       The flux excess factor listed in the Gaia EDR3 archive.
       
   Returns
   -------
   
   The corrected value for the flux excess factor, which is zero for "normal" stars.
   
   Example
   -------
   
   phot_bp_rp_excess_factor_corr = correct_flux_excess_factor(bp_rp, 
                                               phot_bp_rp_flux_excess_factor)
   """

   if np.isscalar(bp_rp) or np.isscalar(phot_bp_rp_excess_factor):
       bp_rp = np.float64(bp_rp)
       phot_bp_rp_excess_factor = np.float64(phot_bp_rp_excess_factor)
   
   if bp_rp.shape != phot_bp_rp_excess_factor.shape:
       raise ValueError('Function parameters must be of the same shape!')
   
   do_not_correct = np.isnan(bp_rp)
   bluerange = np.logical_not(do_not_correct) & (bp_rp < 0.5)
   greenrange = np.logical_not(do_not_correct) & (bp_rp >= 0.5) & (bp_rp < 4.0)
   redrange = np.logical_not(do_not_correct) & (bp_rp > 4.0)
   
   correction = np.zeros_like(bp_rp)
   correction[bluerange] = 1.154360 + 0.033772*bp_rp[bluerange] + 0.032277*np.power(bp_rp[bluerange],2)
   correction[greenrange] = 1.162004 + 0.011464*bp_rp[greenrange] + 0.049255*np.power(bp_rp[greenrange],2) - 0.005879*np.power(bp_rp[greenrange],3)
   correction[redrange] = 1.057572 + 0.140537*bp_rp[redrange]
   
   return phot_bp_rp_excess_factor - correction


def round_significant(x, ex, sig=1):
   """
   This routine returns a quantity rounded to its error significan figures.
   """

   significant = sig-int(floor(log10(abs(ex))))-1

   return round(x, significant), round(ex, significant)


def manual_select_from_cmd(color, mag):

   class SelectFromCollection(object):
      """
      Select indices from a matplotlib collection using `LassoSelector`.
      """
      def __init__(self, ax, collection, alpha_other=0.1):
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

   subplot_kw = dict(autoscale_on = False)
   fig, ax = plt.subplots(subplot_kw = subplot_kw)

   pts = ax.scatter(color, mag, s=1)
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
   ax.set_title("Use your cursor to select likely member stars and press enter to accept.")

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

      has_color = np.isfinite(bp_rp) & (np.isfinite(phot_g_mean_mag))
      u0gc = pd.read_csv('./DR2_RUWE_V1/table_u0_g_col.txt', header =0)

      #histogram
      dx = 0.01
      dy = 0.1
      bins = [np.arange(np.amin(u0gc['g_mag'])-0.5*dx, np.amax(u0gc['g_mag'])+dx, dx), np.arange(np.amin(u0gc[' bp_rp'])-0.5*dy, np.amax(u0gc[' bp_rp'])+dy, dy)]
      
      posx = np.digitize(phot_g_mean_mag[has_color], bins[0])
      posy = np.digitize(bp_rp[has_color], bins[1])

      posx[posx < 1] = 1
      posx[posx > len(bins[0])-1] = len(bins[0])-1
      posy[posy < 1] = 1
      posy[posy > len(bins[1])-1] = len(bins[1])-1

      u0_gc = np.reshape(np.array(u0gc[' u0']), (len(bins[0])-1, len(bins[1])-1))[posx, posy]
      uwe[has_color] /= np.array(u0_gc)

      if not all(has_color):
         u0g = pd.read_csv('./DR2_RUWE_V1/table_u0_g.txt', header =0)

         posx = np.digitize(phot_g_mean_mag[~has_color], bins[0])

         posx[posx < 1] = 1
         posx[posx > len(bins[0])-1] = len(bins[0])-1

         u0_c = u0g[' u0'][posx]
         uwe[~has_color] /= np.array(u0_c)

   return uwe


def get_real_error(table):
   """
   This routine calculates the excess of error that should be added to the listed Gaia errors. The lines are based on Fabricius 2020. Figure 21.
   """

   p5 = table.astrometric_params_solved == 31                                        # 5p parameters solved Brown et al. (2020)
   
   table.loc[:, ['parallax_error_old', 'pmra_error_old', 'pmdec_error_old']] = table.loc[:, ['parallax_error', 'pmra_error', 'pmdec_error']].values
   
   table.loc[p5, ['parallax_error', 'pmra_error', 'pmdec_error']] = np.sqrt(table.loc[p5, ['parallax_error', 'pmra_error', 'pmdec_error']].values**2 * 1.05**2)
   table.loc[~p5, ['parallax_error', 'pmra_error', 'pmdec_error']] = np.sqrt(table.loc[~p5, ['parallax_error', 'pmra_error', 'pmdec_error']].values**2 * 1.22**2)

   return table



def clean_astrometry(ruwe, ipd_gof_harmonic_amplitude, visibility_periods_used, astrometric_excess_noise_sig, astrometric_params_solved, use_5p = False):
   """
   Select stars with good astrometry in Gaia.
   """

   labels_ruwe = ruwe <= 1.4
   labels_harmonic_amplitude = ipd_gof_harmonic_amplitude <= 0.2                                    # Reject blended transits Fabricius et al. (2020)
   labels_visibility = visibility_periods_used >= 9                                                 # Lindengren et al. (2020)
   labels_excess_noise = astrometric_excess_noise_sig <= 2.0                                        # Lindengren et al. (2020)

   labels_astrometric = labels_ruwe & labels_harmonic_amplitude & labels_visibility & labels_excess_noise

   if use_5p:
      labels_params_solved = astrometric_params_solved == 31                                        # 5p parameters solved Brown et al. (2020)
      labels_astrometric = labels_astrometric & labels_params_solved

   return labels_astrometric


def clean_photometry(gmag, corrected_flux_excess_factor, sigma_flux_excess_factor = 3):
   """
   This routine select stars based on their flux_excess_factor. Riello et al.2020
   """
   from matplotlib.path import Path

   def sigma_corrected_C(gmag, sigma_flux_excess_factor):
      return sigma_flux_excess_factor*(0.0059898 + 8.817481e-12 * gmag ** 7.618399)

   mag_nodes = np.linspace(np.min(gmag)-0.1, np.max(gmag)+0.1, 100)
   
   up = [(gmag, sigma) for gmag, sigma in zip(mag_nodes, sigma_corrected_C(mag_nodes, sigma_flux_excess_factor))]
   down = [(gmag, sigma) for gmag, sigma in zip(mag_nodes[::-1], sigma_corrected_C(mag_nodes, -sigma_flux_excess_factor)[::-1])]

   path_C = Path(up+down, closed=True)
   
   labels_photometric = path_C.contains_points(np.array([gmag, corrected_flux_excess_factor]).T)

   return labels_photometric


def pre_clean_data(phot_g_mean_mag, corrected_flux_excess_factor, ruwe, ipd_gof_harmonic_amplitude, visibility_periods_used, astrometric_excess_noise_sig, astrometric_params_solved, sigma_flux_excess_factor = 3, use_5p = False):
   """
   This routine cleans the Gaia data from astrometrically and photometric bad measured stars.
   """
   
   labels_photometric = clean_photometry(phot_g_mean_mag, corrected_flux_excess_factor, sigma_flux_excess_factor = sigma_flux_excess_factor)
   
   labels_astrometric = clean_astrometry(ruwe, ipd_gof_harmonic_amplitude, visibility_periods_used, astrometric_excess_noise_sig, astrometric_params_solved, use_5p = use_5p)
   
   return labels_photometric & labels_astrometric


def remove_jobs():
   """
   This routine removes jobs from the Gaia archive server.
   """

   list_jobs = []
   for job in Gaia.list_async_jobs():
      list_jobs.append(job.get_jobid())
   
   Gaia.remove_jobs(list_jobs)


def gaia_log_in(gaia_user = None, gaia_paswd = None):
   """
   This routine log in to the Gaia archive.
   """

   from astroquery.gaia import Gaia
   import getpass

   while True:
      try:
         Gaia.login(user=gaia_user, password=gaia_paswd)
         print("Welcome to the Gaia server!")
         break
      except:
         print("Please introduce username and password")
         gaia_user = input("Gaia username: ")
         gaia_paswd = getpass.getpass(prompt='Gaia password: ') 

   return Gaia


def gaia_query(Gaia, query, min_gmag, max_gmag, norm_uwe, test_mode, save_individual_queries, load_existing, name, n, n_total):
   """
   This routine launch the query to the Gaia archive.
   """

   query = query + " AND (phot_g_mean_mag > %.4f) AND (phot_g_mean_mag <= %.4f)"%(min_gmag, max_gmag)

   if not test_mode:
      
      individual_query_filename = './%s/Gaia/individual_queries/%s_G_%.4f_%.4f.csv'%(name, name, min_gmag, max_gmag)

      if os.path.isfile(individual_query_filename) and load_existing:
         result = pd.read_csv(individual_query_filename)

      else:
         job = Gaia.launch_job_async(query)
         result = job.get_results()
         removejob = Gaia.remove_jobs([job.jobid])
         result = result.to_pandas()

         if save_individual_queries:
            result.to_csv(individual_query_filename, index = False)

   else:
      result = pd.DataFrame()
   
   print('\n')
   print('----------------------------')
   print('Table %i of %i: %i stars'%(n, n_total, len(result)))
   print('----------------------------')

   return result, query


def get_mag_bins(min_mag, max_mag, area, mag = None):
   """
   This routine generates logarithmic spaced bins for G magnitude.
   """

   num_nodes = np.max((1, np.round( ( (max_mag - min_mag) * max_mag ** 2 * area)*5e-5)))

   bins_mag = (1.0 + max_mag - np.logspace(np.log10(1.), np.log10(1. + max_mag - min_mag), num = int(num_nodes), endpoint = True))

   return bins_mag


def gaia_multi_query_run(args):
   """
   This routine pipes gaia_query into multiple threads.
   """

   return gaia_query(*args)


def columns_n_conditions(source_table, search_type, astrometric_cols, photometric_cols, quality_cols, ra, dec, min_radius = 0.5, max_radius = 1.0, width = 1.0, height = 1.0, max_gmag_error = 0.5, max_rpmag_error = 0.5, max_bpmag_error = 0.5, min_parallax = -2, max_parallax = 1, max_parallax_error = 1.0, min_pmra = -6, max_pmra = 6, max_pmra_error = 1.0, min_pmdec = -6, max_pmdec = 6, max_pmdec_error = 1.0):

   """
   This routine generates the columns and conditions for the query.
   """

   if 'dr3' in source_table:
      if 'ruwe' not in quality_cols:
         quality_cols = 'ruwe' +  (', ' + quality_cols if len(quality_cols) > 1 else '')
   elif 'dr2' in source_table:
      if 'astrometric_n_good_obs_al' not in quality_cols:
         quality_cols = 'astrometric_n_good_obs_al' +  (', ' + quality_cols if len(quality_cols) > 1 else '')
      if 'astrometric_chi2_al' not in quality_cols:
         quality_cols = 'astrometric_chi2_al' +  (', ' + quality_cols if len(quality_cols) > 1 else '')
      if 'phot_bp_rp_excess_factor' not in quality_cols:
         quality_cols = 'phot_bp_rp_excess_factor' +  (', ' + quality_cols if len(quality_cols) > 1 else '')

   if search_type == 'box':
      search_area = "CONTAINS(POINT('ICRS',"+source_table+".ra,"+source_table+".dec),BOX('ICRS',%.8f,%.8f,%.8f,%.8f))=1"%(ra, dec, width, height)
   elif search_type == 'anulus':
      search_area = "CONTAINS(POINT('ICRS',"+source_table+".ra,"+source_table+".dec),CIRCLE('ICRS',%.8f,%.8f,%.8f))=1"%(ra, dec, max_radius) +" AND CONTAINS(POINT('ICRS',"+source_table+".ra,"+source_table+".dec), CIRCLE('ICRS',%.8f,%.8f,%.8f))=0"%(ra, dec, min_radius)
   else:
      search_area = "CONTAINS(POINT('ICRS',"+source_table+".ra,"+source_table+".dec),CIRCLE('ICRS',%.8f,%.8f,%.8f))=1"%(ra, dec, max_radius)

   conditions = search_area + ' AND (pmra > %.4f) AND (pmra < %.4f) AND (pmra_error < %.4f) AND (pmdec > %.4f) AND (pmdec < %.4f) AND (pmdec_error < %.4f) AND (parallax > %.4f) AND (parallax < %.4f) AND (parallax_error < %.4f) AND ((1.09*phot_g_mean_flux_error/phot_g_mean_flux) < %.4f) AND ((1.09*phot_bp_mean_flux_error/phot_bp_mean_flux) < %.4f) AND ((1.09*phot_rp_mean_flux_error/phot_rp_mean_flux) < %.4f)'%(min_pmra, max_pmra, max_pmra_error, min_pmdec, max_pmdec, max_pmdec_error, min_parallax, max_parallax, max_parallax_error, max_gmag_error, max_bpmag_error, max_rpmag_error)

   columns = (", " + astrometric_cols if len(astrometric_cols) > 1 else '') + (", " + photometric_cols if len(photometric_cols) > 1 else '') +  (", " + quality_cols if len(quality_cols) > 1 else '')

   query = "SELECT source_id " + columns + " FROM " + source_table + " WHERE " + conditions

   return query, quality_cols
   

def incremental_query(query, area, min_gmag = 10.0, max_gmag = 19.5, norm_uwe = True, use_parallel = True, test_mode = False, save_individual_queries = False, load_existing = False, name = 'output', gaia_user = None, gaia_paswd = None):

   """
   This routine search the Gaia archive and downloads the stars using parallel workers.
   """

   from multiprocessing import Pool, cpu_count

   if not test_mode:
      Gaia = gaia_log_in(gaia_user = gaia_user, gaia_paswd = gaia_paswd)
   else:
      Gaia = None

   mag_nodes = get_mag_bins(min_gmag, max_gmag, area)
   n_total = len(mag_nodes)
   
   if (n_total > 1) and use_parallel:

      print("Executing %s jobs."%(n_total-1))

      nproc = int(np.min((n_total, 20, cpu_count()*2)))

      pool = Pool(nproc-1)

      args = []
      for n, node in enumerate(range(n_total-1)):
         args.append((Gaia, query, mag_nodes[n+1], mag_nodes[n], norm_uwe, test_mode, save_individual_queries, load_existing, name, n, n_total))

      tables_gaia_queries = pool.map(gaia_multi_query_run, args)

      tables_gaia = [results[0] for results in tables_gaia_queries]
      queries = [results[1] for results in tables_gaia_queries]

      result_gaia = pd.concat(tables_gaia)

      pool.close()

   else:
      result_gaia, queries = gaia_query(Gaia, query, min_gmag, max_gmag, norm_uwe, test_mode, save_individual_queries, load_existing, name, 1, 1)

   if not test_mode:
      Gaia.logout()

   return result_gaia, queries


def get_area(search_type, max_radius, min_radius, width, height, dec):
   """
   This routine calculates the covered area.
   """

   if search_type == 'box':
      area = height * width * np.abs(np.cos(np.deg2rad(dec)))
   elif search_type == 'anulus':
      area = np.pi*max_radius**2 - np.pi*min_radius**2
   else:
      area = np.pi*max_radius**2

   return area


def get_coo_split(args, table):
   
   center = c.SkyCoord(args.ra*u.deg, args.dec*u.deg)
   coords = c.SkyCoord(table.ra*u.deg, table.dec*u.deg)

   table['r'] = center.separation(coords).to("arcsec").value
   #table["phi"] = center.position_angle(coords).value
   table['x'], table['y'] = wcs2xy(table.ra, table.dec, args.ra, args.dec)

   data = table.loc[table.r <= (args.max_search_radius*u.deg).to("arcsec").value, :]
   field = table.loc[table.r > (args.max_search_radius*u.deg).to("arcsec").value, :]

   return data, field


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


def round_significant(x, ex, sig=1):
   """
   This routine returns a quantity rounded to its error significan figures.
   """
   
   try:
      significant = sig-int(floor(log10(abs(ex))))-1
   except:
      significant = 4

   return round(x, significant), round(ex, significant)


def select_conditions(args, table):
   """
   Select table based on simple conditions
   """

   conditions_astrometry = (table.pmra >= args.min_pmra) & (table.pmra <= args.max_pmra)\
                         & (table.pmdec >= args.min_pmdec) & (table.pmdec <= args.max_pmdec)\
                         & (table.parallax >= args.min_parallax) & (table.parallax <= args.max_parallax)
   
   conditions_photometry = (table.bp_rp >= args.min_bp_rp) & (table.bp_rp <= args.max_bp_rp)\
                         & (table.gmag >= args.min_gmag) & (table.gmag <= args.max_gmag)\
                         & (table.bpmag <= args.max_gmag) & (table.rpmag <= args.max_rpmag)\

   conditions_astrometric_error = (table.pmra_error <= args.max_pmra_error) & (table.pmdec_error <= args.max_pmdec_error) & (table.parallax_error <= args.max_parallax_error)

   conditions_photometric_error = (table.bpmag_error <= args.max_bpmag_error) & (table.rpmag_error <= args.max_rpmag_error) & (table.gmag_error <= args.max_gmag_error)

   table = table[conditions_astrometry & conditions_photometry & conditions_astrometric_error & conditions_photometric_error]
   
   return table


def plot_fields(Gaia_table, obs_table, HST_path, min_stars_alignment = 5, name = 'test.png'):
   """
   This routine plots the fields and select Gaia stars within them.
   """

   from matplotlib.patches import Polygon
   from matplotlib.collections import PatchCollection
   from shapely.geometry.polygon import Polygon as shap_polygon
   from shapely.geometry import Point

   def deg_to_hms(lat, even = True):
      from astropy.coordinates import Angle
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
      angle = Angle(lat, unit = u.deg)
      string = angle.to_string(unit=u.degree)
      return string

   def coolwarm(filter, alpha):
      color = [rgb for rgb in plt.cm.coolwarm(255 * (float(filter) - 555) / (850-555))]
      color[-1] = alpha
      return color

   Gaia_table['parent_obsid'] = ""
   
   fig, ax = plt.subplots(1,1, figsize = (5.5, 5.5))
   patches = []
   ecs = []
   fcs = []
   previous_obsid = []
   gaia_stars_per_obs = []
   
   filter_range = [float(s.replace('F', '').replace('W', '').replace('LP', '')) for s in obs_table.filters]

   for index_obs, (footprint_str, obsid, filter, obs_id) in obs_table.loc[:, ['s_region', 'obsid', 'filters', 'obs_id']].sort_values(by=['obsid']).iterrows():
      cli_progress_test(index_obs+1, len(obs_table))

      idx_Gaia_in_field = []
      gaia_stars_per_poly = [] 
      list_coo = footprint_str.split('POLYGON')[1::]

      for poly in list_coo:
         try:
            poly = list(map(float, poly.split()))
         except:
            poly = list(map(float, poly.split('J2000')[1::][0].split()))

         tuples_list = [(ra % 360, dec) for ra, dec in zip(poly[0::2], poly[1::2])]

         # Make sure the field is complete. With at least 4 vertices.
         if len(tuples_list) > 4:
            polygon = Polygon(tuples_list, True)
            ecs.append(coolwarm(filter.replace(r'F', '').replace(r'W', '').replace('LP', ''), 1))
            
            # Check if the set seems downloaded
            if os.path.isfile(HST_path+'mastDownload/HST/'+obs_id+'/'+obs_id+'_drz.fits'):
               fcs.append([0,1,0,0.2])
            else:
               fcs.append([1,1,1,0.2])

            footprint =  shap_polygon(tuples_list)

            star_counts = 0
            for idx, ra, dec in zip(Gaia_table.index, Gaia_table.ra, Gaia_table.dec):
               if Point(ra, dec).within(footprint):
                  idx_Gaia_in_field.append(idx)
                  star_counts += 1

            gaia_stars_per_poly.append(star_counts)
            
            if star_counts >= min_stars_alignment:
               patches.append(polygon)

               annotation_coo = [round(max(tuples_list)[0], 2)-0.028, round(max(tuples_list)[1], 2)-0.028]

               if annotation_coo in previous_obsid:
                  annotation_coo[1] += 0.01

               ax.annotate(index_obs+1, xy=(annotation_coo[0], annotation_coo[1]), xycoords='data', color = coolwarm(filter.replace(r'F', '').replace(r'W', '').replace('LP', ''), 1))

               previous_obsid.append(annotation_coo)

      gaia_stars_per_obs.append(sum(gaia_stars_per_poly))

      Gaia_table.loc[idx_Gaia_in_field, 'parent_obsid'] = Gaia_table.loc[idx_Gaia_in_field, 'parent_obsid'].astype(str) + '%s '%obsid

   print('\n')

   previous_obsid = np.array(previous_obsid)
   obs_table['gaia_stars_per_obs'] = gaia_stars_per_obs
   
   try:
      ra_lims = [max(Gaia_table.ra.max(), previous_obsid[:,0].max()), min(Gaia_table.ra.min(), previous_obsid[:,0].min())]
      dec_lims = [min(Gaia_table.dec.min(), previous_obsid[:,1].min()), max(Gaia_table.dec.max(), previous_obsid[:,1].max())]
   except:
      ra_lims = [Gaia_table.ra.max(), Gaia_table.ra.min()]
      dec_lims = [Gaia_table.dec.min(), Gaia_table.dec.max()]
 
   pe = PatchCollection(patches, alpha = 0.1, ec = 'None', fc = fcs, antialiased = True, lw = 1, zorder = 4)
   pf = PatchCollection(patches, alpha = 1, ec = ecs, fc = 'None', antialiased = True, lw = 1, zorder = 5)
   ax.add_collection(pe)
   ax.add_collection(pf)
   ax.plot(Gaia_table.ra, Gaia_table.dec, '.', color = '0.4', ms = 0.75, zorder = 1)
   ax.set_xlim(ra_lims[0], ra_lims[1])
   ax.set_ylim(dec_lims[0], dec_lims[1])
   ax.grid()

   ax.set_xlabel(r'$\alpha$ [deg]')
   ax.set_ylabel(r'$\delta$ [deg]')

   plt.savefig(name, bbox_inches='tight')

   with plt.rc_context(rc={'interactive': False}):
      plt.gcf().show()

   obs_table = obs_table.loc[obs_table.gaia_stars_per_obs >= min_stars_alignment].sort_values(by =['proposal_id', 'obsid'], ascending = False).reset_index()
   obs_table[''] = ['(%i)'%(ii+1) for ii in np.arange(len(obs_table))]
   
   return Gaia_table, obs_table


def search_mast(ra, dec, width, height, filters = 'any', t_exptime_min = 50, t_exptime_max = 2500, date_second_epoch = 57531.0, time_baseline = 3650):
   """
   This routine search for HST observations in MAST at a given position.
   """

   from astroquery.mast import Catalogs
   from astroquery.mast import Observations

   ra1 = ra - width / 2 + 0.056*np.cos(np.deg2rad(dec))
   ra2 = ra + width / 2 - 0.056*np.cos(np.deg2rad(dec))
   dec1 = dec - height / 2 + 0.056
   dec2 = dec + height / 2 - 0.056

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

   obs_table = obs_table.merge(data_products_by_obs.loc[data_products_by_obs.productSubGroupDescription == 'FLC', :] .groupby(['parent_obsid'])['parent_obsid'].count().rename_axis('obsid').rename('n_exp'), on = ['obsid'])

   obs_table['i_exptime'] = obs_table['t_exptime'] / obs_table['n_exp']

   #For convenience we add an extra column with the time baseline
   obs_time = Time(obs_table['t_max'], format='mjd')
   obs_time.format = 'iso'
   obs_time.out_subfmt = 'date'
   obs_table['obs_time'] = obs_time

   obs_table['t_baseline'] = round((date_second_epoch - obs_table['t_max']) / 365.2422, 2)
   obs_table['filters'] = obs_table['filters'].str.strip('; CLEAR2L CLEAR1L')

   data_products_by_obs = data_products_by_obs.merge(obs_table.loc[:, ['obsid', 'i_exptime', 'filters', 't_baseline', 's_ra', 's_dec']].rename(columns={'obsid':'parent_obsid'}), on = ['parent_obsid'])

   #We select by individual exp time:
   obs_table = obs_table.loc[(obs_table.i_exptime > t_exptime_min) & (obs_table.i_exptime < t_exptime_max) & (obs_table.t_baseline > time_baseline / 365.2422 )]
   data_products_by_obs = data_products_by_obs.loc[(data_products_by_obs.i_exptime > t_exptime_min) & (data_products_by_obs.i_exptime < t_exptime_max) & (data_products_by_obs.t_baseline > time_baseline / 365.2422 )]

   return obs_table.astype({'obsid': 'int64'}).reset_index(drop = False), data_products_by_obs.astype({'parent_obsid': 'int64'}).reset_index(drop = False)


def search_data_products_by_obs(obs_table):
   """
   This routine search for images in MAST related to the given observations table.
   """
   
   from astroquery.mast import Observations

   data_products_by_obs = Observations.get_product_list(obs_table)

   return data_products_by_obs[((data_products_by_obs['productSubGroupDescription'] == 'FLC') | (data_products_by_obs['productSubGroupDescription'] == 'DRZ')) & (data_products_by_obs['obs_collection'] == 'HST')]


def download_HST_images(data_products_by_obs, path = './'):
   """
   This routine downloads the selected HST images from MAST.
   """
   
   from astroquery.mast import Observations

   images = Observations.download_products(data_products_by_obs, download_dir=path)
   
   print('')

   return images


def read_isochrones(Ages, Zs, max_gmag = -3.5):
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
         Max_mag = isochrone[:,3] <= max_gmag
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

   member_cmd = pd.Series(index = table.index, dtype=bool)

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
      label_GMM = log_prob >= np.median(log_prob[clustering_data])-clipping_prob*np.std(log_prob[clustering_data])

      results.loc[has_vars, 'member_clustering_prob'] = log_prob
      results.loc[has_vars, 'member_clustering'] = label_GMM

   return results


def pm_cleaning_GMM_recursive(table, vars, alt_table = None, data_0 = None, n_components = 1, covariance_type = 'full', clipping_prob = 3, plots = True, verbose = False, plot_name = ''):
   """
   This routine iteratively find members using a Gaussian mixture model.
   """
   
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


def applied_pert(XYmqxyrd_filename):
   """
   This routine will read the XYmqxyrd file and find whether the PSF perturbation worked . If file not found then False.
   """

   try:
      f = open(XYmqxyrd_filename, 'r')
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
         print('CAUTION: No information about PSF perturbation found in %s!'%XYmqxyrd_filename)
         return True
   except:
      return False


def get_fmin(i_exptime):
   """
   This routine returns the best value for FMIN to be used in hst1pass execution based on the integration time
   """

   # These are the values for fmin at the given lower_exptime and upper_exptime
   lower_fmin, upper_fmin = 1000, 5000
   lower_exptime, upper_exptime = 50, 500

   return min(max((int(lower_fmin + (upper_fmin - lower_fmin) * (i_exptime - lower_exptime) / (upper_exptime - lower_exptime)), 1000)), 10000)


def hst1pass_multiproc_run(args):
   """
   This routine pipes hst1pass into multiple threads.
   """

   return hst1pass(*args)


def hst1pass(HST_path, obs_id, HST_image, force_fmin, force_hst1pass, remove_previous_XYmqxyrd, verbose):
   """
   This routine will execute hst1pass Fortran routine using the correct arguments.
   """

   #Define HST servicing missions times
   t_sm3 = Time('2002-03-12').mjd
   t_sm4 = Time('2009-05-24').mjd

   output_dir = HST_path+'mastDownload/HST/'+obs_id+'/'
   HST_image_filename = output_dir+HST_image
   XYmqxyrd_filename = HST_image_filename.split('.fits')[0]+'.XYmqxyrd'

   if not os.path.isfile(XYmqxyrd_filename) or force_hst1pass:
      if verbose:
         print('Finding sources in', HST_image)

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
         print('  EXPTIME = %.1f. Using FMIN = %i'%(hdul[0].header['EXPTIME'], fmin))

      psf_filename = './Auxiliary/STDPSFs/%s%s/PSFSTD_%s%s_%s%s.fits'%(instrument, detector, instrument, detector, filter, sm)
      gdc_filename = './Auxiliary/STDGDCs/%s%s/STDGDC_%s%s_%s.fits'%(instrument, detector, instrument, detector, filter)

      if remove_previous_XYmqxyrd:
         remove_file(XYmqxyrd_filename)

      pert_grid = 5
      while not applied_pert(XYmqxyrd_filename) and (pert_grid > 0):
         # In principle, we should be abble to fine-tune FMIN knowing the exptime and the filter. Maybe for the next version.
         bashCommand = "./Auxiliary/hst1pass.e HMIN=5 FMIN=%s PMAX=999999 GDC=%s PSF=%s PERT%i=AUTO OUT=XYmqxyrd OUTDIR=%s %s"%(fmin, gdc_filename, psf_filename, pert_grid, output_dir, HST_image_filename)
         process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

         output, error = process.communicate()
         pert_grid -= 1

      if verbose:
         print('    Used PERT PSF = %sx%s'%(pert_grid, pert_grid))

      remove_file(HST_image.replace('_flc','_psf'))


def launch_hst1pass(flc_images, HST_obs_to_use, HST_path, force_fmin = None, force_hst1pass = True, remove_previous_XYmqxyrd = True, verbose = True, use_parallel = True):
   """
   This routine will launch hst1pass routine in parallel or serial 
   """
   from multiprocessing import Pool, cpu_count

   args = []
   for HST_image_obsid in [HST_obs_to_use] if not isinstance(HST_obs_to_use, list) else HST_obs_to_use:
      for index_image, (obs_id, HST_image) in flc_images.loc[flc_images['parent_obsid'] == HST_image_obsid, ['obs_id', 'productFilename']].iterrows():
         args.append((HST_path, obs_id, HST_image, force_fmin, force_hst1pass, remove_previous_XYmqxyrd, verbose))
   
   if (len(args) > 1) and use_parallel:
      pool = Pool(min(cpu_count(), len(args)))
      pool.map(hst1pass_multiproc_run, args)
      pool.close()
   
   else:
      for arg in args:
         hst1pass_multiproc_run(arg)

   remove_file('LOG.psfperts.fits')
   remove_file('fort.99')


def check_mat(mat_filename, iteration, min_stars_alignment = 100, alpha = 0.01, center_tolerance = 1e-3, plots = True, fix_mat = True, clipping_prob = 2, verbose= True):
   """
   This routine will read the transformation file MAT and provide a quality flag based on how Gaussian the transformation is.
   """

   mat = np.loadtxt(mat_filename)

   if fix_mat:
      clf = mixture.GaussianMixture(n_components = 1, covariance_type = 'spherical')
      clf.fit(mat[:, 6:8])
      log_prob = clf.score_samples(mat[:, 6:8])
      good_for_alignment = log_prob >= np.median(log_prob)-clipping_prob*np.std(log_prob)
      if verbose and (good_for_alignment.sum() < len(log_prob)): print('   Fixing MAT file for next iteration.')
      np.savetxt(mat_filename, mat[good_for_alignment, :], fmt='%12.4f')
   else:
      good_for_alignment = [True]*len(mat)

   with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      # Obtain the Shapiro–Wilk statistics
      stat, p = stats.shapiro(mat[:, 6:8])
   
   valid = [p > alpha,  (mat[:, 6:8].mean(axis = 0) < center_tolerance).all(), len(mat) > min_stars_alignment]

   if plots:
      plt.close()
      fig, ax1 = plt.subplots(1, 1)
      try:
         ax1.plot(mat[~good_for_alignment,6], mat[~good_for_alignment,7], '.', ms = 2, label = 'Rejected')
      except:
         pass
      ax1.plot(mat[good_for_alignment,6], mat[good_for_alignment,7], '.', ms = 2, label = 'Used')
      ax1.axvline(x=0, linewidth = 0.75, color = 'k')
      ax1.axhline(y=0, linewidth = 0.75, color = 'k')
      ax1.set_xlabel('X_Gaia - X_HST [pixels]')
      ax1.set_ylabel('Y_Gaia - Y_HST [pixels]')
      ax1.grid()
      
      add_inner_title(ax1, 'Valid=%s\np=%.4f\ncen=(%.4f,%.4f)\nnum=%i'%(all(valid), p, mat[:, 6].mean(), mat[:, 7].mean(), len(mat)), 1)
      
      plt.savefig(mat_filename.split('.MAT')[0]+'_MAT_%i.png'%iteration, bbox_inches='tight')
      plt.close()

   return valid


def xym2pm_Gaia(iteration, Gaia_HST_table_field, Gaia_HST_table_filename, HST_image_filename, lnk_filename, mat_filename, amp_filename, date_reference_second_epoch, only_use_members, force_pixel_scale, force_max_separation, force_use_sat, fix_mat, force_wcs_search_radius, min_stars_alignment, verbose, force_xym2pm, plots, force_use_amp, min_stars_amp):
   """
   This routine will execute xym2pm_Gaia Fortran routine using the correct arguments.
   """

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

   if not os.path.isfile(lnk_filename) or force_xym2pm:

      f = open(Gaia_HST_table_filename, 'w+')
      f.write('# ')
      Gaia_HST_table_field.loc[:, ['ra', 'ra_error', 'dec', 'dec_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'gmag', 'use_for_alignment']].astype({'use_for_alignment': 'int32'}).to_csv(f, index = False, sep = ' ', na_rep = 0)
      f.close()

      # Here it goes the executable line. Input values can be fine-tuned here
      if force_use_sat:
         use_sat = ' USESAT+'
      else:
         use_sat = ''

      if (fix_mat) and (iteration > 0):
         use_mat = " MAT=\"%s\" USEMAT+"%(mat_filename.split('./')[1])
      else:
         use_mat = ''

      if force_wcs_search_radius is None:
         use_brute = ''
      else:
         use_brute = ' BRUTE=%.1f'%force_wcs_search_radius

      if force_max_separation is None:
         max_separation = 5.0
      else:
         max_separation = force_max_separation

      if force_use_amp:
         use_amp = ' NAMP=%i AMP+'%min_stars_amp
      else:
         use_amp = ''

      bashCommand = "./Auxiliary/xym2pm_Gaia.e %s %s RACEN=%f DECEN=%f XCEN=5000.0 YCEN=5000.0 PSCL=%s SIZE=10000 DISP=%.1f NMIN=%i TIME=2016.0%s%s%s%s"%(Gaia_HST_table_filename, HST_image_filename, ra_cent, dec_cent, pixel_scale, max_separation, min_stars_alignment, use_sat, use_mat, use_brute, use_amp)

      process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      output, error = process.communicate()

   try:
      # Next are threshold rejection values for the MAT files.
      if (iteration > 0) & only_use_members:
         alpha = 1e-5
      else:
         alpha = 1e-10

      valid_mat = check_mat(mat_filename, iteration, min_stars_alignment, alpha = alpha, center_tolerance = 1e-2, fix_mat = fix_mat, clipping_prob = 3., plots = plots, verbose = verbose)

      if all(valid_mat):

         f = open(lnk_filename, 'r')
         header = [w.replace('m_hst', filter) for w in f.readline().rstrip().strip("# ").split(' ')]
         lnk = pd.read_csv(f, names=header, sep = '\s+', comment='#', na_values = 0.0).set_index(Gaia_HST_table_field.index)
         f.close()

         if os.path.isfile(amp_filename):
            f = open(amp_filename, 'r')
            header = f.readline().rstrip().strip("# ").split(' ')
            ampfile = pd.read_csv(f, names=header, sep = '\s+', comment='#', na_values = 0.0).set_index(Gaia_HST_table_field.index)
            f.close()
            lnk['xhst_gaia'] = ampfile.xhst_amp_gaia.values
            lnk['yhst_gaia'] = ampfile.yhst_amp_gaia.values

            f = open(lnk_filename.replace(".LNK", "_amp.LNK"), 'w+')
            f.write('# ')
            lnk.to_csv(f, index = False, sep = ' ', na_rep = 0)
            f.close()

         # Positional and mag error is know to be proportional to the QFIT parameter.
         eradec_hst = lnk.q_hst
         # Assign the maximum (worst) QFIT parameter to saturated stars.
         eradec_hst[lnk.xhst_gaia.notnull()] = lnk[lnk.xhst_gaia.notnull()].q_hst.replace({np.nan:lnk.q_hst.max()})
         # 0.85 seems reasonable, although this may be tuned through an empirical function.
         eradec_hst *= pixel_scale_mas * 0.85

         lnk['relative_hst_gaia_pmra'] = -(lnk.x_gaia - lnk.xhst_gaia) * pixel_scale_mas / t_baseline
         lnk['relative_hst_gaia_pmdec'] = (lnk.y_gaia - lnk.yhst_gaia) * pixel_scale_mas / t_baseline

         # Notice the 1e3. xym2pm_Gaia.e takes the error in mas/yr but returns it in arcsec.
         lnk['relative_hst_gaia_pmra_error'] = eradec_hst / t_baseline
         lnk['relative_hst_gaia_pmdec_error'] = eradec_hst / t_baseline
         lnk['gaia_ra_uncertaintity'] = 1e3 * lnk.era_gaia / t_baseline
         lnk['gaia_dec_uncertaintity'] = 1e3 * lnk.edec_gaia / t_baseline

         lnk['%s_error'%filter] = lnk.q_hst.replace({0:lnk.q_hst.max()})

         match = lnk.loc[:, [filter, '%s_error'%filter, 'relative_hst_gaia_pmra', 'relative_hst_gaia_pmra_error', 'relative_hst_gaia_pmdec', 'relative_hst_gaia_pmdec_error', 'gaia_ra_uncertaintity', 'gaia_dec_uncertaintity']]

         print('-->%s: matched %i stars.'%(os.path.basename(HST_image_filename), len(match)))

      else:
         if verbose:
            print('-->%s: bad quality match:'%os.path.basename(HST_image_filename))
            if (p < alpha):
               print('   Non Gaussian distribution found in the transformation')
            if ((mat.mean(axis = 0) > 1e-2).all()):
               print('   Not (0,0) average of the distribution')
            if (len(mat) < min_stars_alignment):
               print('   Less than %i stars used during the transformation'%min_stars_alignment)
            print('   Skipping image.')
         match = pd.DataFrame()
   except:
      print('-->%s: no match found.'%os.path.basename(HST_image_filename))
      match = pd.DataFrame()

   return match


def xym2pm_Gaia_multiproc(args):
   """
   This routine pipes xym2pm_Gaia into multiple threads.
   """

   return xym2pm_Gaia(*args)


def launch_xym2pm_Gaia(Gaia_HST_table, data_products_by_obs, HST_obs_to_use, HST_path, date_reference_second_epoch, only_use_members = False, force_pixel_scale = None, force_max_separation = None, force_use_sat = True, fix_mat = True, force_use_amp = False, min_stars_amp = 25, force_wcs_search_radius = None, n_components = 1, clipping_prob = 6, min_stars_alignment = 100, use_mean = 'wmean', plots = True, verbose = True, force_xym2pm = True, remove_previous_files = True, use_parallel = True, plot_name = ''):
   """
   This routine will launch xym2pm_Gaia Fortran routine in parallel or serial using the correct arguments.
   """
   from multiprocessing import Pool, cpu_count

   n_images = len(data_products_by_obs.loc[data_products_by_obs['parent_obsid'].isin([HST_obs_to_use] if not isinstance(HST_obs_to_use, list) else HST_obs_to_use), :])
   
   if (n_images > 1) and use_parallel:
      pool = Pool(min(cpu_count(), n_images))
      plots = False
      verbose = False

   convergence = False
   iteration = 0
   while not convergence:
      print("\n-----------")
      print("Iteration %i"%(iteration))
      print("-----------")

      args = []
      for index_image, (obs_id, HST_image) in data_products_by_obs.loc[data_products_by_obs['parent_obsid'].isin([HST_obs_to_use] if not isinstance(HST_obs_to_use, list) else HST_obs_to_use), ['obs_id', 'productFilename']].iterrows():

         HST_image_filename = HST_path+'mastDownload/HST/'+obs_id+'/'+HST_image
         Gaia_HST_table_filename = HST_path+'Gaia_%s.ascii'%HST_image.split('.fits')[0]
         lnk_filename = HST_image_filename.split('.fits')[0]+'.LNK'
         mat_filename = HST_image_filename.split('.fits')[0]+'.MAT'
         amp_filename = HST_image_filename.split('.fits')[0]+'.AMP'

         if iteration == 0:
            if remove_previous_files:
               remove_file(mat_filename)
               remove_file(lnk_filename)
               remove_file(amp_filename)

            Gaia_HST_table = find_stars_to_align(Gaia_HST_table, HST_image_filename)
            if (Gaia_HST_table.loc[Gaia_HST_table['HST_image'].str.contains(str(obs_id)), 'use_for_alignment'].sum() < min_stars_alignment):
               Gaia_HST_table.loc[Gaia_HST_table['HST_image'].str.contains(str(obs_id)), 'use_for_alignment'] = True

         Gaia_HST_table_field = Gaia_HST_table.loc[Gaia_HST_table['HST_image'].str.contains(str(obs_id)), :]

         args.append((iteration, Gaia_HST_table_field, Gaia_HST_table_filename, HST_image_filename, lnk_filename, mat_filename, amp_filename, date_reference_second_epoch, only_use_members, force_pixel_scale, force_max_separation, force_use_sat, fix_mat, force_wcs_search_radius, min_stars_alignment, verbose, force_xym2pm, plots, force_use_amp, min_stars_amp))

      if (len(args) > 1) and use_parallel:
         lnks = pool.map(xym2pm_Gaia_multiproc, args)

      else:
         lnks = []
         for arg in args:
            lnks.append(xym2pm_Gaia_multiproc(arg))

      try:
         lnks = pd.concat(lnks, sort=True)
      except:
         print('WARNING: No match could be found for any of the images. Please try with other parameters.\nExiting now.')
         sys.exit(1)

      lnks_averaged = lnks.groupby(lnks.index).apply(weighted_avg_err)

      # Gaia positional errors have to be added in quadrature 
      lnks_averaged['relative_hst_gaia_pmra_mean_error'] = np.sqrt(lnks_averaged['relative_hst_gaia_pmra_mean_error']**2 +  lnks_averaged['gaia_ra_uncertaintity_mean']**2)
      lnks_averaged['relative_hst_gaia_pmdec_mean_error'] = np.sqrt(lnks_averaged['relative_hst_gaia_pmdec_mean_error']**2 +  lnks_averaged['gaia_dec_uncertaintity_mean']**2)

      try:
         Gaia_HST_table.drop(columns = lnks_averaged.columns, inplace = True)
      except:
         pass

      Gaia_HST_table = Gaia_HST_table.join(lnks_averaged)

      # Membership selection
      if only_use_members:
         if iteration == 0:
            # Select stars in theCMD
            hst_filters = [col for col in lnks_averaged.columns if ('F' in col) & ('error' not in col) & ('std' not in col) & ('_mean' not in col)]
            hst_filters.sort()
            if len(hst_filters) == 2:
               Gaia_HST_table['clustering_data'] =  manual_select_from_cmd((Gaia_HST_table[hst_filters[0]]-Gaia_HST_table[hst_filters[1]]).rename('%s - %s'%(hst_filters[0], hst_filters[1])), Gaia_HST_table[hst_filters[1]])
            else:
               for cmd_filter in hst_filters:
                  Gaia_HST_table['%s_clustering_data_cmd'%cmd_filter] =  manual_select_from_cmd((Gaia_HST_table['gmag']-Gaia_HST_table[cmd_filter]).rename('Gmag - %s'%cmd_filter), Gaia_HST_table['gmag'].rename('Gmag'))

               cmd_clustering_filters = [col for col in Gaia_HST_table.columns if '_clustering_data_cmd' in col]
               Gaia_HST_table['clustering_data'] = (Gaia_HST_table.loc[:, cmd_clustering_filters] == True).any(axis = 1)
               Gaia_HST_table.drop(columns = cmd_clustering_filters, inplace = True)

         # Select stars in the PM space asuming spherical covariance (Reasonable for dSphs and globular clusters) 
         pm_clustering = pm_cleaning_GMM_recursive(Gaia_HST_table.copy(), ['relative_hst_gaia_pmra_%s'%use_mean, 'relative_hst_gaia_pmdec_%s'%use_mean], data_0 = [0, 0], n_components = 1, covariance_type = 'spherical', clipping_prob = clipping_prob, plots = plots, plot_name = '%s_%i'%(plot_name, iteration))
         new_use_for_alignment = pm_clustering & Gaia_HST_table.clustering_data

         if iteration > 9:
            print('\nWARNING: Max number of iterations reached: Something might have gone wrong!\nPlease check the results carefully.')
            convergence = True
         elif iteration > 0:
            convergence = np.array_equal(Gaia_HST_table['use_for_alignment'].values, new_use_for_alignment.values)

         if not convergence:
            Gaia_HST_table['use_for_alignment'] = new_use_for_alignment

         iteration += 1
      else:
         convergence = True
   
   if (n_images > 1) and use_parallel:
      pool.close()

   Gaia_HST_table = Gaia_HST_table[Gaia_HST_table['relative_hst_gaia_pmdec_%s'%use_mean].notnull() & Gaia_HST_table['relative_hst_gaia_pmra_%s'%use_mean].notnull()]

   return Gaia_HST_table


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


def weighted_avg_err(table):
   """
   Weighted average its error and the standard deviation.
   """

   var_cols = [x for x in table.columns if not '_error' in x]
   
   x_i = table.loc[:, var_cols]
   ex_i = get_errors(table, used_cols = var_cols)
   ex_i.columns = ex_i.columns.str.rstrip('_error')

   weighted_variance = (1./(1./ex_i**2).sum(axis = 0))
   weighted_avg = ((x_i.div(ex_i.values**2)).sum(axis = 0) * weighted_variance.values).add_suffix('_wmean')
   weighted_avg_error = np.sqrt(weighted_variance[~weighted_variance.index.duplicated()]).add_suffix('_wmean_error')

   avg = x_i.mean().add_suffix('_mean')
   avg_error = x_i.std().add_suffix('_mean_error')/np.sqrt(len(x_i))
   std = x_i.std().add_suffix('_std')

   return pd.concat([weighted_avg, weighted_avg_error, avg, avg_error, std])


def absolute_pm(table):
   """
   This routine computes the absolute PM just adding the absolute differences between Gaia and HST PMs.
   """   

   pm_differences_wmean = table.loc[:, ['pmra', 'pmdec']] - table.loc[:, ['relative_hst_gaia_pmra_wmean', 'relative_hst_gaia_pmdec_wmean']].values
   pm_differences_wmean_error = np.sqrt(table.loc[:, ['pmra_error', 'pmdec_error']]**2 + table.loc[:, ['relative_hst_gaia_pmra_wmean_error', 'relative_hst_gaia_pmdec_wmean_error']].values**2)

   pm_differences_mean = table.loc[:, ['pmra', 'pmdec']] - table.loc[:, ['relative_hst_gaia_pmra_mean', 'relative_hst_gaia_pmdec_mean']].values
   pm_differences_mean_error = np.sqrt(table.loc[:, ['pmra_error', 'pmdec_error']]**2 + table.loc[:, ['relative_hst_gaia_pmra_mean_error', 'relative_hst_gaia_pmdec_mean_error']].values**2)

   pm_differences_weighted = weighted_avg_err(pm_differences_wmean.join(pm_differences_wmean_error))
   pm_differences = weighted_avg_err(pm_differences_mean.join(pm_differences_mean_error))

   table['hst_gaia_pmra_wmean'], table['hst_gaia_pmdec_wmean'] = table.relative_hst_gaia_pmra_wmean + pm_differences_weighted.pmra_wmean, table.relative_hst_gaia_pmdec_wmean + pm_differences_weighted.pmdec_wmean
   table['hst_gaia_pmra_wmean_error'], table['hst_gaia_pmdec_wmean_error'] = np.sqrt(table.relative_hst_gaia_pmra_wmean_error**2 + pm_differences_weighted.pmra_wmean_error**2), np.sqrt(table.relative_hst_gaia_pmdec_wmean_error**2 + pm_differences_weighted.pmdec_wmean_error**2)

   table['hst_gaia_pmra_mean'], table['hst_gaia_pmdec_mean'] = table.relative_hst_gaia_pmra_mean + pm_differences.pmra_mean, table.relative_hst_gaia_pmdec_mean + pm_differences.pmdec_mean
   table['hst_gaia_pmra_mean_error'], table['hst_gaia_pmdec_mean_error'] = np.sqrt(table.relative_hst_gaia_pmra_mean_error**2 + pm_differences.pmra_mean_error**2), np.sqrt(table.relative_hst_gaia_pmdec_mean_error**2 + pm_differences.pmdec_mean_error**2)

   return table


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


def plot_results(table, hst_image_list, HST_path, use_mean = 'wmean', plot_name_1 = 'output1', plot_name_2 = 'output2', plot_name_3 = 'output3', plot_name_4 = 'output4', ext = '.pdf'):
   """
   Plot results
   """

   pmra_lims = [table['hst_gaia_pmra_%s'%use_mean].mean()-5*table['hst_gaia_pmra_%s'%use_mean].std(), table['hst_gaia_pmra_%s'%use_mean].mean()+5*table['hst_gaia_pmra_%s'%use_mean].std()]
   pmdec_lims = [table['hst_gaia_pmdec_%s'%use_mean].mean()-5*table['hst_gaia_pmdec_%s'%use_mean].std(), table['hst_gaia_pmdec_%s'%use_mean].mean()+5*table['hst_gaia_pmdec_%s'%use_mean].std()]
   
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


   ax2.plot(table['hst_gaia_pmra_%s'%use_mean][table.use_for_alignment == False], table['hst_gaia_pmdec_%s'%use_mean][table.use_for_alignment == False], 'r.', ms =1, alpha = 0.35)
   ax2.plot(table['hst_gaia_pmra_%s'%use_mean][table.use_for_alignment == True], table['hst_gaia_pmdec_%s'%use_mean][table.use_for_alignment == True], 'r.', ms =1)
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

   ax3.plot(table.gmag[table.use_for_alignment == False], np.sqrt(table['hst_gaia_pmra_%s_error'%use_mean][table.use_for_alignment == False]**2+table['hst_gaia_pmdec_%s_error'%use_mean][table.use_for_alignment == False]**2), 'r.', ms = 1, alpha = 0.35)
   ax3.plot(table.gmag[table.use_for_alignment == True], np.sqrt(table['hst_gaia_pmra_%s_error'%use_mean][table.use_for_alignment == True]**2+table['hst_gaia_pmdec_%s_error'%use_mean][table.use_for_alignment == True]**2), 'r.', ms = 1)
   ax3.grid()
   try:
      ax3.set_ylim(0, np.sqrt(table.pmra_error**2+table.pmdec_error**2).max())
   except:
      pass
   ax3.set_xlabel('Gmag')
   ax3.set_ylabel(r'$\sqrt{\sigma(\mu_{\alpha*})^2 + \sigma(\mu_{\delta})^2 }$ [m.a.s./yr.]')

   plt.subplots_adjust(wspace=0.25, hspace=0.1)

   plt.savefig(plot_name_1+ext, bbox_inches='tight')

   plt.close('all')

   fig2, (ax1, ax2) = plt.subplots(1,2, sharex = False, sharey = False, figsize = (12, 3.5))
   ax1.errorbar(table.pmra, table['hst_gaia_pmra_%s'%use_mean], xerr=table.pmra_error, yerr=table['hst_gaia_pmra_%s_error'%use_mean], fmt = '.', ms=2, color = '0.1', zorder = 1, alpha = 0.5, elinewidth = 0.5)
   ax1.plot([pmra_lims[0], pmra_lims[1]], [pmra_lims[0], pmra_lims[1]], 'r-', linewidth = 0.5)
   ax1.grid()
   ax1.set_xlabel(r'Gaia $\mu_{\alpha*}$ [m.a.s./yr.]')
   ax1.set_ylabel(r'HST + Gaia $\mu_{\alpha*}$ [m.a.s./yr.]')
   try:
      ax1.set_xlim(pmra_lims)
      ax1.set_ylim(pmra_lims)      
   except:
      pass

   ax2.errorbar(table.pmdec, table['hst_gaia_pmdec_%s'%use_mean], xerr=table.pmdec_error, yerr=table['hst_gaia_pmdec_%s_error'%use_mean], fmt = '.', ms=2, color = '0.1', zorder = 1, alpha = 0.5, elinewidth = 0.5)
   ax2.plot([pmdec_lims[0], pmdec_lims[1]], [pmdec_lims[0], pmdec_lims[1]], 'r-', linewidth = 0.5)
   ax2.grid()
   ax2.set_xlabel(r'Gaia $\mu_{\delta}$ [m.a.s./yr.]')
   ax2.set_ylabel(r'HST + Gaia $\mu_{\delta}$ [m.a.s./yr.]')
   try:
      ax2.set_xlim(pmdec_lims)
      ax2.set_ylim(pmdec_lims)      
   except:
      pass

   plt.savefig(plot_name_2+ext, bbox_inches='tight')

   plt.close('all')


   plt.close('all')

   fig2, ax = plt.subplots(1,1, sharex = False, sharey = False, figsize = (5.5, 5.5))

   hst_filters = [col for col in table.columns if ('F' in col) & ('error' not in col) & ('std' not in col) & ('_mean' not in col)]
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

   plt.savefig(plot_name_3+ext, bbox_inches='tight')


   plt.close('all')
   
   try:
      hdu_list = []
      for index_image, (obs_id, HST_image) in hst_image_list.loc[:, ['obs_id', 'productFilename']].iterrows():

         HST_image_filename = HST_path+'mastDownload/HST/'+obs_id+'/'+HST_image

         hdu_list.append(fits.open(HST_image_filename)[1])

      if len(hdu_list) > 1:
         from reproject.mosaicking import find_optimal_celestial_wcs
         from reproject import reproject_interp
         from reproject.mosaicking import reproject_and_coadd
         
         wcs, shape = find_optimal_celestial_wcs(hdu_list, resolution=0.5 * u.arcsec)
         image_data, image_footprint = reproject_and_coadd(hdu_list, wcs, shape_out=shape, reproject_function=reproject_interp, order = 'nearest-neighbor', match_background=True)

      else:
         wcs = WCS(hdu_list[0].header)
         image_data = hdu_list[0].data

      fig = plt.figure(figsize=(10,10),dpi=250)
      ax = fig.add_subplot(111, projection=wcs)

      norm = ImageNormalize(image_data, interval = ManualInterval(0.0,0.15))

      im = ax.imshow(image_data, cmap='gray_r', origin='lower', norm=norm)

      ax.set_xlabel("RA")
      ax.set_ylabel("Dec")

      try:
         p1 = ax.scatter(table['ra'][table.use_for_alignment == False],table['dec'][table.use_for_alignment == False], transform=ax.get_transform('world'), s=20, linewidth = 1, edgecolor='salmon', facecolor='none', label='Not used')
      except:
         pass
      try:
         with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q1 = ax.quiver(table['ra'][table.use_for_alignment == False], table['dec'][table.use_for_alignment == False], -table['hst_gaia_pmra_%s'%use_mean][table.use_for_alignment == False], table['hst_gaia_pmdec_%s'%use_mean][table.use_for_alignment == False], transform=ax.get_transform('world'), color='salmon', width = 0.003, angles = 'xy')
      except:
         pass

      try:
         p2 = ax.scatter(table['ra'][table.use_for_alignment == True],table['dec'][table.use_for_alignment == True], transform=ax.get_transform('world'), s=20, linewidth = 1, edgecolor='limegreen', facecolor='none', label='Used')
      except:
         pass
      try:
         with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q2 = ax.quiver(table['ra'][table.use_for_alignment == True],table['dec'][table.use_for_alignment == True], table['hst_gaia_pmra_%s'%use_mean][table.use_for_alignment == True], table['hst_gaia_pmdec_%s'%use_mean][table.use_for_alignment == True], transform=ax.get_transform('world'), color='limegreen', width = 0.003, angles = 'xy')
      except:
         pass

      ax.grid()

      if table.use_for_alignment.sum() < len(table):
         plt.legend()

      plt.savefig(plot_name_4+'.png', bbox_inches='tight')

   except:
      print('A problem was found when generating the footprint.')


def get_object_properties(args):
   """
   This routine will try to obtain all the required object properties from Simbad or from the user.
   """

   #Try to get object:
   if (args.ra is None) or (args.dec is None):
      try:
         from astroquery.simbad import Simbad
         import astropy.units as u
         from astropy.coordinates import SkyCoord

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

         #Try to get radius
         if args.max_search_radius is None:
            if (object_table['GALDIM_MAJAXIS'].mask == False):
               args.max_search_radius = max(2.0 * np.round(float(2. * object_table['GALDIM_MAJAXIS'] / 60.), 2), 0.1)
            else:
               try:
                  args.max_search_radius = float(input('Search radius not defined, please enter the search radius in degrees (Press enter to adopt the default value of 1 deg): '))
               except:
                  args.max_search_radius = 1.0

         if (args.search_type == 'anulus') and (args.min_search_radius is None):
            if (object_table['GALDIM_MAJAXIS'].mask == False):
               args.min_search_radius = max(0.5 * np.round(float(2. * object_table['GALDIM_MAJAXIS'] / 60.), 2), 0.1)
            else:
               if not args.silent:
                  try:
                     args.min_search_radius = float(input('Inner radius of the anulus search not defined, please enter the inner radius in degrees (Press enter to adopt the default value of 0.5 deg): '))
                  except:
                     args.min_search_radius = 0.5

         #We try to get PMs:
         if any((args.min_pmra == None, args.max_pmra == None)):
            if (object_table['PMRA'].mask == False):
               args.pmra = float(object_table['PMRA'])
            else:
               if not args.silent:
                  try:
                     args.pmra = float(input('PMRA of the objec not defined, please enter PMRA in mas/yr (Press enter to ignore): '))
                  except:
                     args.pmra = 0.

         if any((args.min_pmdec == None, args.max_pmdec == None)):
            if (object_table['PMDEC'].mask == False):
               args.pmdec = float(object_table['PMDEC'])
            else:
               if not args.silent:
                  try:
                     args.pmdec = float(input('PMDEC of the object not defined, please enter PMDEC in mas/yr (Press enter to ignore): '))
                  except:
                     args.pmdec = 0.

      except:
         if args.ra is None:
            args.ra = float(input('R.A. not defined, please enter R.A. in degrees: '))
         if args.dec is None:
            args.dec = float(input('Dec not defined, please enter Dec in degrees: '))

         if not args.silent:
            #Try to get radius
            if args.max_search_radius is None:
               try:
                  args.max_search_radius = float(input('Search radius not defined, please enter the search radius in degrees (Press enter to adopt the default value of 1 deg): '))
               except:
                  args.max_search_radius = 1.0

            if (args.search_type == 'anulus') and (args.min_search_radius is None):
               try:
                  args.min_search_radius = float(input('Inner radius of the anulus search not defined, please enter the inner radius in degrees (Press enter to adopt the default value of 0.5 deg): '))
               except:
                  args.min_search_radius = 0.5


   if args.max_pmra is None:
      args.max_pmra = 999.
   if args.min_pmra is None:
      args.min_pmra = -999.
   if args.pmra is None:
      args.pmra = 0.

   if args.max_pmdec is None:
      args.max_pmdec = 999.
   if args.min_pmdec is None:
      args.min_pmdec = -999.
   if args.pmdec is None:
      args.pmdec = 0.

   if args.min_parallax is None:
      args.min_parallax = -999.
   if args.max_parallax is None:
      args.max_parallax = 999.
   if args.parallax is None:
      args.parallax = 0.

   if args.max_search_radius is None:
      args.max_search_radius = 1.0
   if args.min_search_radius is None:
      args.min_search_radius = 0.5

   if (args.search_type == 'box'):
      if (args.search_height is None):
         try:
            args.search_height = 2.*args.max_search_radius
         except:
            args.search_height = 0.5
      if (args.search_width is None):
         try:
            args.search_width = np.abs(2.*args.max_search_radius/np.cos(np.deg2rad(args.dec)))
         except:
            args.search_width = 0.5

   setattr(args, 'area', get_area(args.search_type, args.max_search_radius, args.min_search_radius, args.search_width, args.search_height, args.dec))
   setattr(args, 'download_radius', args.max_search_radius)

   if args.error_weighted:
      args.use_mean = 'wmean'
   else:
      args.use_mean = 'mean'

   name_coo = 'ra_%.3f_dec_%.3f_r_%.2f'%(args.ra, args.dec, args.max_search_radius)

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
   args.Gaia_ind_queries_path = args.Gaia_path+'individual_queries/'
   
   args.used_HST_obs_table_filename = args.base_path + args.base_file_name+'_used_HST_images.csv'
   args.HST_Gaia_table_filename = args.base_path + args.base_file_name+'.csv'
   args.logfile = args.base_path + args.base_file_name+'.log'
   args.queries = args.Gaia_path + args.base_file_name+'_queries.log'
   
   args.Gaia_raw_table_filename = args.Gaia_path + args.base_file_name+'_raw.csv'
   args.Gaia_raw_sel_table_filename = args.Gaia_path + args.base_file_name+'_raw_selection.csv'
   args.HST_obs_table_filename = args.HST_path + args.base_file_name+'_obs.csv'
   args.HST_data_table_products_filename = args.HST_path + args.base_file_name+'_data_products.csv'

   args.date_second_epoch = Time('%4i-%02i-%02iT00:00:00.000'%(args.date_second_epoch[2], args.date_second_epoch[0], args.date_second_epoch[1])).mjd
   args.date_reference_second_epoch = Time(args.date_reference_second_epoch).mjd

   print('\n')
   print(' USED PARAMETERS '.center(42, '*'))
   print('- (ra, dec) = (%s, %s) deg.'%(round(args.ra, 5), round(args.dec, 5)))
   print('- pmra = [%s, %s] m.a.s./yr.'%(round(args.min_pmra, 5), round(args.max_pmra, 5)))
   print('- pmdec = [%s, %s] m.a.s./yr.'%(round(args.min_pmdec, 5), round(args.max_pmdec, 5)))
   print('- parallax = [%s, %s] m.a.s.'%(round(args.min_parallax, 5), round(args.max_parallax, 5)))
   print('- radius = %s deg.'%args.max_search_radius)
   print('*'*42+'\n')

   return args


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


def main(argv):  
   """
   Inputs
   """

   parser = argparse.ArgumentParser(description="This script derives proper motions (PM) combining HST and Gaia data.")
   
   # Object info
   parser.add_argument('--name', type=str, default = 'Output', help='Name for the Output table.')
   parser.add_argument('--ra', type=float, default = None, help='Central R.A.')
   parser.add_argument('--dec', type=float, default = None, help='Central Dec.')
   parser.add_argument('--distance', type=float, default = None, help='Distance to the object.')
   parser.add_argument('--pmra', type=float, default= None, help='Proper motion of the object in R.A., if known, in mas. Default will try to find the info in Simbad or use the middle value between "min_pmra" and "max_pmra"')
   parser.add_argument('--pmdec', type=float, default= None, help='Proper motion in Dec. of the object, if known, in mas. Default will try to find the info in Simbad or use the middle value between "min_pmra" and "max_pmra"')
   parser.add_argument('--parallax', type=float, default=None, help='Parallax of the object, if known, in mas. Default will try to find the info in Simbad or use the middle value between "min_parallax" and "max_parallax"')
   parser.add_argument('--AV', type=float, default = None, help='Reddening in mag (AV).')
   parser.add_argument('--age', type=float, nargs='+', default= [12.], help='Age of the system in Gyr. Both, a single value or a range can be provided. Default is age within [8., 13.7].')
   parser.add_argument('--age_step', type=float, default= 0.1, help='Age resolution.')
   parser.add_argument('--age_mode', type=str, default= "discrete", help="If 'discrete', only the ages specified will be used. If 'continuous', ages between the max and min of --age will be used every --age_step")
   parser.add_argument('--feh', type=float, nargs='+', default= None, help='Metallicity ([Fe/H]) of the system. Both, a single value or a range can be provided. Default is range [Fe/H] within [-2.5, -0.5].')

   # Membership selection
   parser.add_argument('--use_members', type=str2bool, default=True, help='Whether to use only member stars for the epochs alignment or to use all available stars.')
   parser.add_argument('--force_manual_cmd_cleaning', type=str2bool, default=False, help='Force the program to use manually selected stars from the CMD. Default False.')
   parser.add_argument('--cmd_broadening', type=float, default=0.1, help='CMD intrinsic color broadening in magnitudes. It is used to compute the maximum distance in color to a star as to consider it as possible bember of an isochrone population. Default is 0.1.')
   parser.add_argument('--clipping_sigma_cmd', type=float, default=6., help='Sigma used for clipping in the cmd. i.e. distance to the isochrone. Default is 3.')
   parser.add_argument('--extend_HB', type=str2bool, default=False, help='Whether to extend the HB of the isochrones in order to cover extremely low-metallicity populations.')
   parser.add_argument('--prepare_for_clustering', type = float, default = 0, help = 'Preselect sources before the last clustering')
   parser.add_argument('--clipping_prob_pm', type=float, default=3., help='Sigma used for clipping pm and parallax. Default is 3.')
   parser.add_argument('--pm_n_components', type=int, default=2, help='Number of Gaussian componnents for pm and parallax clustering. Default is 1.')

   # Search options
   parser.add_argument('--search_type', type=str, default = 'box', help='Shape of the area to search. Options are "box", "cone" or "anulus". The "box" size is controlled by the "search_width" and "search_height" parameters. The "cone" radius is controlled by the "search_radius" parameter.')
   parser.add_argument('--search_width', type=float, default = None, help='Width for the cone search in degrees.')
   parser.add_argument('--search_height', type=float, default = None, help='Height for the cone search in degrees.')
   parser.add_argument('--max_search_radius', type=float, default = None, help='Radius of search in degrees.')
   parser.add_argument('--min_search_radius', type=float, default = None, help='Inner radius for the cone search in degrees. Useful for anulus search.')
   parser.add_argument('--field_radius', type=float, default = None, help='Outer radius for the annular region used as control sample. By default is "max_search_radius" + 0.25 deg.')
   parser.add_argument('--min_pmra', type=float, default= None, help='Min pmra in mas.')
   parser.add_argument('--max_pmra', type=float, default = None, help='Max pmra in mas.')
   parser.add_argument('--max_pmra_error', type=float, default = 1.5, help='Max error in pmra in mas.')
   parser.add_argument('--min_pmdec', type=float, default= None, help='Min pmdec in mas.')
   parser.add_argument('--max_pmdec', type=float, default = None, help='Max pmdec in mas.')
   parser.add_argument('--max_pmdec_error', type=float, default = 1.5, help='Max error in pmra in mas.')
   parser.add_argument('--min_parallax', type=float, default= None, help='Min parallax in mas.')
   parser.add_argument('--max_parallax', type=float, default = None, help='Max parallax in mas.')
   parser.add_argument('--max_parallax_error', type=float, default = 1.5, help='Max error in parallax in mas.')
   parser.add_argument('--min_gmag', type=float, default = 16.0, help='Brighter G magnitude')
   parser.add_argument('--max_gmag', type=float, default = 21.5, help='Fainter G magnitude')
   parser.add_argument('--max_bpmag', type=float, default = 21.5, help='Fainter BP magnitude')
   parser.add_argument('--max_rpmag', type=float, default = 21.5, help='Fainter RP magnitude')
   parser.add_argument('--min_bp_rp', type=float, default = -2.0, help='Bluest color.')
   parser.add_argument('--max_bp_rp', type=float, default = 5.0, help='Reddest color')
   parser.add_argument('--max_gmag_error', type=float, default = 0.02, help='Max error in G magnitude.')
   parser.add_argument('--max_rpmag_error', type=float, default = 0.1, help='Max error in RP magnitude.')
   parser.add_argument('--max_bpmag_error', type=float, default = 0.1, help='Max error in BP magnitude.')

   # Gaia options
   parser.add_argument('--gaia_user', type=str, default = None, help='Gaia username. Useful for automatization of the script.')
   parser.add_argument('--gaia_paswd', type=str, default = None, help='Gaia password. Useful for automatization of the script.')
   parser.add_argument('--clean_uwe', type = str2bool, default = True)
   parser.add_argument('--norm_uwe', type = str2bool, default = True)
   parser.add_argument('--source_table', type = str, default = 'gaiaedr3.gaia_source', help='Gaia source table. Default is gaiaedr3.gaia_source.')
   parser.add_argument('--save_individual_queries', type = str2bool, default = True, help='If True, the code will save the individual queries.')
   parser.add_argument('--remove_quality_cols', type = str2bool, default = False, help='If True, the code will remove all quality columns from the final table, except "clean_label".')
   parser.add_argument('--clean_data', type = str2bool, default = False, help = 'Screen out bad measurements based on Gaia EDR3 quality flags. Default is False.')
   parser.add_argument('--sigma_flux_excess_factor', type=float, default=3., help='Sigma used for clipping in flux_excess_factor. Default is 3.')
   parser.add_argument('--only_5p_solutions', type = str2bool, default = False, help='If True, only 5p solution stars will be used. Default is False.')
   parser.add_argument('--date_second_epoch', type=int, nargs='+', default= [5, 28, 2017], help='Second epoch adquisition date. Default is Gaia EDR3 (05-28-2017).')
   parser.add_argument('--date_reference_second_epoch', type=str, default= 'J2016.0', help='Second epoch reference date. Default is Gaia EDR3 J2016.0.')

   # HST options
   parser.add_argument('--force_hst1pass', type=str2bool, default=False, help='Force the program to perform the search for sources in the HST images. Default is False, which will use existing files if any.')
   parser.add_argument('--fmin', type=int, default= None, help='Minimum flux above the sky to substract a source in the HST image. Default is automatic from HST integration time.')
   parser.add_argument('--pixel_scale', type=float, default= None, help='Pixel scale in arcsec/pixel used to compute the tangential plane during the match between epochs. Default is automatic from HST images.')
   parser.add_argument('--hst_filters', type=str, nargs='+', default = ['any'], help='Required filter for the HST images.')
   parser.add_argument('--hst_integration_time_min', type=float, default = 50, help='Required integration time for the HST images.')
   parser.add_argument('--hst_integration_time_max', type=float, default = 550, help='Required integration time for the HST images.')
   parser.add_argument('--time_baseline', type=float, default = 1460, help='Minimum time baseline with respect to Gaia EDR3 in days. Default 1460.')

   # HST-Gaia match options
   parser.add_argument('--force_xym2pm', type=str2bool, default=False, help='Force the program to perform the match between Gaia and HST sources. Default is False, which will use existing files if any.')
   parser.add_argument('--fix_mat', type=str2bool, default=True, help='Force to select the best stars in the MAT files to perform the alignment in the next iteration.')
   parser.add_argument('--max_separation', type=float, default= None, help='Maximum allowed separation in pixels during the match between epochs. Default 10 pixels.')
   parser.add_argument('--force_use_sat', type=str2bool, default=True, help='Force the program to use saturated stars during the match between epochs. Default is True.')
   parser.add_argument('--force_wcs_search_radius', type=float, default= None, help='When set to a radius (in arcsec), the program search the closest Gaia star to each bright star in the HST image within that distance to perform a pre-alignment between the two frames. Useful when not many stars are available. Default is None.')
   parser.add_argument('--min_stars_alignment', type=int, default = 25, help='Minimum number of stars per HST image to be used for the epochs alignment. Default 25.')
   parser.add_argument('--force_use_amp', type=str2bool, default=False, help='Force the program to use amplifier-based transformations. Default is False.')
   parser.add_argument('--min_stars_amp', type=int, default = 15, help='Minimum number of stars per HST amplifier to compute transformations. Default is 20.')

   #Miscellaneus options
   parser.add_argument('--use_parallel', type = str2bool, default = True, help='Use parallelized computation when possible. Default is True.')
   parser.add_argument('--load_existing', type = str2bool, default = False, help='If True, the code will try to resume the previous search loading previous individual queries. It should be set to False if a new table is being downloaded. True when a specific search is failing due to connection problems.')
   parser.add_argument('--plots', type=str2bool, default=True, help='Create sanity plots. Default is True.')
   parser.add_argument('--silent', type=str2bool, default = False, help='Accept all default values without asking. Default is False.')
   parser.add_argument('--error_weighted', type = str2bool, default = True, help = 'The program will use error-weighted statistics to compute average PMs, if possible.')
   parser.add_argument('--remove_previous_files', type=str2bool, default=True, help='Remove previous intermediate files.')
   parser.add_argument('--verbose', type=str2bool, default=True, help='Program verbosity. Default True.')
   parser.add_argument('--test_mode', type = str2bool, default = False)

   args = parser.parse_args(argv)

   args = get_object_properties(args)

   """
   The script creates directories and set files names
   """

   create_dir(args.base_path)
   create_dir(args.HST_path)
   create_dir(args.Gaia_path)
   if args.save_individual_queries:
      create_dir(args.Gaia_ind_queries_path)

   """
   The script tries to load an existing Gaia table, otherwise it will download it from the Gaia archive.
   """

   astrometric_cols = 'l, b, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, dr2_radial_velocity, dr2_radial_velocity_error, ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr'

   photometric_cols = 'phot_g_mean_flux, phot_g_mean_mag AS gmag, (1.086*phot_g_mean_flux_error/phot_g_mean_flux) AS gmag_error, phot_bp_mean_mag AS bpmag, (1.086*phot_bp_mean_flux_error/phot_bp_mean_flux) AS bpmag_error, phot_rp_mean_mag AS rpmag, (1.086*phot_rp_mean_flux_error/phot_rp_mean_flux) AS rpmag_error, bp_rp, sqrt( power( (1.086*phot_bp_mean_flux_error/phot_bp_mean_flux), 2) + power( (1.086*phot_rp_mean_flux_error/phot_rp_mean_flux), 2) ) as bp_rp_error'

   quality_cols = 'ecl_lat, pseudocolour, nu_eff_used_in_astrometry, visibility_periods_used, astrometric_excess_noise_sig, astrometric_params_solved, astrometric_n_good_obs_al, astrometric_chi2_al, phot_bp_rp_excess_factor, ruwe, (phot_bp_n_blended_transits+phot_rp_n_blended_transits) *1.0 / (phot_bp_n_obs + phot_rp_n_obs) AS beta, ipd_gof_harmonic_amplitude, phot_bp_n_contaminated_transits, phot_rp_n_contaminated_transits'

   query, quality_cols = columns_n_conditions(args.source_table, args.search_type, astrometric_cols, photometric_cols, quality_cols, args.ra, args.dec,
                                              args.min_search_radius, args.download_radius, args.search_width, args.search_height,
                                              max_gmag_error = args.max_gmag_error, max_rpmag_error = args.max_rpmag_error,
                                              max_bpmag_error = args.max_bpmag_error, min_parallax = args.min_parallax, max_parallax = args.max_parallax,
                                              max_parallax_error = args.max_parallax_error, min_pmra = args.min_pmra, max_pmra = args.max_pmra,
                                              max_pmra_error = args.max_pmra_error, min_pmdec = args.min_pmdec, max_pmdec = args.max_pmdec, max_pmdec_error = args.max_pmdec_error)
   try:
      Gaia_table = pd.read_csv(args.Gaia_raw_table_filename)
   except:
      Gaia_table, Gaia_queries = incremental_query(query, args.area, min_gmag = args.min_gmag, max_gmag = args.max_gmag, norm_uwe = args.norm_uwe, use_parallel = args.use_parallel,
                                                   test_mode = args.test_mode, save_individual_queries = args.save_individual_queries, name = args.name, gaia_user = args.gaia_user, gaia_paswd = args.gaia_paswd)

      Gaia_table.to_csv(args.Gaia_raw_table_filename, index = False)

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
   zpt.load_tables()
   Gaia_table['corrected_parallax'] = Gaia_table['parallax'] - zpt.get_zpt(Gaia_table['gmag'], Gaia_table['nu_eff_used_in_astrometry'], Gaia_table['pseudocolour'], Gaia_table['ecl_lat'], Gaia_table['astrometric_params_solved'])
   Gaia_table['corrected_gmag'], Gaia_table['corrected_phot_g_mean_flux'] = correct_gband(Gaia_table['bp_rp'], Gaia_table['astrometric_params_solved'], Gaia_table['gmag'], Gaia_table['phot_g_mean_flux'])
   Gaia_table['corrected_flux_excess_factor'] = correct_flux_excess_factor(Gaia_table['bp_rp'], Gaia_table['phot_bp_rp_excess_factor'])

   clean_label = pre_clean_data(Gaia_table['gmag'], Gaia_table['corrected_flux_excess_factor'], Gaia_table['ruwe'], Gaia_table['ipd_gof_harmonic_amplitude'], Gaia_table['visibility_periods_used'], Gaia_table['astrometric_excess_noise_sig'], Gaia_table['astrometric_params_solved'], sigma_flux_excess_factor = args.sigma_flux_excess_factor, use_5p = True)

   Gaia_table['clean_label'] = clean_label

   Gaia_table = get_real_error(Gaia_table)
   Gaia_table = select_conditions(args, Gaia_table)

   Gaia_table.to_csv(args.Gaia_raw_sel_table_filename)

   if args.clean_data:
      Gaia_table = Gaia_table[Gaia_table.clean_label == True]


   if args.use_members:
      if (args.distance is not None) and (args.force_manual_cmd_cleaning is False):
         """
         If distance is defined, the code will attempt to first select stars using isochrones.
         """
         isochrones = read_isochrones(args.age, args.z, max_gmag = args.max_gmag)
         isochrones_cmd = combine_isochrones(isochrones, cmd_broadening = args.cmd_broadening, extended_HB = args.extend_HB)
         Gaia_table['member_cmd_gaia'] = cmd_cleaning(Gaia_table.copy(), isochrones_cmd, distance = args.distance, AV = args.AV, clipping_sigma = args.clipping_sigma_cmd, plots = args.plots, plot_name = args.Gaia_path+'CMD_selection.png')

      else:
         Gaia_table['member_cmd_gaia'] = manual_select_from_cmd(Gaia_table.bp_rp, Gaia_table.gmag)

      """
      Perform the selection in the PM-parallax space.
      """
      Gaia_table['clustering_data'] = Gaia_table['member_cmd_gaia']
      Gaia_table['member_pm_gaia'] = pm_cleaning_GMM_recursive(Gaia_table.copy(), ['pmra', 'pmdec', 'parallax'], data_0 = [args.pmra, args.pmdec, args.parallax], n_components = args.pm_n_components, clipping_prob = args.clipping_prob_pm, plots = args.plots, plot_name = args.Gaia_path+'PM_selection')
      
      if args.clean_data:
         gaia_selection_vars = ['member_cmd_gaia', 'member_pm_gaia', 'clean_label']
      else:
         gaia_selection_vars = ['member_cmd_gaia', 'member_pm_gaia']
      
      Gaia_table['use_for_alignment'] = (Gaia_table.loc[:, gaia_selection_vars] == True).all(axis = 1)
   else:
      Gaia_table['use_for_alignment'] = True

   """
   The script tries to load an existing HST table, otherwise it will download it from the MAST archive.
   """
   obs_table, data_products_by_obs = search_mast(args.ra, args.dec, args.search_width, args.search_height, filters = args.hst_filters, t_exptime_min = args.hst_integration_time_min, t_exptime_max = args.hst_integration_time_max, date_second_epoch = args.date_second_epoch, time_baseline = args.time_baseline)

   obs_table.to_csv(args.HST_obs_table_filename, index = False)
   data_products_by_obs.to_csv(args.HST_data_table_products_filename, index = False)

   """
   Plot results and find Gaia stars within HST fields
   """
   Gaia_table, obs_table = plot_fields(Gaia_table, obs_table, args.HST_path, min_stars_alignment = args.min_stars_alignment, name = args.base_path+args.base_file_name+'_search_footprint.png')

   if len(obs_table) > 0:

      """
      Ask whether the user wish to download the available HST images 
      """

      print('Would you like to use the following HST observations?\n')
      print(obs_table.loc[:, ['obsid', 'filters', 'n_exp', 'i_exptime', 'obs_time', 't_baseline', 'gaia_stars_per_obs', 'proposal_id', '']].to_string(index=False), '\n')

      print("Type 'y' for all observations, 'n' for none. Type the number within parentheses at the right if you wish to use that specific set of observations. You can enter several numbers separated by space. \n")
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
         if HST_obs_to_use is True:
            HST_obs_to_use = list(obs_table['obsid'].values)
         hst_images = download_HST_images(Table.from_pandas(data_products_by_obs.loc[data_products_by_obs['parent_obsid'].isin(HST_obs_to_use), :]), path = args.HST_path)
      else:
         print('\nExiting now.\n')
         sys.exit(1)

      """
      Select only flc
      """
      drz_images = data_products_by_obs[(data_products_by_obs['productSubGroupDescription'] == 'DRZ') & (data_products_by_obs['parent_obsid'].isin(HST_obs_to_use))]
      flc_images = data_products_by_obs[(data_products_by_obs['productSubGroupDescription'] == 'FLC') & (data_products_by_obs['parent_obsid'].isin(HST_obs_to_use))]

      """
      Call hst1pass
      """
      launch_hst1pass(flc_images, HST_obs_to_use, args.HST_path, force_fmin = args.fmin, remove_previous_XYmqxyrd = args.remove_previous_files, force_hst1pass = args.force_hst1pass, use_parallel = args.use_parallel)
      """
      Call xym2pm_Gaia
      """
      Gaia_table_hst = launch_xym2pm_Gaia(Gaia_table.copy(), flc_images, HST_obs_to_use, args.HST_path, args.date_reference_second_epoch, only_use_members = args.use_members, force_pixel_scale = args.pixel_scale, force_max_separation = args.max_separation, force_use_sat = args.force_use_sat, fix_mat = args.fix_mat, force_use_amp = args.force_use_amp, min_stars_amp = args.min_stars_amp, force_wcs_search_radius = args.force_wcs_search_radius, n_components = args.pm_n_components, clipping_prob = args.clipping_prob_pm, min_stars_alignment = args.min_stars_alignment, use_mean = args.use_mean, plots = args.plots, verbose = args.verbose, force_xym2pm = args.force_xym2pm, remove_previous_files = args.remove_previous_files, use_parallel = args.use_parallel, plot_name = args.base_path+'PM_selection')

      """
      Obtain absolute PMs
      """
      Gaia_table_hst = absolute_pm(Gaia_table_hst.copy())

      """
      Save Gaia and HST tables
      """
      obs_table.to_csv(args.HST_obs_table_filename, index = False)
      data_products_by_obs.to_csv(args.HST_data_table_products_filename, index = False)

      flc_images.to_csv(args.used_HST_obs_table_filename, index = False)
      Gaia_table_hst.to_csv(args.HST_Gaia_table_filename, index = False)

      avg_pm = weighted_avg_err(Gaia_table_hst.loc[Gaia_table_hst.use_for_alignment, ['hst_gaia_pmra_%s'%args.use_mean, 'hst_gaia_pmdec_%s'%args.use_mean, 'hst_gaia_pmra_%s_error'%args.use_mean, 'hst_gaia_pmdec_%s_error'%args.use_mean]])

      """
      Print a summary with the location of files and plot the results
      """
      plot_results(Gaia_table_hst, drz_images, args.HST_path, use_mean = args.use_mean, plot_name_1 = args.base_path+args.base_file_name+'_vpd', plot_name_2 = args.base_path+args.base_file_name+'_diff', plot_name_3 = args.base_path+args.base_file_name+'_cmd', plot_name_4 = args.base_path+args.base_file_name+'_footprint', ext = '.pdf')

      logresults = ' RESULTS '.center(82, '-')+'\n - Final table: %s'%args.HST_Gaia_table_filename+'\n - Used HST observations: %s'%args.used_HST_obs_table_filename+'\n'+'-'*82+'\n - A total of %i stars were used.\n'%Gaia_table_hst.use_for_alignment.sum() +' - Average absolute PM of used stars: \n   pmra = %s+-%s \n'%(round_significant(avg_pm['hst_gaia_pmra_%s_%s'%(args.use_mean, args.use_mean)], avg_pm['hst_gaia_pmra_%s_%s_error'%(args.use_mean, args.use_mean)]))+'   pmdec = %s+-%s \n '%(round_significant(avg_pm['hst_gaia_pmdec_%s_%s'%(args.use_mean, args.use_mean)], avg_pm['hst_gaia_pmdec_%s_%s_error'%(args.use_mean, args.use_mean)]))+'-'*82 + '\n \n Execution ended.\n'

      print('\n')
      print(logresults)

      f = open(args.logfile, 'w+')
      f.write(logresults)
      f.close()

   else:
      input('No suitable HST observations were found. Please try with different parameters.\nPress enter to exit.\n')
   
   print('TODO:\n-New Photometric bands in EDR3.\n-Fix download of HST images\n-Local Correction using k-neigbors. Figure 3 in https://iopscience.iop.org/article/10.3847/1538-4357/aaa3ec/pdf')

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)

"""
Andres del Pino Molina
"""

