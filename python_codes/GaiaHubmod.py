#!/usr/bin/env python

import sys
import os
import subprocess
import warnings
import re
import shutil

import itertools
import matplotlib.pyplot as plt
from sklearn import mixture
from scipy import stats
from math import log10, floor
import numpy as np
import pandas as pd
pd.options.mode.use_inf_as_na = True

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import (ImageNormalize, ManualInterval)
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astroquery.mast import Observations


def round_significant(x, ex, sig=1):
   """
   This routine returns a quantity rounded to its error significan figures.
   """

   significant = sig-int(floor(log10(abs(ex))))-1

   return round(x, significant), round(ex, significant)


def manual_select_from_cmd(color, mag):
   """
   Select stars based on their membership probabilities and cmd position
   """
   from matplotlib.path import Path
   from matplotlib.widgets import LassoSelector

   class SelectFromCollection(object):
      """
      Select indices from a matplotlib collection using `LassoSelector`.
      """
      def __init__(self, ax, collection, alpha_other=0.15):
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

         lineprops = {'color': 'r', 'linewidth': 1, 'alpha': 0.8}
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

   help =  '----------------------------------------------------------------------------\n'\
           'Please, select likely member stars in the color-magnitude diagram (CMD).\n'\
           '----------------------------------------------------------------------------\n'\
           '- Look in the CMD for any sequence formed by possible member stars.\n'\
           '- Click and drag your cursor to draw a region around these stars.\n'\
           '- On release, the stars contained within the drawn region will be selected.\n'\
           '- Repeat if necessary until you are satisfied with the selection.\n'\
           '- Press enter once finished and follow the instructions in the terminal.\n'\
           '----------------------------------------------------------------------------'

   print('\n'+help+'\n')

   subplot_kw = dict(autoscale_on = False)
   fig, ax = plt.subplots(subplot_kw = subplot_kw)

   pts = ax.scatter(color, mag, c = '0.2', s=1)

   try:
      ax.set_xlim(np.nanmin(color)-0.1, np.nanmax(color)+0.1)
      ax.set_ylim(np.nanmax(mag)+0.05, np.nanmin(mag)-0.05)
   except:
      pass

   ax.grid()
   ax.set_xlabel(color.name.replace("_wmean","").replace("_mean",""))
   ax.set_ylabel(mag.name.replace("_wmean","").replace("_mean",""))

   selector = SelectFromCollection(ax, pts)

   def accept(event):
      if event.key == "enter":
         selector.disconnect()
         plt.close('all')

   fig.canvas.mpl_connect("key_press_event", accept)
   fig.suptitle("Click and move your cursor to select member stars. Press enter to accept.", fontsize=11)
   
   plt.tight_layout()
   plt.show()

   input("Please, press enter to continue.")

   try:
      return selector.selection
   except:
      return [True]*len(mag)



def manual_select_from_pm(pmra, pmdec):
   """
   Select stars based on their membership probabilities and VPD position
   """
   from matplotlib.path import Path
   from matplotlib.widgets import LassoSelector

   class SelectFromCollection(object):
      """
      Select indices from a matplotlib collection using `LassoSelector`.
      """
      def __init__(self, ax, collection, alpha_other=0.15):
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

         lineprops = {'color': 'r', 'linewidth': 1, 'alpha': 0.8}
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

   help =  '----------------------------------------------------------------------------\n'\
           'Please, select likely member stars in the vector-point diagram (VPD).\n'\
           '----------------------------------------------------------------------------\n'\
           '- Look in the VPD for any clump formed by possible member stars.\n'\
           '- Click and drag your cursor to draw a region around these stars.\n'\
           '- On release, the stars contained within the drawn region will be selected.\n'\
           '- Repeat if necessary until you are satisfied with the selection.\n'\
           '- Press enter once finished and follow the instructions in the terminal.\n'\
           '----------------------------------------------------------------------------'

   print('\n'+help+'\n')

   subplot_kw = dict(autoscale_on = False)
   fig, ax = plt.subplots(subplot_kw = subplot_kw)

   pts = ax.scatter(pmra, pmdec, c = '0.2', s=1)

   try:
      margin = 2*(np.nanstd(pmra)+np.nanstd(pmdec))/2
      ax.set_xlim(np.nanmedian(pmra)-margin, np.nanmedian(pmra)+margin)
      ax.set_ylim(np.nanmedian(pmdec)-margin, np.nanmedian(pmdec)+margin)
   except:
      pass

   ax.grid()
   ax.set_xlabel(r'$\mu_{\alpha\star}$')
   ax.set_ylabel(r'$\mu_{\delta}$')

   selector = SelectFromCollection(ax, pts)

   def accept(event):
      if event.key == "enter":
         selector.disconnect()
         plt.close('all')

   fig.canvas.mpl_connect("key_press_event", accept)
   fig.suptitle("Click and move your cursor to select member stars. Press enter to accept.", fontsize=11)
   
   plt.tight_layout()
   plt.show()

   input("Please, press enter to continue.")

   try:
      return selector.selection
   except:
      return [True]*len(mag)


def correct_flux_excess_factor(bp_rp, phot_bp_rp_excess_factor):
   """
   Calculate the corrected flux excess factor for the input Gaia EDR3 data.
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
   
   return labels_astrometric & labels_photometric


def remove_jobs():
   """
   This routine removes jobs from the Gaia archive server.
   """

   list_jobs = []
   for job in Gaia.list_async_jobs():
      list_jobs.append(job.get_jobid())
   
   Gaia.remove_jobs(list_jobs)


def gaia_query(Gaia, query, min_gmag, max_gmag, save_individual_queries, load_existing, name, n, n_total):
   """
   This routine launch the query to the Gaia archive.
   """

   query = query + " AND (phot_g_mean_mag > %.4f) AND (phot_g_mean_mag <= %.4f)"%(min_gmag, max_gmag)

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


def columns_n_conditions(source_table, astrometric_cols, photometric_cols, quality_cols, ra, dec, width = 1.0, height = 1.0):

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

   conditions = "CONTAINS(POINT('ICRS',"+source_table+".ra,"+source_table+".dec),BOX('ICRS',%.8f,%.8f,%.8f,%.8f))=1"%(ra, dec, width, height)

   columns = (", " + astrometric_cols if len(astrometric_cols) > 1 else '') + (", " + photometric_cols if len(photometric_cols) > 1 else '') +  (", " + quality_cols if len(quality_cols) > 1 else '')

   query = "SELECT source_id " + columns + " FROM " + source_table + " WHERE " + conditions

   return query, quality_cols
   

def use_processors(n_processes):

   """
   This routine finds the number of available processors in your machine
   """

   from multiprocessing import cpu_count
   
   available_processors = cpu_count()

   n_processes = n_processes % (available_processors+1)

   if n_processes == 0:
      n_processes = 1
      print('WARNING: Found n_processes = 0. Falling back to default single-threaded execution (n_processes = 1).')

   return n_processes


def incremental_query(query, area, min_gmag = 10.0, max_gmag = 19.5, n_processes = 1, save_individual_queries = False, load_existing = False, name = 'output'):

   """
   This routine search the Gaia archive and downloads the stars using parallel workers.
   """

   print("\n---------------------")
   print("Downloading Gaia data")
   print("---------------------")

   from astroquery.gaia import Gaia
   from multiprocessing import Pool, cpu_count

   mag_nodes = get_mag_bins(min_gmag, max_gmag, area)
   n_total = len(mag_nodes)
   
   if (n_total > 1) and (n_processes != 1):

      print("Executing %s jobs."%(n_total-1))

      pool = Pool(int(np.min((n_total, 20, n_processes*2))))

      args = []
      for n, node in enumerate(range(n_total-1)):
         args.append((Gaia, query, mag_nodes[n+1], mag_nodes[n], save_individual_queries, load_existing, name, n, n_total))

      tables_gaia_queries = pool.map(gaia_multi_query_run, args)

      tables_gaia = [results[0] for results in tables_gaia_queries]
      queries = [results[1] for results in tables_gaia_queries]

      result_gaia = pd.concat(tables_gaia)

      pool.close()

   else:
      result_gaia, queries = gaia_query(Gaia, query, min_gmag, max_gmag, save_individual_queries, load_existing, name, 1, 1)

   return result_gaia, queries


def plot_fields(Gaia_table, obs_table, HST_path, use_only_good_gaia = False, min_stars_alignment = 5, no_plots = False, name = 'test.png'):
   """
   This routine plots the fields and select Gaia stars within them.
   """

   from matplotlib.patheffects import withStroke
   from matplotlib.patches import (Polygon, Patch)
   from matplotlib.collections import PatchCollection
   from matplotlib.lines import Line2D
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

   def coolwarm(filter, alpha = 1):
      color = [rgb for rgb in plt.cm.coolwarm(int(255 * (float(filter) - 555) / (850-555)))]
      color[-1] = alpha
      return color

   # If no_plots == False, then python opens a plotting device
   if no_plots == False:
      fig, ax = plt.subplots(1,1, figsize = (5., 4.75))

   if use_only_good_gaia:
      Gaia_table_count = Gaia_table[Gaia_table.clean_label == True]
   else:
      Gaia_table_count = Gaia_table

   patches = []
   fields_data = []
   bad_patches = []
   bad_fields_data = []
   
   # We rearrange obs_table for legibility
   obs_table = obs_table.sort_values(by =['s_ra', 's_dec', 'proposal_id', 'obsid'], ascending = False).reset_index(drop=True)

   for index_obs, (s_ra, s_dec, footprint_str, obsid, filter, t_bl, obs_id) in obs_table.loc[:, ['s_ra', 's_dec', 's_region', 'obsid', 'filters', 't_baseline', 'obs_id']].iterrows():
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
            
            # Check if the set seems downloaded
            if os.path.isfile(HST_path+'mastDownload/HST/'+obs_id+'/'+obs_id+'_drz.fits'):
               fc = [0,1,0,0.2]
            else:
               fc = [1,1,1,0.2]

            footprint =  shap_polygon(tuples_list)

            star_counts = 0
            for idx, ra, dec in zip(Gaia_table_count.index, Gaia_table_count.ra, Gaia_table_count.dec):
               if Point(ra, dec).within(footprint):
                  idx_Gaia_in_field.append(idx)
                  star_counts += 1

            gaia_stars_per_poly.append(star_counts)
            
            if star_counts >= min_stars_alignment:
               patches.append(polygon)
               fields_data.append([index_obs, round(s_ra, 2), round(s_dec, 2), filter, coolwarm(float(filter.replace(r'F', '').replace(r'W', '').replace('LP', ''))), t_bl, fc, sum(gaia_stars_per_poly)])
            else:
               bad_patches.append(polygon)
               bad_fields_data.append([round(s_ra, 2), round(s_dec, 2), filter, coolwarm(float(filter.replace(r'F', '').replace(r'W', '').replace('LP', ''))), fc])
   print('\n')

   fields_data = pd.DataFrame(data = fields_data, columns=['Index_obs', 'ra', 'dec', 'filter', 'filter_color', 't_baseline', 'download_color', 'gaia_stars_per_obs'])
   bad_fields_data = pd.DataFrame(data = bad_fields_data, columns=['ra', 'dec', 'filter', 'filter_color', 'download_color'])

   # Select only the observations with enough Gaia stars
   try:
      obs_table = obs_table.iloc[fields_data.Index_obs, :].reset_index(drop=True)
      obs_table['gaia_stars_per_obs'] = fields_data.gaia_stars_per_obs
   except:
      pass

   if no_plots == False:

      try:
         ra_lims = [min(Gaia_table.ra.max(), fields_data.ra.max()+0.2/np.cos(np.deg2rad(fields_data.dec.mean()))), max(Gaia_table.ra.min(), fields_data.ra.min()-0.2/np.cos(np.deg2rad(fields_data.dec.mean())))]
         dec_lims = [max(Gaia_table.dec.min(), fields_data.dec.min()-0.2), min(Gaia_table.dec.max(), fields_data.dec.max()+0.2)]
      except:
         ra_lims = [Gaia_table.ra.max(), Gaia_table.ra.min()]
         dec_lims = [Gaia_table.dec.min(), Gaia_table.dec.max()]

      bpe = PatchCollection(bad_patches, alpha = 0.1, ec = 'None', fc = bad_fields_data.download_color, antialiased = True, lw = 1, zorder = 2)
      bpf = PatchCollection(bad_patches, alpha = 1, ec = bad_fields_data.filter_color, fc = 'None', antialiased = True, lw = 1, zorder = 3, hatch='/////')

      ax.add_collection(bpe)
      ax.add_collection(bpf)

      pe = PatchCollection(patches, alpha = 0.1, ec = 'None', fc = fields_data.download_color, antialiased = True, lw = 1, zorder = 2)
      pf = PatchCollection(patches, alpha = 1, ec = fields_data.filter_color, fc = 'None', antialiased = True, lw = 1, zorder = 3)

      ax.add_collection(pe)
      ax.add_collection(pf)

      ax.plot(Gaia_table.ra[~Gaia_table.clean_label], Gaia_table.dec[~Gaia_table.clean_label], '.', color = '0.6', ms = 0.75, zorder = 0)
      ax.plot(Gaia_table.ra[Gaia_table.clean_label], Gaia_table.dec[Gaia_table.clean_label], '.', color = '0.2', ms = 0.75, zorder = 1)

      for coo, obs_id in fields_data.groupby(['ra','dec']).apply(lambda x: x.index.tolist()).iteritems():
         if len(obs_id) > 1:
            obs_id = '%i-%i'%(min(obs_id)+1, max(obs_id)+1)
         else:
            obs_id = obs_id[0]+1
         ax.annotate(obs_id, xy=(coo[0], coo[1]), xycoords='data', color = 'k', zorder = 3)
      
      for ii, (t_bl, obs_id) in enumerate(fields_data.groupby(['t_baseline']).apply(lambda x: x.index.tolist()).iteritems()):
         if len(obs_id) > 1:
            obs_id = '%i-%i, %.2f years'%(min(obs_id)+1, max(obs_id)+1, t_bl)
         else:
            obs_id = '%i, %.2f years'%(obs_id[0]+1, t_bl)
         t = ax.annotate(obs_id, xy=(0.05, 0.95-0.05*ii), xycoords='axes fraction', fontsize = 9, color = 'k', zorder = 3)
         t.set_path_effects([withStroke(foreground="w", linewidth=3)])

      ax.set_xlim(ra_lims[0], ra_lims[1])
      ax.set_ylim(dec_lims[0], dec_lims[1])
      ax.grid()

      ax.set_xlabel(r'RA [$^\circ$]')
      ax.set_ylabel(r'Dec [$^\circ$]')

      legend_elements = [Line2D([0], [0], marker='.', color='None', markeredgecolor='0.2', markerfacecolor='0.2', label = 'Good stars'), Line2D([0], [0], marker='.', color='None', markeredgecolor='0.6', markerfacecolor='0.6', label = 'Bad stars')]

      if [0,1,0,0.2] in fields_data.download_color.tolist():
         legend_elements.extend([Patch(facecolor=[0,1,0,0.2], edgecolor='0.4',
                                 label='Previously downloaded'),
                           Patch(facecolor=[1,1,1,0.2], edgecolor='0.4',
                                 label='Not yet downloaded')])
      if len(bad_patches) > 0:
         legend_elements.append(Patch(facecolor=[1,1,1,0.2], edgecolor='0.4', hatch='/////',
                                 label='Not enough good stars'))

      for filter, filter_color in fields_data.groupby(['filter'])['filter_color'].first().iteritems():
         legend_elements.append(Patch(facecolor='1', edgecolor=filter_color, label=filter))

      ax.legend(handles=legend_elements, prop={'size': 7})

      plt.tight_layout()

      plt.savefig(name, bbox_inches='tight')

      with plt.rc_context(rc={'interactive': False}):
         plt.gcf().show()

   obs_table['field_id'] = ['(%i)'%(ii+1) for ii in np.arange(len(obs_table))]

   return obs_table


def search_mast(ra, dec, search_width = 0.25, search_height = 0.25, filters = ['any'], project = ['HST'], t_exptime_min = 50, t_exptime_max = 2500, date_second_epoch = 57531.0, time_baseline = 3650):
   """
   This routine search for HST observations in MAST at a given position.
   """

   ra1 = ra - search_width / 2 + 0.056 / np.cos(np.deg2rad(dec))
   ra2 = ra + search_width / 2 - 0.056 / np.cos(np.deg2rad(dec))
   dec1 = dec - search_height / 2 + 0.056
   dec2 = dec + search_height / 2 - 0.056

   t_max = date_second_epoch - time_baseline
   
   if type(filters) is not list:
      filters = [filters]

   obs_table = Observations.query_criteria(dataproduct_type=['image'], obs_collection=['HST'], s_ra=[ra1, ra2], s_dec=[dec1, dec2], instrument_name=['ACS/WFC', 'WFC3/UVIS'], t_max=[0, t_max], filters = filters, project = project)

   data_products_by_obs = search_data_products_by_obs(obs_table)
   
   #Pandas is easier:
   obs_table = obs_table.to_pandas()
   data_products_by_obs = data_products_by_obs.to_pandas()

   # We are only interested in FLC and DRZ images
   data_products_by_obs = data_products_by_obs.loc[data_products_by_obs.project != 'HAP', :]
   obs_table = obs_table.merge(data_products_by_obs.loc[data_products_by_obs.productSubGroupDescription == 'FLC', :].groupby(['parent_obsid'])['parent_obsid'].count().rename_axis('obsid').rename('n_exp'), on = ['obsid'])

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

   return obs_table.astype({'obsid': 'int64'}).reset_index(drop = True), data_products_by_obs.astype({'parent_obsid': 'int64'}).reset_index(drop = True)


def search_data_products_by_obs(obs_table):
   """
   This routine search for images in MAST related to the given observations table.
   """

   data_products_by_obs = Observations.get_product_list(obs_table)

   return data_products_by_obs[((data_products_by_obs['productSubGroupDescription'] == 'FLC') | (data_products_by_obs['productSubGroupDescription'] == 'DRZ')) & (data_products_by_obs['obs_collection'] == 'HST')]


def download_HST_images(data_products_by_obs, path = './'):
   """
   This routine downloads the selected HST images from MAST.
   """

   try:
      images = Observations.download_products(Table.from_pandas(data_products_by_obs), download_dir=path)
   except:
      images = Observations.download_products(data_products_by_obs, download_dir=path)

   return images


def create_dir(path):
   """
   This routine creates directories.
   """

   if not os.path.isdir(path):
      try:
         tree = path.split('/')
         previous_tree = tree[0]
         for leave in tree[1:]:
            previous_tree = '%s/%s'%(previous_tree,leave)
            try:
               os.mkdir(previous_tree)
            except:
               pass
      except OSError:  
         print ("Creation of the directory %s failed" % path)
      else:  
         print ("Successfully created the directory %s " % path)


def members_prob(table, clf, vars, errvars, clipping_prob = 3, data_0 = None):
   """
   This routine will find probable members through scoring of a passed model (clf).
   """

   has_vars = table.loc[:, vars].notnull().all(axis = 1)

   data = table.loc[has_vars, vars]
   err = table.loc[has_vars, errvars]

   clustering_data = table.loc[has_vars, 'clustering_data'] == 1

   results = pd.DataFrame(columns = ['logprob', 'member_logprob', 'member_zca'], index = table.index)

   for var in vars:
      results.loc[:, 'w_%s'%var] = np.nan

   if (clustering_data.sum() > 1):

      if data_0 is None:
         data_0 = data.loc[clustering_data, vars].median().values

      data -= data_0

      clf.fit(data.loc[clustering_data, :])
      
      logprob = clf.score_samples(data)
      label_logprob = logprob >= np.nanmedian(logprob[clustering_data])-clipping_prob*np.nanstd(logprob[clustering_data])

      label_wzca = []
      for mean, covariances in zip(clf.means_, clf.covariances_):
         data_c = data-mean

         if clf.covariance_type == 'full':
            cov = covariances
         elif clf.covariance_type == 'diag':
            cov = np.array([[covariances[0],0], [0, covariances[1]]])
         else:
            cov = np.array([[covariances,0], [0, covariances]])

         eigVals, eigVecs = np.linalg.eig(cov)

         diagw = np.diag(1/((eigVals+.1e-6)**0.5)).real.round(5)
         Wzca = np.dot(np.dot(eigVecs, diagw), eigVecs.T)

         wdata = np.dot(data_c, Wzca)
         werr = np.dot(err, Wzca)

         label_wzca.append(((wdata**2).sum(axis =1) <= clipping_prob**2) & ((werr**2).sum(axis =1) <= clipping_prob**2))

      if len(label_wzca) > 1:
         label_wzca = list(map(all, zip(*label_wzca)))
      else:
         label_wzca = label_wzca[0]

      results.loc[has_vars, 'member_logprob'] = label_logprob
      results.loc[has_vars, 'member_zca'] = label_wzca
      results.loc[has_vars, 'logprob'] = logprob

      for var, wdata_col in zip(vars, wdata.T):
         results.loc[has_vars, 'w_%s'%var] = wdata_col
    
   return results


def pm_cleaning_GMM_recursive(table, vars, errvars, alt_table = None, data_0 = None, n_components = 1, covariance_type = 'full', clipping_prob = 3, no_plots = True, verbose = True, plot_name = ''):
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
   
   if verbose:
      print('')
      print('Finding member stars...')

   convergence = False
   iteration = 0
   while not convergence:
      if verbose & (iteration > 0):
         print("\rIteration %i, %i objects remain."%(iteration, table.clustering_data.sum()))

      clust = table.loc[:, vars+errvars+['clustering_data']]

      if iteration > 3:
         data_0 = None

      fitting = members_prob(clust, clf, vars, errvars, clipping_prob = clipping_prob,  data_0 = data_0)
      
      # If the ZCA detects too few members, we use the logprob.
      if fitting.member_zca.sum() < 10:
         print('WARNING: Not enough members after ZTA whitening. Switching to selection based on logarithmic probability.')
         table['member'] = fitting.member_logprob
      else:
         table['member'] = fitting.member_zca

      table['logprob'] = fitting.logprob
      table['clustering_data'] = (table.clustering_data == 1) & (table.member == 1)  & (table.real_data == 1)

      if (iteration > 999):
         convergence = True
      elif iteration > 0:
         convergence = fitting.equals(previous_fitting)

      previous_fitting = fitting.copy()
      iteration += 1
   
   if no_plots == False:
      plt.close('all')
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10.5, 4.75), dpi=200)
      ax1.plot(table.loc[table.real_data == 1 ,vars[0]], table.loc[table.real_data == 1 , vars[1]], 'k.', ms = 0.5, zorder = 0)
      ax1.scatter(table.loc[table.clustering_data == 1 ,vars[0]], table.loc[table.clustering_data == 1 ,vars[1]], c = fitting.loc[table.clustering_data == 1, 'logprob'], s = 1, zorder = 1)
      ax1.set_xlabel(r'$\mu_{\alpha*}$')
      ax1.set_ylabel(r'$\mu_{\delta}$')
      ax1.grid()

      t = np.linspace(0, 2*np.pi, 100)
      xx = clipping_prob*np.sin(t)
      yy = clipping_prob*np.cos(t)

      ax2.plot(fitting.loc[table.real_data == 1 , 'w_%s'%vars[0]], fitting.loc[table.real_data == 1 , 'w_%s'%vars[1]], 'k.', ms = 0.5, zorder = 0)
      ax2.scatter(fitting.loc[table.clustering_data == 1 , 'w_%s'%vars[0]], fitting.loc[table.clustering_data == 1 , 'w_%s'%vars[1]], c = fitting.loc[table.clustering_data == 1, 'logprob'].values, s = 1, zorder = 1)
      ax2.plot(xx, yy, 'r-', linewidth = 1)

      ax2.set_xlabel(r'$\sigma(\mu_{\alpha*})$')
      ax2.set_ylabel(r'$\sigma(\mu_{\delta})$')
      ax2.grid()

      try:
         margin = 2*(np.nanstd(table.loc[table.real_data == 1 ,vars[0]])+np.nanstd(table.loc[table.real_data == 1 ,vars[1]]))/2
         ax1.set_xlim(np.nanmedian(table.loc[table.real_data == 1 ,vars[0]])-margin, np.nanmedian(table.loc[table.real_data == 1 ,vars[0]])+margin)
         ax1.set_ylim(np.nanmedian(table.loc[table.real_data == 1 ,vars[1]])-margin, np.nanmedian(table.loc[table.real_data == 1 ,vars[1]])+margin)

         ax2.set_xlim(-2*clipping_prob, 2*clipping_prob)
         ax2.set_ylim(-2*clipping_prob, 2*clipping_prob)

      except:
         pass

      plt.subplots_adjust(wspace=0.3, hspace=0.1)

      plt.savefig(plot_name, bbox_inches='tight')

   if verbose:
      print('')

   if alt_table is not None:
      return table.loc[table.real_data == 1, 'clustering_data'], fitting.loc[table.real_data == 0, 'clustering_data']
   else:
      return table.clustering_data


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


def hst1pass(HST_path, exec_path, obs_id, HST_image, force_fmin, force_hst1pass, verbose):
   """
   This routine will execute hst1pass Fortran routine using the correct arguments.
   """

   #Define HST servicing missions times
   t_sm3 = Time('2002-03-12').mjd
   t_sm4 = Time('2009-05-24').mjd

   output_dir = HST_path+'mastDownload/HST/'+obs_id+'/'
   HST_image_filename = output_dir+HST_image
   XYmqxyrd_filename = HST_image_filename.split('.fits')[0]+'.XYmqxyrd'

   if force_hst1pass or (force_fmin is not None):
      remove_file(XYmqxyrd_filename)

   if not os.path.isfile(XYmqxyrd_filename):
      if verbose:
         print('Finding sources in', HST_image)
      
      #Read information from the header of the image
      hdul = fits.open(HST_image_filename)
      instrument = hdul[0].header['INSTRUME']
      detector = hdul[0].header['DETECTOR']
      if detector == 'UVIS':
         detector = 'UV'

      try:
         filter = [filter for filter in [hdul[0].header['FILTER1'], hdul[0].header['FILTER2']] if 'CLEAR' not in filter][0]
      except:
         filter = hdul[0].header['FILTER']

      # No standard PSF library for F555W filter. We use the F606W.
      if 'F555W' in filter:
         filter_psf = filter.replace('F555W', 'F606W')
      else:
         filter_psf = filter

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
         print('%s exptime = %.1f. Using fmin = %i'%(HST_image, hdul[0].header['EXPTIME'], fmin))

      psf_filename = '%s/lib/STDPSFs/%s%s/PSFSTD_%s%s_%s%s.fits'%(exec_path, instrument, detector, instrument, detector, filter_psf, sm)
      gdc_filename = '%s/lib/STDGDCs/%s%s/STDGDC_%s%s_%s.fits'%(exec_path, instrument, detector, instrument, detector, filter)

      if os.path.isfile(psf_filename) and os.path.isfile(gdc_filename):
         pert_grid = 5
         while not applied_pert(XYmqxyrd_filename) and (pert_grid > 0):
            bashCommand = "%s/fortran_codes/hst1pass.e HMIN=5 FMIN=%s PMAX=999999 GDC=%s PSF=%s PERT%i=AUTO OUT=XYmqxyrd OUTDIR=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, pert_grid, output_dir, HST_image_filename)

            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            output, error = process.communicate()
            pert_grid -= 1

         if verbose:
            print('%s PERT PSF = %sx%s'%(HST_image, pert_grid, pert_grid))

         remove_file(HST_image.replace('_flc','_psf'))

      else:
         if not os.path.isfile(psf_filename):
            print('WARNING: %s file not found!'%os.path.basename(psf_filename))
         if not os.path.isfile(gdc_filename):
            print('WARNING: %s file not found!'%os.path.basename(gdc_filename))
         print('Skipping %s.'%HST_image)


def launch_hst1pass(flc_images, HST_obs_to_use, HST_path, exec_path, force_fmin = None, force_hst1pass = True, verbose = True, n_processes = 1):
   """
   This routine will launch hst1pass routine in parallel or serial 
   """
   from multiprocessing import Pool, cpu_count
   
   print("\n---------------------------------")
   print("Finding sources in the HST images")
   print("---------------------------------")

   args = []
   for HST_image_obsid in [HST_obs_to_use] if not isinstance(HST_obs_to_use, list) else HST_obs_to_use:
      for index_image, (obs_id, HST_image) in flc_images.loc[flc_images['parent_obsid'] == HST_image_obsid, ['obs_id', 'productFilename']].iterrows():
         args.append((HST_path, exec_path, obs_id, HST_image, force_fmin, force_hst1pass, verbose))
   
   if (len(args) > 1) and (n_processes != 1):
      pool = Pool(min(n_processes, len(args)))
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
   
   valid = [p >= alpha,  (mat[:, 6:8].mean(axis = 0) <= center_tolerance).all(), len(mat) >= min_stars_alignment]

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
      ax1.set_xlabel(r'$X_{Gaia} - X_{HST}$ [pixels]')
      ax1.set_ylabel(r'$Y_{Gaia} - Y_{HST}$ [pixels]')
      ax1.grid()
      
      add_inner_title(ax1, 'Valid=%s\np=%.4f\ncen=(%.4f,%.4f)\nnum=%i'%(all(valid), p, mat[:, 6].mean(), mat[:, 7].mean(), len(mat)), 1)
      
      plt.savefig(mat_filename.split('.MAT')[0]+'_MAT_%i.png'%iteration, bbox_inches='tight')
      plt.close()

   return valid


def xym2pm_Gaia(iteration, Gaia_HST_table_field, Gaia_HST_table_filename, HST_image_filename, lnk_filename, mat_filename, amp_filename, exec_path, date_reference_second_epoch, only_use_members, rewind_stars, force_pixel_scale, force_max_separation, force_use_sat, fix_mat, force_wcs_search_radius, min_stars_alignment, verbose, previous_xym2pm, mat_plots, no_amplifier_based, min_stars_amp, use_mean):
   """
   This routine will execute xym2pm_Gaia Fortran routine using the correct arguments.
   """

   hdul = fits.open(HST_image_filename)
   t_max = hdul[0].header['EXPEND']
   ra_cent = hdul[0].header['RA_TARG']
   dec_cent = hdul[0].header['DEC_TARG']
   exptime = hdul[0].header['EXPTIME']

   try:
      filter = [filter for filter in [hdul[0].header['FILTER1'], hdul[0].header['FILTER2']] if 'CLEAR' not in filter][0]
   except:
      filter = hdul[0].header['FILTER']

   t_baseline = (date_reference_second_epoch.mjd - t_max) / 365.25
   timeref = str(date_reference_second_epoch.jyear)

   if force_pixel_scale is None:
      # Infer pixel scale from the header
      pixel_scale = round(np.mean([proj_plane_pixel_scales(WCS(hdul[3].header)).mean(), proj_plane_pixel_scales(WCS(hdul[5].header)).mean()])*3600, 3)
   else:
      pixel_scale = force_pixel_scale

   pixel_scale_mas = 1e3 * pixel_scale

   if (iteration > 0) & (rewind_stars):
      align_var = ['ra', 'ra_error', 'dec', 'dec_error', 'hst_gaia_pmra_%s'%use_mean, 'hst_gaia_pmra_%s_error'%use_mean, 'hst_gaia_pmdec_%s'%use_mean, 'hst_gaia_pmdec_%s_error'%use_mean, 'gmag', 'use_for_alignment']
   else:
      align_var = ['ra', 'ra_error', 'dec', 'dec_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'gmag', 'use_for_alignment']

   if not os.path.isfile(lnk_filename) or not previous_xym2pm:

      f = open(Gaia_HST_table_filename, 'w+')
      f.write('# ')
      Gaia_HST_table_field.loc[:, align_var].astype({'use_for_alignment': 'int32'}).to_csv(f, index = False, sep = ' ', na_rep = 0)
      f.close()

      # Here it goes the executable line. Input values can be fine-tuned here
      if (iteration > 0) & (rewind_stars):
         time = ' TIME=%s'%str(round(Time(t_max, format='mjd').jyear, 3))
      else:
         time = ''

      if force_use_sat:
         use_sat = ' USESAT+'
      else:
         use_sat = ''

      if (fix_mat) and (iteration > 0) and not rewind_stars:
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

      if not no_amplifier_based:
         use_amp = ' NAMP=%i AMP+'%min_stars_amp
      else:
         use_amp = ''

      bashCommand = "%s/fortran_codes/xym2pm_Gaia.e %s %s RACEN=%f DECEN=%f XCEN=5000.0 YCEN=5000.0 PSCL=%s SIZE=10000 DISP=%.1f NMIN=%i TIMEREF=%s%s%s%s%s%s"%(exec_path, Gaia_HST_table_filename, HST_image_filename, ra_cent, dec_cent, pixel_scale, max_separation, min_stars_alignment, timeref, time, use_sat, use_mat, use_brute, use_amp)

      process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      output, error = process.communicate()

      f = open(lnk_filename.replace(".LNK", "_6p_transformation.txt"), 'w+')
      f.write(output.decode("utf-8"))
      f.close()

   try:
      #Next are threshold rejection values for the MAT files. The alpha is the Shapiro–Wilk statistics. Lower values are more permissive.
      if (iteration > 0) and only_use_members and not rewind_stars:
         alpha = 1e-64
      else:
         alpha = 0

      valid_mat = check_mat(mat_filename, iteration, min_stars_alignment, alpha = alpha, center_tolerance = 1e-2, fix_mat = fix_mat, clipping_prob = 3., plots = mat_plots, verbose = verbose)

      if all(valid_mat):

         f = open(lnk_filename, 'r')
         header = [w.replace('m_hst', filter) for w in f.readline().rstrip().strip("# ").split(' ')]
         lnk = pd.read_csv(f, names=header, sep = '\s+', comment='#', na_values = 0.0).set_index(Gaia_HST_table_field.index)
         f.close()
         
         if os.path.isfile(amp_filename) and not no_amplifier_based:
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
         eradec_hst = lnk.q_hst.copy()
         # Assign the maximum (worst) QFIT parameter to saturated stars.
         eradec_hst[lnk.xhst_gaia.notnull()] = lnk[lnk.xhst_gaia.notnull()].q_hst.replace({np.nan:lnk.q_hst.mean()+3*lnk.q_hst.std()}).copy()
         # 0.8 seems reasonable, although this may be tuned through an empirical function.
         eradec_hst *= pixel_scale_mas * 0.8

         lnk['relative_hst_gaia_pmra'] = -(lnk.x_gaia - lnk.xhst_gaia) * pixel_scale_mas / t_baseline
         lnk['relative_hst_gaia_pmdec'] = (lnk.y_gaia - lnk.yhst_gaia) * pixel_scale_mas / t_baseline

         # Notice the 1e3. xym2pm_Gaia.e takes the error in mas but returns it in arcsec.
         lnk['relative_hst_gaia_pmra_error'] = eradec_hst / t_baseline
         lnk['relative_hst_gaia_pmdec_error'] = eradec_hst / t_baseline
         lnk['gaia_dra_uncertaintity'] = 1e3 * lnk.era_gaia / t_baseline
         lnk['gaia_ddec_uncertaintity'] = 1e3 * lnk.edec_gaia / t_baseline
         # Here we arbitrarily divided by 100 just for aesthetic reasons.
         lnk['%s_error'%filter] = lnk.q_hst.replace({0:lnk.q_hst.max()}) * 0.01

         match = lnk.loc[:, ['xc_hst', 'yc_hst',  filter, '%s_error'%filter, 'relative_hst_gaia_pmra', 'relative_hst_gaia_pmra_error', 'relative_hst_gaia_pmdec', 'relative_hst_gaia_pmdec_error', 'gaia_dra_uncertaintity', 'gaia_ddec_uncertaintity', 'q_hst']]

         print('-->%s (%s, %ss): matched %i stars.'%(os.path.basename(HST_image_filename), filter, exptime, len(match)))
      
      else:
         print('-->%s (%s, %ss): bad quality match:'%(os.path.basename(HST_image_filename), filter, exptime))
         if verbose:
            if valid_mat[0] != True:
               print('   Non Gaussian distribution found in the transformation')
            if valid_mat[1] != True:
               print('   Not (0,0) average of the distribution')
            if valid_mat[2] != True:
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


def launch_xym2pm_Gaia(Gaia_HST_table, data_products_by_obs, HST_obs_to_use, HST_path, exec_path, date_reference_second_epoch, only_use_members = False, preselect_cmd = False, preselect_pm = False, rewind_stars = True, force_pixel_scale = None, force_max_separation = None, force_use_sat = True, fix_mat = True, no_amplifier_based = False, min_stars_amp = 25, force_wcs_search_radius = None, n_components = 1, clipping_prob = 6, use_only_good_gaia = False, min_stars_alignment = 100, use_mean = 'wmean', no_plots = False, verbose = True, quiet = False, ask_user_stop = False, max_iterations = 10, previous_xym2pm = False, remove_previous_files = True, n_processes = 1, plot_name = ''):

   """
   This routine will launch xym2pm_Gaia Fortran routine in parallel or serial using the correct arguments.
   """
   from multiprocessing import Pool, cpu_count

   n_images = len(data_products_by_obs.loc[data_products_by_obs['parent_obsid'].isin([HST_obs_to_use] if not isinstance(HST_obs_to_use, list) else HST_obs_to_use), :])
   
   if (n_images > 1) and (n_processes != 1):
      pool = Pool(min(n_processes, n_images))
      mat_plots = False
   else:
      mat_plots = ~no_plots

   if use_only_good_gaia:
      Gaia_HST_table['use_for_alignment'] = Gaia_HST_table.clean_label.values
   else:
      Gaia_HST_table['use_for_alignment'] = True

   convergence = False
   iteration = 0

   pmra_evo = []
   pmdec_evo = []

   pmra_diff_evo = []
   pmdec_diff_evo = []

   hst_gaia_pmra_lsqt_evo = []
   hst_gaia_pmdec_lsqt_evo = []

   while not convergence:
      print("\n-----------")
      print("Iteration %i"%(iteration))
      print("-----------")
      
      # Close previous plots
      plt.close('all')

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

            n_field_stars = Gaia_HST_table.loc[Gaia_HST_table['HST_image'].str.contains(str(obs_id)), 'use_for_alignment'].count()
            u_field_stars = Gaia_HST_table.loc[Gaia_HST_table['HST_image'].str.contains(str(obs_id)), 'use_for_alignment'].sum()
            
            # We assume that the some parts of the image can have slightly lower stars density than others.
            # Therefore, we require 5 times the number of stars per amplifier in the entire image instead of 4.
            if (n_field_stars < min_stars_amp*5) and (no_amplifier_based == False):
               print('WARNING: Not enough stars in %s as to separate amplifiers. Only one channel will be used.'%HST_image)
               no_amplifier_based_inuse = True
            elif (n_field_stars >= min_stars_amp*5) and (no_amplifier_based == False):
               no_amplifier_based_inuse = False

            if (u_field_stars < min_stars_alignment):
               print('WARNING: Not enough member stars in %s. Using all the stars in the field.'%HST_image)
               Gaia_HST_table.loc[Gaia_HST_table['HST_image'].str.contains(str(obs_id)), 'use_for_alignment'] = True

         Gaia_HST_table_field = Gaia_HST_table.loc[Gaia_HST_table['HST_image'].str.contains(str(obs_id)), :]

         args.append((iteration, Gaia_HST_table_field, Gaia_HST_table_filename, HST_image_filename, lnk_filename, mat_filename, amp_filename, exec_path, date_reference_second_epoch, only_use_members, rewind_stars, force_pixel_scale, force_max_separation, force_use_sat, fix_mat, force_wcs_search_radius, min_stars_alignment, verbose, previous_xym2pm, mat_plots, no_amplifier_based_inuse, min_stars_amp, use_mean))

      if (len(args) > 1) and (n_processes != 1):
         lnks = pool.map(xym2pm_Gaia_multiproc, args)

      else:
         lnks = []
         for arg in args:
            lnks.append(xym2pm_Gaia_multiproc(arg))

      lnks = pd.concat(lnks, sort=True)

      if len(lnks) == 0:
         print('WARNING: No match could be found for any of the images. Please try with other parameters.\nExiting now.\n')
         remove_file(exec_path)
         sys.exit(1)
      else:
         print("-----------")
         print('%i stars were used in the transformation.'%(min([Gaia_HST_table.use_for_alignment.sum(), n_field_stars])))

      lnks_averaged = lnks.groupby(lnks.index).apply(weighted_avg_err)

      # Gaia positional errors have to be added in quadrature
      lnks_averaged['relative_hst_gaia_pmra_mean_error'] = np.sqrt(lnks_averaged['relative_hst_gaia_pmra_mean_error']**2 + lnks_averaged['gaia_dra_uncertaintity_mean']**2)
      lnks_averaged['relative_hst_gaia_pmdec_mean_error'] = np.sqrt(lnks_averaged['relative_hst_gaia_pmdec_mean_error']**2 + lnks_averaged['gaia_ddec_uncertaintity_mean']**2)
      lnks_averaged['relative_hst_gaia_pmra_wmean_error'] = np.sqrt(lnks_averaged['relative_hst_gaia_pmra_wmean_error']**2 + lnks_averaged['gaia_dra_uncertaintity_mean']**2)
      lnks_averaged['relative_hst_gaia_pmdec_wmean_error'] = np.sqrt(lnks_averaged['relative_hst_gaia_pmdec_wmean_error']**2 + lnks_averaged['gaia_ddec_uncertaintity_mean']**2)

      # Remove redundant columns
      lnks_averaged = lnks_averaged.drop(columns=[col for col in lnks_averaged if col.startswith('gaia') or (col.startswith('q_hst') and 'wmean' in col)])

      try:
         Gaia_HST_table.drop(columns = lnks_averaged.columns, inplace = True)
      except:
         pass

      # Obtain absolute PMs
      Gaia_HST_table = absolute_pm(Gaia_HST_table.join(lnks_averaged))

      # Membership selection
      if only_use_members:
         if (preselect_cmd == True) and (no_plots == False) and (quiet == False):
            if (iteration == 0):
               # Select stars in theCMD
               hst_filters = [col for col in lnks_averaged.columns if ('F' in col) & ('error' not in col) & ('std' not in col) & ('_mean' not in col)]
               hst_filters.sort()

               min_HST_HST_stars = 0.9*len(Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, ['relative_hst_gaia_pmra_%s'%use_mean, 'relative_hst_gaia_pmdec_%s'%use_mean]].dropna())
               
               cmd_sel = False
               if len(hst_filters) >= 2:
                  for f1, f2 in itertools.combinations(hst_filters, 2):
                     if (len((Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, f1] - Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, f2]).dropna()) >=  min_HST_HST_stars):
                        Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, 'clustering_cmd'] =  manual_select_from_cmd((Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment,hst_filters[0]]-Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, hst_filters[1]]).rename('%s - %s'%(hst_filters[0], hst_filters[1])), Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, hst_filters[1]])
                        cmd_sel = True
                        break

               if (len(hst_filters) == 1) or not cmd_sel:
                  for cmd_filter in hst_filters:
                     if int(re.findall(r'\d+', cmd_filter)[0]) <= 606:
                        HST_Gaia_color = (Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, cmd_filter]-Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, 'gmag']).rename('%s - Gmag'%cmd_filter)
                     else:
                        HST_Gaia_color = (Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, 'gmag']-Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, cmd_filter]).rename('Gmag - %s'%cmd_filter)

                     Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, '%s_clustering_data_cmd'%cmd_filter] =  manual_select_from_cmd(HST_Gaia_color, Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, 'gmag'].rename('Gmag'))

                  cmd_clustering_filters = [col for col in Gaia_HST_table.columns if '_clustering_data_cmd' in col]
                  Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, 'clustering_cmd'] = (Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, cmd_clustering_filters] == True).any(axis = 1)
                  Gaia_HST_table.drop(columns = cmd_clustering_filters, inplace = True)
         else:
            Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, 'clustering_cmd'] = True


         if (preselect_pm == True) and (no_plots == False) and (quiet == False):
            if (iteration == 0):
               Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, 'clustering_pm'] =  manual_select_from_pm(Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, 'relative_hst_gaia_pmra_%s'%use_mean], Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, 'relative_hst_gaia_pmdec_%s'%use_mean])
               gauss_center = Gaia_HST_table.loc[Gaia_HST_table.clustering_pm == True, ['relative_hst_gaia_pmra_%s'%use_mean, 'relative_hst_gaia_pmdec_%s'%use_mean]].mean().values

         else:
            Gaia_HST_table.loc[Gaia_HST_table.use_for_alignment, 'clustering_pm'] = True
            gauss_center = [0,0]

         Gaia_HST_table['clustering_data'] = Gaia_HST_table.clustering_cmd & Gaia_HST_table.clustering_pm

         # Select stars in the PM space asuming spherical covariance (Reasonable for dSphs and globular clusters) 
         pm_clustering = pm_cleaning_GMM_recursive(Gaia_HST_table.copy(), ['relative_hst_gaia_pmra_%s'%use_mean, 'relative_hst_gaia_pmdec_%s'%use_mean], ['relative_hst_gaia_pmra_%s_error'%use_mean, 'relative_hst_gaia_pmdec_%s_error'%use_mean], data_0 = gauss_center, n_components = n_components, covariance_type = 'full', clipping_prob = clipping_prob, verbose = verbose,  no_plots = no_plots, plot_name = '%s_%i.png'%(plot_name, iteration))

         Gaia_HST_table['use_for_alignment'] = pm_clustering & Gaia_HST_table.clustering_data

      elif not rewind_stars:
         convergence = True
      
      # Useful statistics:
      id_pms = np.isfinite(Gaia_HST_table['hst_gaia_pmra_%s'%use_mean]) & np.isfinite(Gaia_HST_table['hst_gaia_pmdec_%s'%use_mean])

      pmra_evo.append(Gaia_HST_table.loc[id_pms, 'hst_gaia_pmra_%s'%use_mean])
      pmdec_evo.append(Gaia_HST_table.loc[id_pms, 'hst_gaia_pmdec_%s'%use_mean])

      hst_gaia_pmra_lsqt_evo.append(np.nanstd( (Gaia_HST_table.loc[id_pms, 'hst_gaia_pmra_%s'%use_mean]  - Gaia_HST_table.loc[id_pms, 'pmra'])))
      hst_gaia_pmdec_lsqt_evo.append(np.nanstd( (Gaia_HST_table.loc[id_pms, 'hst_gaia_pmdec_%s'%use_mean] - Gaia_HST_table.loc[id_pms, 'pmdec'])))
      print('RMS(PM_HST+Gaia - PM_Gaia) = (%.4e, %.4e) m.a.s.' %(hst_gaia_pmra_lsqt_evo[-1], hst_gaia_pmdec_lsqt_evo[-1]))

      if iteration >= (max_iterations-1):
         print('\nWARNING: Max number of iterations reached: Something might have gone wrong!\nPlease check the results carefully.')
         convergence = True

      elif iteration > 0:
         pmra_diff_evo.append(np.nanmean(pmra_evo[-1] - pmra_evo[-2]))
         pmdec_diff_evo.append(np.nanmean(pmdec_evo[-1] - pmdec_evo[-2]))
         
         # If rewind_stars is True, the code will converge when the difference between interations is smaller than the error in PMs
         threshold = np.nanmean(Gaia_HST_table.loc[id_pms, ['relative_hst_gaia_pmra_%s_error'%use_mean, 'relative_hst_gaia_pmdec_%s_error'%use_mean]].mean())*1e-1

         print('PM variation = (%.4e, %.4e)  m.a.s.' %(pmra_diff_evo[-1], pmdec_diff_evo[-1]))

         if rewind_stars:
            print('Threshold = %.4e  m.a.s.'%threshold)
            convergence = (np.abs(pmra_diff_evo[-1]) <= threshold) & (np.abs(pmdec_diff_evo[-1]) <= threshold)
         else:
            convergence = Gaia_HST_table.use_for_alignment.equals(previous_use_for_alignment)

      if ask_user_stop & (iteration > 0) & (quiet == False):
         with plt.rc_context(rc={'interactive': False}):
            plt.gcf().show()
         try:
            print('\nCheck the preliminary results in the VPD.')
            continue_loop = input('Continue with the next iteration? ') or 'y'
            continue_loop = str2bool(continue_loop)
            print('')
            if not continue_loop:
               convergence = True
         except:
            print('WARNING: Answer not understood. Continuing execution.')

      previous_use_for_alignment = Gaia_HST_table.use_for_alignment.copy()
      iteration += 1

   if (n_images > 1) and (n_processes != 1):
      pool.close()

   Gaia_HST_table = Gaia_HST_table[Gaia_HST_table['relative_hst_gaia_pmdec_%s'%use_mean].notnull() & Gaia_HST_table['relative_hst_gaia_pmra_%s'%use_mean].notnull()]

   pmra_evo = np.array(pmra_evo).T
   pmdec_evo = np.array(pmdec_evo).T

   hst_gaia_pm_lsqt_evo = np.array([hst_gaia_pmra_lsqt_evo, hst_gaia_pmdec_lsqt_evo]).T
   pm_diff_evo = np.array([pmra_diff_evo, pmdec_diff_evo]).T

   if (iteration > 1) and (no_plots == False):
      try:
         plt.close()
         fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True)

         ax1.plot(np.arange(iteration-2)+1, np.abs(pm_diff_evo[:,0]), '-', label = r'$\Delta(\mu_{\alpha *})$')
         ax1.plot(np.arange(iteration-2)+1, np.abs(pm_diff_evo[:,1]), '-', label = r'$\Delta(\mu_{\delta})$')

         ax2.plot(np.arange(iteration), hst_gaia_pm_lsqt_evo[:,0], '-', label = r'RMS$(\mu_{\alpha *, HST+Gaia} - \mu_{\alpha *, Gaia})$')
         ax2.plot(np.arange(iteration), hst_gaia_pm_lsqt_evo[:,1], '-', label = r'RMS$(\mu_{\delta, HST+Gaia} - \mu_{\delta, Gaia})$')

         ax1.axhline(y=threshold, linewidth = 0.75, color = 'k')
         ax1.set_ylabel(r'$\Delta(\mu)$')
         ax2.set_ylabel(r'RMS$(\mu_{HST-Gaia} - \mu_{Gaia})$')
         ax2.set_xlabel(r'iteration #')

         ax1.grid()
         ax2.grid()

         ax1.legend(shadow=True, fancybox=True)
         ax2.legend(shadow=True, fancybox=True)

         plt.savefig('%s_PM_RMS_iterations.pdf'%plot_name, bbox_inches='tight')
         plt.close()
      except:
         pass

   return Gaia_HST_table, lnks


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

      footprint_chip[np.where(footprint_chip[:,0] < center_chip[0]),0] -= 0.0028 / np.cos(np.deg2rad(center_chip[1]))
      footprint_chip[np.where(footprint_chip[:,0] > center_chip[0]),0] += 0.0028 / np.cos(np.deg2rad(center_chip[1]))

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


def weighted_avg_err(table):
   """
   Weighted average its error and the standard deviation.
   """

   var_cols = [x for x in table.columns if not '_error' in x]
   var_cols_err = ['%s_error'%col for col in var_cols]

   x_i = table.loc[:, var_cols]
   ex_i = table.reindex(columns = var_cols_err)
   ex_i.columns = ex_i.columns.str.rstrip('_error')

   weighted_variance = (1./(1./ex_i**2).sum(axis = 0))
   weighted_avg = ((x_i.div(ex_i**2)).sum(axis = 0) * weighted_variance).add_suffix('_wmean')
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
   """
   Add text with stroke inside plots 
   """
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


def bin_errors(x, y, y_error, n_bins = 10):
   """
   Binned statistics
   """

   bins = np.quantile(x, np.linspace(0,1,n_bins+1))
   bins_centers = (bins[:-1] + bins[1:]) / 2

   mean = stats.binned_statistic(x, y, statistic='mean', bins=bins).statistic
   std = stats.binned_statistic(x, y, statistic='std', bins=bins).statistic
   mean_error = stats.binned_statistic(x, y_error, statistic='mean', bins=bins).statistic

   return bins_centers, mean, std, mean_error
   

def plot_results(table, lnks, hst_image_list, HST_path, avg_pm, use_mean = 'wmean', use_members = True, plot_name_1 = 'output1', plot_name_2 = 'output2', plot_name_3 = 'output3', plot_name_4 = 'output4', plot_name_5 = 'output5', plot_name_6 = 'output6', plot_name_7 = 'output7', ext = '.pdf'):
   """
   Plot results
   """
   
   GDR = '(E)DR3'
   GaiaHub_GDR = 'GaiaHub + %s'%GDR
   sigma_lims = 3

   pmra_lims = [table['hst_gaia_pmra_%s'%use_mean].mean()-sigma_lims*table['hst_gaia_pmra_%s'%use_mean].std(), table['hst_gaia_pmra_%s'%use_mean].mean()+sigma_lims*table['hst_gaia_pmra_%s'%use_mean].std()]
   pmdec_lims = [table['hst_gaia_pmdec_%s'%use_mean].mean()-sigma_lims*table['hst_gaia_pmdec_%s'%use_mean].std(), table['hst_gaia_pmdec_%s'%use_mean].mean()+sigma_lims*table['hst_gaia_pmdec_%s'%use_mean].std()]
   
   # Plot the VPD and errors
   
   plt.close('all')
   
   fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex = False, sharey = False, figsize = (10, 3))

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
   add_inner_title(ax1, GDR, loc=1)


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
   add_inner_title(ax2, GaiaHub_GDR, loc=1)

   ax3.plot(table.gmag[table.use_for_alignment == False], np.sqrt(0.5*table.pmra_error[table.use_for_alignment == False]**2 + 0.5*table.pmdec_error[table.use_for_alignment == False]**2), 'k.', ms = 1, alpha = 0.35)
   ax3.plot(table.gmag[table.use_for_alignment == True], np.sqrt(0.5*table.pmra_error[table.use_for_alignment == True]**2 + 0.5*table.pmdec_error[table.use_for_alignment == True]**2), 'k.', ms = 1, label = GDR)

   ax3.plot(table.gmag[table.use_for_alignment == False], np.sqrt(0.5*table['hst_gaia_pmra_%s_error'%use_mean][table.use_for_alignment == False]**2 + 0.5*table['hst_gaia_pmdec_%s_error'%use_mean][table.use_for_alignment == False]**2), 'r.', ms = 1, alpha = 0.35)
   ax3.plot(table.gmag[table.use_for_alignment == True], np.sqrt(0.5*table['hst_gaia_pmra_%s_error'%use_mean][table.use_for_alignment == True]**2 + 0.5*table['hst_gaia_pmdec_%s_error'%use_mean][table.use_for_alignment == True]**2), 'r.', ms = 1, label = GaiaHub_GDR)

   ax3.grid()
   ax3.legend(prop={'size': 8})

   try:
      ax3.set_ylim(0, np.sqrt(table.pmra_error**2+table.pmdec_error**2).max())
   except:
      pass
   ax3.set_xlabel(r'$G$')
   ax3.set_ylabel(r'$\Delta\mu$ [m.a.s./yr.]')

   plt.subplots_adjust(wspace=0.3, hspace=0.1)

   plt.savefig(plot_name_1+ext, bbox_inches='tight')

   plt.close('all')


   # Plot the difference between Gaia and HST+Gaia

   fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex = False, sharey = False, figsize = (10, 3))
   ax1.errorbar(table.pmra, table['hst_gaia_pmra_%s'%use_mean], xerr=table.pmra_error, yerr=table['hst_gaia_pmra_%s_error'%use_mean], fmt = '.', ms=2, color = '0.1', zorder = 1, alpha = 0.5, elinewidth = 0.5)
   ax1.plot([pmra_lims[0], pmra_lims[1]], [pmra_lims[0], pmra_lims[1]], 'r-', linewidth = 0.5)
   ax1.grid()
   ax1.set_xlabel(r'$\mu_{\alpha*, Gaia}$ [m.a.s./yr.]')
   ax1.set_ylabel(r'$\mu_{\alpha*, HST + Gaia }$ [m.a.s./yr.]')
   try:
      ax1.set_xlim(pmra_lims)
      ax1.set_ylim(pmra_lims)      
   except:
      pass

   ax2.errorbar(table.pmdec, table['hst_gaia_pmdec_%s'%use_mean], xerr=table.pmdec_error, yerr=table['hst_gaia_pmdec_%s_error'%use_mean], fmt = '.', ms=2, color = '0.1', zorder = 1, alpha = 0.5, elinewidth = 0.5)
   ax2.plot([pmdec_lims[0], pmdec_lims[1]], [pmdec_lims[0], pmdec_lims[1]], 'r-', linewidth = 0.5)
   ax2.grid()
   ax2.set_xlabel(r'$\mu_{\delta, Gaia}$ [m.a.s./yr.]')
   ax2.set_ylabel(r'$\mu_{\delta, HST + Gaia}$ [m.a.s./yr.]')
   try:
      ax2.set_xlim(pmdec_lims)
      ax2.set_ylim(pmdec_lims)      
   except:
      pass

   ax3.errorbar(table.pmra - table['hst_gaia_pmra_%s'%use_mean], table.pmdec - table['hst_gaia_pmdec_%s'%use_mean], xerr=np.sqrt(table.pmra_error**2 + table['hst_gaia_pmra_%s_error'%use_mean]**2), yerr=np.sqrt(table.pmdec_error**2 + table['hst_gaia_pmdec_%s_error'%use_mean]**2), fmt = '.', ms=2, color = '0.1', zorder = 1, alpha = 0.5, elinewidth = 0.5)
   
   ax3.axvline(x=0, color ='r', linewidth = 0.5)
   ax3.axhline(y=0, color ='r', linewidth = 0.5)

   ax3.grid()
   ax3.set_aspect('equal', adjustable='datalim')
   
   ax3.set_xlabel(r'$\mu_{\alpha*, Gaia}$ - $\mu_{\alpha*, HST + Gaia}$ [m.a.s./yr.]')
   ax3.set_ylabel(r'$\mu_{\delta, Gaia}$ - $\mu_{\delta, HST + Gaia}$ [m.a.s./yr.]')

   plt.subplots_adjust(wspace=0.3, hspace=0.1)

   plt.savefig(plot_name_2+ext, bbox_inches='tight')


   plt.close('all')

   # Plot the CMD

   fig, ax = plt.subplots(1,1, sharex = False, sharey = False, figsize = (5., 5.))

   hst_filters = [col for col in table.columns if ('F' in col) & ('error' not in col) & ('std' not in col) & ('_mean' not in col)]
   hst_filters.sort()
   if len(hst_filters) >= 2:
      name = r'%s - %s'%(hst_filters[0], hst_filters[1])
      color = (table[hst_filters[0]]-table[hst_filters[1]]).rename(name)
      mag = table[hst_filters[1]]
   else:
      name = r'G - %s'%hst_filters[0]
      color = (table['gmag']-table[hst_filters[0]]).rename(name)
      mag = table[hst_filters[0]]

   ax.plot(color[table.use_for_alignment == False], mag[table.use_for_alignment == False], 'k.', ms=2, alpha = 0.35)
   ax.plot(color[table.use_for_alignment == True], mag[table.use_for_alignment == True], 'k.', ms=2)

   ax.set_xlabel(color.name.replace("_wmean","").replace("_mean",""))
   ax.set_ylabel(mag.name.replace("_wmean","").replace("_mean",""))

   try:
      ax.set_xlim(np.nanmin(color)-0.1, np.nanmax(color)+0.1)
      ax.set_ylim(np.nanmax(mag)+0.25, np.nanmin(mag)-0.25)
   except:
      pass
   
   ax.grid()
   
   plt.savefig(plot_name_3+ext, bbox_inches='tight')

   plt.close('all')


   # Plot the sky projection

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

   # Identify stars with Gaia PMs
   id_gaia = np.isfinite(table['pmra']) & np.isfinite(table['pmdec'])
   id_hst_gaia = np.isfinite(table['hst_gaia_pmra_%s'%use_mean]) & np.isfinite(table['hst_gaia_pmdec_%s'%use_mean])

   fig = plt.figure(figsize=(5., 5.), dpi = 250)
   ax = fig.add_subplot(111, projection=wcs)

   norm = ImageNormalize(image_data, interval = ManualInterval(0.0,0.15))

   im = ax.imshow(image_data, cmap='gray_r', origin='lower', norm=norm, zorder = 0)

   ax.set_xlabel("RA")
   ax.set_ylabel("Dec")

   if (table.use_for_alignment.sum() < len(table)) & (table.use_for_alignment.sum() > 0):
      try:
         p1 = ax.scatter(table['ra'][table.use_for_alignment == False & id_gaia], table['dec'][table.use_for_alignment == False & id_gaia], transform=ax.get_transform('world'), s=10, linewidth = 1, facecolor='none', edgecolor='k', alpha = 0.35, zorder = 1)
         p1 = ax.scatter(table['ra'][table.use_for_alignment == False & id_hst_gaia], table['dec'][table.use_for_alignment == False & id_hst_gaia], transform=ax.get_transform('world'), s=30, linewidth = 1, facecolor='none', edgecolor='r', alpha = 0.35, zorder = 2)
      except:
         pass
      try:
         with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q1 = ax.quiver(table['ra'][table.use_for_alignment == False & id_hst_gaia], table['dec'][table.use_for_alignment == False & id_hst_gaia], -table['hst_gaia_pmra_%s'%use_mean][table.use_for_alignment == False & id_hst_gaia], table['hst_gaia_pmdec_%s'%use_mean][table.use_for_alignment == False & id_hst_gaia], transform=ax.get_transform('world'), width = 0.003, angles = 'xy', color='r', alpha = 0.35, zorder = 2)
      except:
         pass
   if table.use_for_alignment.sum() > 0:
      try:
         p2 = ax.scatter(table['ra'][table.use_for_alignment == True & id_gaia], table['dec'][table.use_for_alignment == True & id_gaia], transform=ax.get_transform('world'), s=10, linewidth = 1, facecolor='none', label=GDR, edgecolor='k', zorder = 1)
         p2 = ax.scatter(table['ra'][table.use_for_alignment == True & id_hst_gaia], table['dec'][table.use_for_alignment == True & id_hst_gaia], transform=ax.get_transform('world'), s=30, linewidth = 1, facecolor='none', label=GaiaHub_GDR, edgecolor='r', zorder = 2)
      except:
         pass
      try:
         with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q2 = ax.quiver(table['ra'][table.use_for_alignment == True & id_hst_gaia],table['dec'][table.use_for_alignment == True & id_hst_gaia], -table['hst_gaia_pmra_%s'%use_mean][table.use_for_alignment == True & id_hst_gaia], table['hst_gaia_pmdec_%s'%use_mean][table.use_for_alignment == True & id_hst_gaia], transform=ax.get_transform('world'), width = 0.003, angles = 'xy', color='r', zorder = 2)
      except:
         pass

   ax.grid()

   plt.legend()
   
   plt.tight_layout()
   plt.savefig(plot_name_4+'.pdf', bbox_inches='tight')
   
   plt.close('all')
   
   
   # Systematics plot
   saturation_qfit = lnks.q_hst.max()*0.8
   
   typical_dispersion = (avg_pm['hst_gaia_pmra_%s_std'%(use_mean)] + avg_pm['hst_gaia_pmdec_%s_std'%(use_mean)]) / 2
   pmra_lims = [avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)] - 5 * typical_dispersion, avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)] + 5 * typical_dispersion]
   pmdec_lims = [avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)] - 5 * typical_dispersion, avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)] + 5 * typical_dispersion]
   

   # Astrometric plots
   
   id_std_hst_gaia = np.isfinite(lnks.xc_hst) & np.isfinite(lnks.yc_hst) & np.isfinite(lnks.relative_hst_gaia_pmra) & np.isfinite(lnks.relative_hst_gaia_pmdec)
   lnks_members = lnks.index.isin(table[table.use_for_alignment == True].index) & np.isfinite(lnks.xc_hst) & np.isfinite(lnks.yc_hst) & np.isfinite(lnks.relative_hst_gaia_pmra) & np.isfinite(lnks.relative_hst_gaia_pmdec)

   fig, axs = plt.subplots(2, 2, sharex = False, sharey = False, figsize = (10, 5))

   bin_xhst, mean_xhst_pmra_hst_gaia, std_xhst_pmra_hst_gaia, mean_xhst_pmra_error_hst_gaia = bin_errors(lnks.xc_hst[lnks_members], lnks.relative_hst_gaia_pmra[lnks_members], lnks.relative_hst_gaia_pmra_error[lnks_members], n_bins = 10)
   bin_yhst, mean_yhst_pmra_hst_gaia, std_yhst_pmra_hst_gaia, mean_yhst_pmra_error_hst_gaia = bin_errors(lnks.yc_hst[lnks_members], lnks.relative_hst_gaia_pmra[lnks_members], lnks.relative_hst_gaia_pmra_error[lnks_members], n_bins = 10)

   bin_xhst, mean_xhst_pmdec_hst_gaia, std_xhst_pmdec_hst_gaia, mean_xhst_pmdec_error_hst_gaia = bin_errors(lnks.xc_hst[lnks_members], lnks.relative_hst_gaia_pmdec[lnks_members], lnks.relative_hst_gaia_pmdec_error[lnks_members], n_bins = 10)
   bin_yhst, mean_yhst_pmdec_hst_gaia, std_yhst_pmdec_hst_gaia, mean_yhst_pmdec_error_hst_gaia = bin_errors(lnks.yc_hst[lnks_members], lnks.relative_hst_gaia_pmdec[lnks_members], lnks.relative_hst_gaia_pmdec_error[lnks_members], n_bins = 10)

   q_hst = axs[0,0].scatter(lnks.xc_hst[lnks_members], lnks.relative_hst_gaia_pmra[lnks_members] + avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)], c = lnks.q_hst[lnks_members], s = 1)
   axs[0,0].scatter(lnks.xc_hst[~lnks_members], lnks.relative_hst_gaia_pmra[~lnks_members] + avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)], c = lnks.q_hst[~lnks_members], s = 1, alpha = 0.35)
   axs[0,0].axhline(y = avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'k')

   axs[0,0].plot(bin_xhst, mean_xhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'r', label='mean')
   axs[0,0].plot(bin_xhst, mean_xhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)] + std_xhst_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r', label=r'$\sigma$')
   axs[0,0].plot(bin_xhst, mean_xhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)] - std_xhst_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,0].plot(bin_xhst, mean_xhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)] + mean_xhst_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r', label=r'$error$')
   axs[0,0].plot(bin_xhst, mean_xhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)] - mean_xhst_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')


   axs[0,1].scatter(lnks.xc_hst[lnks_members], lnks.relative_hst_gaia_pmdec[lnks_members] + avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)], c = lnks.q_hst[lnks_members], s = 1)
   axs[0,1].scatter(lnks.xc_hst[~lnks_members], lnks.relative_hst_gaia_pmdec[~lnks_members] + avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)], c = lnks.q_hst[~lnks_members], s = 1, alpha = 0.35)
   axs[0,1].axhline(y = avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'k')

   axs[0,1].plot(bin_xhst, mean_xhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'r')
   axs[0,1].plot(bin_xhst, mean_xhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)] + std_xhst_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_xhst, mean_xhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)] - std_xhst_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_xhst, mean_xhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)] + mean_xhst_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_xhst, mean_xhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)] - mean_xhst_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')


   axs[1,0].scatter(lnks.yc_hst[lnks_members], lnks.relative_hst_gaia_pmra[lnks_members] + avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)], c = lnks.q_hst[lnks_members], s = 1)
   axs[1,0].scatter(lnks.yc_hst[~lnks_members], lnks.relative_hst_gaia_pmra[~lnks_members] + avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)], c = lnks.q_hst[~lnks_members], s = 1, alpha = 0.35)
   axs[1,0].axhline(y = avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'k')

   axs[1,0].plot(bin_yhst, mean_yhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'r')
   axs[1,0].plot(bin_yhst, mean_yhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)] + std_yhst_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[1,0].plot(bin_yhst, mean_yhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)] - std_yhst_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[1,0].plot(bin_yhst, mean_yhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)] + mean_yhst_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
   axs[1,0].plot(bin_yhst, mean_yhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)] - mean_yhst_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')


   axs[1,1].scatter(lnks.yc_hst[lnks_members], lnks.relative_hst_gaia_pmdec[lnks_members] + avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)], c = lnks.q_hst[lnks_members], s = 1)
   axs[1,1].scatter(lnks.yc_hst[~lnks_members], lnks.relative_hst_gaia_pmdec[~lnks_members] + avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)], c = lnks.q_hst[~lnks_members], s = 1, alpha = 0.35)
   axs[1,1].axhline(y = avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'k')

   axs[1,1].plot(bin_yhst, mean_yhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'r')
   axs[1,1].plot(bin_yhst, mean_yhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)] + std_yhst_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[1,1].plot(bin_yhst, mean_yhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)] - std_yhst_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[1,1].plot(bin_yhst, mean_yhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)] + mean_yhst_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
   axs[1,1].plot(bin_yhst, mean_yhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)] - mean_yhst_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')

   # Set titles
   [ax.set_ylabel(r'$\mu_{\alpha*}$ [m.a.s./yr.]') for ax in axs[:,0]]
   [ax.set_ylabel(r'$\mu_{\delta}$ [m.a.s./yr.]') for ax in axs[:,1]]
   [ax.set_xlabel(r'x [pix]') for ax in axs[0,:]]
   [ax.set_xlabel(r'y [pix]') for ax in axs[1,:]]
   [ax.grid() for ax in axs.flatten()]
   
   axs[0,0].legend()


   # Set limits
   [ax.set_ylim(pmra_lims) for ax in axs[:,0]]
   [ax.set_ylim(pmdec_lims) for ax in axs[:,1]]

   plt.subplots_adjust(wspace=0.5, hspace=0.3, right = 0.8)
   
   # Colorbar
   cbar = fig.colorbar(q_hst, ax=axs.ravel().tolist(), ticks=[lnks.q_hst.min(), saturation_qfit, lnks.q_hst.max()], aspect = 40, pad = 0.05)
   cbar.ax.set_yticklabels([str(round(lnks.q_hst.min(), 1)), 'sat', str(round(lnks.q_hst.max(), 1))])
   cbar.set_label('qfit')
   cbar.ax.plot([0, 1], [saturation_qfit, saturation_qfit], 'k')

   plt.savefig(plot_name_5+ext, bbox_inches='tight')

   
   plt.close('all')
   # Photometry plots

   id_std_gaia = np.isfinite(table['pmra']) & np.isfinite(table['pmdec']) & (table.use_for_alignment == True)
   id_std_hst_gaia = np.isfinite(table['hst_gaia_pmra_%s'%use_mean]) & np.isfinite(table['hst_gaia_pmdec_%s'%use_mean]) & (table.use_for_alignment == True)

   bin_mag, mean_mag_pmra_hst_gaia, std_mag_pmra_hst_gaia, mean_mag_pmra_error_hst_gaia = bin_errors(table.gmag[id_std_hst_gaia], table['hst_gaia_pmra_%s'%use_mean][id_std_hst_gaia], table['hst_gaia_pmra_%s_error'%use_mean][id_std_hst_gaia], n_bins = 10)
   bin_mag, mean_mag_pmdec_hst_gaia, std_mag_pmdec_hst_gaia, mean_mag_pmdec_error_hst_gaia = bin_errors(table.gmag[id_std_hst_gaia], table['hst_gaia_pmdec_%s'%use_mean][id_std_hst_gaia], table['hst_gaia_pmdec_%s_error'%use_mean][id_std_hst_gaia], n_bins = 10)

   # Find HST filters
   hst_filters = [col for col in table.columns if ('F' in col) & ('error' not in col) & ('std' not in col) & ('_mean' not in col)]
   hst_filters.sort()

   fig, axs = plt.subplots(len(hst_filters)+1, 2, sharex = False, sharey = False, figsize = (10.0, 3*len(hst_filters)+1))

   axs[0,0].scatter(table.gmag[table.use_for_alignment == False], table['hst_gaia_pmra_%s'%use_mean][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
   axs[0,0].scatter(table.gmag[table.use_for_alignment == True], table['hst_gaia_pmra_%s'%use_mean][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)
   
   axs[0,0].axhline(y = avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'k')
   
   axs[0,0].plot(bin_mag, mean_mag_pmra_hst_gaia, linestyle = '-', linewidth = 1, color = 'r', label=r'mean')
   axs[0,0].plot(bin_mag, mean_mag_pmra_hst_gaia + std_mag_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r', label=r'$\sigma$')
   axs[0,0].plot(bin_mag, mean_mag_pmra_hst_gaia - std_mag_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,0].plot(bin_mag, mean_mag_pmra_hst_gaia + mean_mag_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r', label=r'error')
   axs[0,0].plot(bin_mag, mean_mag_pmra_hst_gaia - mean_mag_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')


   axs[0,1].scatter(table.gmag[table.use_for_alignment == False], table['hst_gaia_pmdec_%s'%use_mean][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
   axs[0,1].scatter(table.gmag[table.use_for_alignment == True], table['hst_gaia_pmdec_%s'%use_mean][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)

   axs[0,1].axhline(y = avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'k')
   axs[0,1].plot(bin_mag, mean_mag_pmdec_hst_gaia, linestyle = '-', linewidth = 1, color = 'r')
   axs[0,1].plot(bin_mag, mean_mag_pmdec_hst_gaia + std_mag_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_mag, mean_mag_pmdec_hst_gaia - std_mag_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_mag, mean_mag_pmdec_hst_gaia + mean_mag_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_mag, mean_mag_pmdec_hst_gaia - mean_mag_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')

   for ii, cmd_filter in enumerate(hst_filters):
      hst_filter = table[cmd_filter].rename('%s'%cmd_filter.replace("_wmean","").replace("_mean",""))
   
      id_std_hst_gaia = np.isfinite(table['hst_gaia_pmra_%s'%use_mean]) & np.isfinite(table['hst_gaia_pmdec_%s'%use_mean]) & np.isfinite(hst_filter) & (table.use_for_alignment == True)
      
      bin_mag, mean_mag_pmra_hst_gaia, std_mag_pmra_hst_gaia, mean_mag_pmra_error_hst_gaia = bin_errors(hst_filter[id_std_hst_gaia], table['hst_gaia_pmra_%s'%use_mean][id_std_hst_gaia], table['hst_gaia_pmra_%s_error'%use_mean][id_std_hst_gaia], n_bins = 10)
      bin_mag, mean_mag_pmdec_hst_gaia, std_mag_pmdec_hst_gaia, mean_mag_pmdec_error_hst_gaia = bin_errors(hst_filter[id_std_hst_gaia], table['hst_gaia_pmdec_%s'%use_mean][id_std_hst_gaia], table['hst_gaia_pmdec_%s_error'%use_mean][id_std_hst_gaia], n_bins = 10)

      axs[ii+1,0].scatter(hst_filter[table.use_for_alignment == False], table['hst_gaia_pmra_%s'%use_mean][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
      axs[ii+1,0].scatter(hst_filter[table.use_for_alignment == True], table['hst_gaia_pmra_%s'%use_mean][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)

      axs[ii+1,0].axhline(y = avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'k')
      axs[ii+1,0].plot(bin_mag, mean_mag_pmra_hst_gaia, linestyle = '-', linewidth = 1, color = 'r')
      axs[ii+1,0].plot(bin_mag, mean_mag_pmra_hst_gaia + std_mag_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,0].plot(bin_mag, mean_mag_pmra_hst_gaia - std_mag_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,0].plot(bin_mag, mean_mag_pmra_hst_gaia + mean_mag_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
      axs[ii+1,0].plot(bin_mag, mean_mag_pmra_hst_gaia - mean_mag_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')

      axs[ii+1,1].scatter(hst_filter[table.use_for_alignment == False], table['hst_gaia_pmdec_%s'%use_mean][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
      axs[ii+1,1].scatter(hst_filter[table.use_for_alignment == True], table['hst_gaia_pmdec_%s'%use_mean][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)

      axs[ii+1,1].axhline(y = avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'k')
      axs[ii+1,1].plot(bin_mag, mean_mag_pmdec_hst_gaia, linestyle = '-', linewidth = 1, color = 'r')
      axs[ii+1,1].plot(bin_mag, mean_mag_pmdec_hst_gaia + std_mag_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,1].plot(bin_mag, mean_mag_pmdec_hst_gaia - std_mag_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,1].plot(bin_mag, mean_mag_pmdec_hst_gaia + mean_mag_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
      axs[ii+1,1].plot(bin_mag, mean_mag_pmdec_hst_gaia - mean_mag_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')

      axs[ii+1,0].set_xlabel(hst_filter.name)
      axs[ii+1,1].set_xlabel(hst_filter.name)


   # Set titles
   [ax.set_ylabel(r'$\mu_{\alpha*}$ [m.a.s./yr.]') for ax in axs[:,0]]
   [ax.set_ylabel(r'$\mu_{\delta}$ [m.a.s./yr.]') for ax in axs[:,1]]
   [ax.set_xlabel(r'G') for ax in axs[0,:]]
   [ax.grid() for ax in axs.flatten()]
   
   axs[0,0].legend()
   
   # Set limits
   [ax.set_ylim(pmra_lims) for ax in axs[:,0]]
   [ax.set_ylim(pmdec_lims) for ax in axs[:,1]]

   plt.subplots_adjust(wspace=0.5, hspace=0.3, right = 0.8)
   
   # Colorbar
   cbar = fig.colorbar(q_hst, ax=axs.ravel().tolist(), ticks=[lnks.q_hst.min(), saturation_qfit, lnks.q_hst.max()], aspect = 40, pad = 0.05)
   cbar.ax.set_yticklabels([str(round(lnks.q_hst.min(), 1)), 'sat', str(round(lnks.q_hst.max(), 1))])
   cbar.set_label('qfit')
   cbar.ax.plot([0, 1], [saturation_qfit, saturation_qfit], 'k')

   plt.savefig(plot_name_6+ext, bbox_inches='tight')


   plt.close('all')

   # Color figures
   id_std_gaia = np.isfinite(table['pmra']) & np.isfinite(table['pmdec']) & (table.use_for_alignment == True)
   id_std_hst_gaia = np.isfinite(table['hst_gaia_pmra_%s'%use_mean]) & np.isfinite(table['hst_gaia_pmdec_%s'%use_mean]) & (table.use_for_alignment == True) & np.isfinite(table['bp_rp'])

   bin_color, mean_color_pmra_hst_gaia, std_color_pmra_hst_gaia, mean_color_pmra_error_hst_gaia = bin_errors(table.bp_rp[id_std_hst_gaia], table['hst_gaia_pmra_%s'%use_mean][id_std_hst_gaia], table['hst_gaia_pmra_%s_error'%use_mean][id_std_hst_gaia], n_bins = 10)
   bin_color, mean_color_pmdec_hst_gaia, std_color_pmdec_hst_gaia, mean_color_pmdec_error_hst_gaia = bin_errors(table.bp_rp[id_std_hst_gaia], table['hst_gaia_pmdec_%s'%use_mean][id_std_hst_gaia], table['hst_gaia_pmdec_%s_error'%use_mean][id_std_hst_gaia], n_bins = 10)

   fig, axs = plt.subplots(len(hst_filters)+1, 2, sharex = False, sharey = False, figsize = (10, 3*len(hst_filters)+1))

   axs[0,0].scatter(table.bp_rp[table.use_for_alignment == False], table['hst_gaia_pmra_%s'%use_mean][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
   axs[0,0].scatter(table.bp_rp[table.use_for_alignment == True], table['hst_gaia_pmra_%s'%use_mean][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)
   
   axs[0,0].axhline(y = avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'k')
   axs[0,0].plot(bin_color, mean_color_pmra_hst_gaia, linestyle = '-', linewidth = 1, color = 'r', label = 'mean')
   axs[0,0].plot(bin_color, mean_color_pmra_hst_gaia + std_color_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r', label = r'$\sigma$')
   axs[0,0].plot(bin_color, mean_color_pmra_hst_gaia - std_color_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,0].plot(bin_color, mean_color_pmra_hst_gaia + mean_color_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r', label = 'error')
   axs[0,0].plot(bin_color, mean_color_pmra_hst_gaia - mean_color_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')


   axs[0,1].scatter(table.bp_rp[table.use_for_alignment == False], table['hst_gaia_pmdec_%s'%use_mean][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
   axs[0,1].scatter(table.bp_rp[table.use_for_alignment == True], table['hst_gaia_pmdec_%s'%use_mean][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)

   axs[0,1].axhline(y = avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'k')
   axs[0,1].plot(bin_color, mean_color_pmdec_hst_gaia, linestyle = '-', linewidth = 1, color = 'r')
   axs[0,1].plot(bin_color, mean_color_pmdec_hst_gaia + std_color_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_color, mean_color_pmdec_hst_gaia - std_color_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_color, mean_color_pmdec_hst_gaia + mean_color_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_color, mean_color_pmdec_hst_gaia - mean_color_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')

   for ii, cmd_filter in enumerate(hst_filters):
      if int(re.findall(r'\d+', cmd_filter)[0]) <= 606:
         HST_Gaia_color = (table[cmd_filter]-table['gmag']).rename('%s - G'%cmd_filter.replace("_wmean","").replace("_mean",""))
      else:
         HST_Gaia_color = (table['gmag']-table[cmd_filter]).rename('G - %s'%cmd_filter.replace("_wmean","").replace("_mean",""))

      id_std_hst_gaia = np.isfinite(table['hst_gaia_pmra_%s'%use_mean]) & np.isfinite(table['hst_gaia_pmdec_%s'%use_mean]) & np.isfinite(HST_Gaia_color) & (table.use_for_alignment == True)

      bin_color, mean_color_pmra_hst_gaia, std_color_pmra_hst_gaia, mean_color_pmra_error_hst_gaia = bin_errors(HST_Gaia_color[id_std_hst_gaia], table['hst_gaia_pmra_%s'%use_mean][id_std_hst_gaia], table['hst_gaia_pmra_%s_error'%use_mean][id_std_hst_gaia], n_bins = 10)
      bin_color, mean_color_pmdec_hst_gaia, std_color_pmdec_hst_gaia, mean_color_pmdec_error_hst_gaia = bin_errors(HST_Gaia_color[id_std_hst_gaia], table['hst_gaia_pmdec_%s'%use_mean][id_std_hst_gaia], table['hst_gaia_pmdec_%s_error'%use_mean][id_std_hst_gaia], n_bins = 10)

      axs[ii+1,0].scatter(HST_Gaia_color[table.use_for_alignment == False], table['hst_gaia_pmra_%s'%use_mean][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
      axs[ii+1,0].scatter(HST_Gaia_color[table.use_for_alignment == True], table['hst_gaia_pmra_%s'%use_mean][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)

      axs[ii+1,0].axhline(y = avg_pm['hst_gaia_pmra_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'k')
      axs[ii+1,0].plot(bin_color, mean_color_pmra_hst_gaia, linestyle = '-', linewidth = 1, color = 'r')
      axs[ii+1,0].plot(bin_color, mean_color_pmra_hst_gaia + std_color_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,0].plot(bin_color, mean_color_pmra_hst_gaia - std_color_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,0].plot(bin_color, mean_color_pmra_hst_gaia + mean_color_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
      axs[ii+1,0].plot(bin_color, mean_color_pmra_hst_gaia - mean_color_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')

      axs[ii+1,1].scatter(HST_Gaia_color[table.use_for_alignment == False], table['hst_gaia_pmdec_%s'%use_mean][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
      axs[ii+1,1].scatter(HST_Gaia_color[table.use_for_alignment == True], table['hst_gaia_pmdec_%s'%use_mean][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)

      axs[ii+1,1].axhline(y = avg_pm['hst_gaia_pmdec_%s_%s'%(use_mean, use_mean)], linestyle = '-', linewidth = 1, color = 'k')
      axs[ii+1,1].plot(bin_color, mean_color_pmdec_hst_gaia, linestyle = '-', linewidth = 1, color = 'r')
      axs[ii+1,1].plot(bin_color, mean_color_pmdec_hst_gaia + std_color_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,1].plot(bin_color, mean_color_pmdec_hst_gaia - std_color_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,1].plot(bin_color, mean_color_pmdec_hst_gaia + mean_color_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
      axs[ii+1,1].plot(bin_color, mean_color_pmdec_hst_gaia - mean_color_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')

      axs[ii+1,0].set_xlabel(HST_Gaia_color.name)
      axs[ii+1,1].set_xlabel(HST_Gaia_color.name)


   # Set titles
   [ax.set_ylabel(r'$\mu_{\alpha*}$ [m.a.s./yr.]') for ax in axs[:,0]]
   [ax.set_ylabel(r'$\mu_{\delta}$ [m.a.s./yr.]') for ax in axs[:,1]]
   [ax.set_xlabel(r'$G$') for ax in axs[0,:]]
   [ax.grid() for ax in axs.flatten()]
   
   # Set limits
   [ax.set_ylim(pmra_lims) for ax in axs[:,0]]
   [ax.set_ylim(pmdec_lims) for ax in axs[:,1]]

   plt.subplots_adjust(wspace=0.5, hspace=0.3, right = 0.8)
   
   # Colorbar
   cbar = fig.colorbar(q_hst, ax=axs.ravel().tolist(), ticks=[lnks.q_hst.min(), saturation_qfit, lnks.q_hst.max()], aspect = 40, pad = 0.05)
   cbar.ax.set_yticklabels([str(round(lnks.q_hst.min(), 1)), 'sat', str(round(lnks.q_hst.max(), 1))])
   cbar.set_label('qfit')
   cbar.ax.plot([0, 1], [saturation_qfit, saturation_qfit], 'k')

   plt.savefig(plot_name_7+ext, bbox_inches='tight')
   

def get_object_properties(args):
   """
   This routine will try to obtain all the required object properties from Simbad or from the user.
   """

   print('\n'+'-'*42)
   print("Commencing execution")
   print('-'*42)

   #Try to get object:
   if (args.ra is None) or (args.dec is None):
      try:
         from astroquery.simbad import Simbad
         import astropy.units as u
         from astropy.coordinates import SkyCoord

         customSimbad = Simbad()
         customSimbad.add_votable_fields('dim')

         object_table = customSimbad.query_object(args.name)
         
         object_name = str(object_table['MAIN_ID'][0]).replace("b'NAME ","'").replace("b' ","'")
         coo = SkyCoord(ra = object_table['RA'], dec = object_table['DEC'], unit=(u.hourangle, u.deg))

         args.ra = float(coo.ra.deg)
         args.dec = float(coo.dec.deg)

         #Try to get the search radius
         if all((args.search_radius == None, any((args.search_width == None, args.search_height == None)))):
            if (object_table['GALDIM_MAJAXIS'].mask == False):
               args.search_radius = max(np.round(float(2. * object_table['GALDIM_MAJAXIS'] / 60.), 2), 0.1)

      except:
         object_name = args.name
         if ((args.ra is None) or (args.dec is None)) and (args.quiet is False):
            print('\n')
            try:
               if (args.ra is None):
                  args.ra = float(input('R.A. not defined, please enter R.A. in degrees: '))
               if args.dec is None:
                  args.dec = float(input('Dec not defined, please enter Dec in degrees: '))
            except:
               print('No valid input. Float number required.')
               print('\nExiting now.\n')
               sys.exit(1)

         elif ((args.ra is None) or (args.dec is None)) and (args.quiet is True):
            print('GaiaHub could not find the object coordinates. Please check that the name of the object is written correctly. You can also run GaiaHub deffining explictly the coordinates using the "--ra" and "--dec" options.')
            sys.exit(1)

   else:
      object_name = args.name

   if (args.search_radius is None) and (args.quiet is False):
      print('\n')
      args.search_radius = float(input('Search radius not defined, please enter the search radius in degrees (Press enter to adopt the default value of 0.25 deg): ') or 0.25)
   elif (args.search_radius is None) and (args.quiet is True):
      args.search_radius = 0.25

   if (args.search_height is None):
      try:
         args.search_height = 2.*args.search_radius
      except:
         args.search_height = 1.0
   if (args.search_width is None):
      try:
         args.search_width = np.abs(2.*args.search_radius/np.cos(np.deg2rad(args.dec)))
      except:
         args.search_width = 1.0

   setattr(args, 'area', args.search_height * args.search_width * np.abs(np.cos(np.deg2rad(args.dec))))

   if args.no_error_weighted:
      args.use_mean = 'mean'
   else:
      args.use_mean = 'wmean'

   if args.hst_filters == ['any']:
      args.hst_filters = ['F555W','F606W','F775W','F814W','F850LP']

   name_coo = 'ra_%.3f_dec_%.3f_r_%.2f'%(args.ra, args.dec, args.search_radius)

   if args.name is not None:
      args.name = args.name.replace(" ", "_")
      args.base_file_name = args.name+'_'+name_coo
   else:
      args.name = name_coo
      args.base_file_name = name_coo
   
   args.exec_path = './tmp_%s'%args.name
   
   #The script creates directories and set files names
   args.base_path = './%s/'%(args.name)
   args.HST_path = args.base_path+'HST/'
   args.Gaia_path = args.base_path+'Gaia/'
   args.Gaia_ind_queries_path = args.Gaia_path+'individual_queries/'
   
   args.used_HST_obs_table_filename = args.base_path + args.base_file_name+'_used_HST_images.csv'
   args.HST_Gaia_table_filename = args.base_path + args.base_file_name+'.csv'
   args.logfile = args.base_path + args.base_file_name+'.log'
   args.queries = args.Gaia_path + args.base_file_name+'_queries.log'
   
   args.Gaia_clean_table_filename = args.Gaia_path + args.base_file_name+'_gaia.csv'
   args.HST_obs_table_filename = args.HST_path + args.base_file_name+'_obs.csv'
   args.HST_data_table_products_filename = args.HST_path + args.base_file_name+'_data_products.csv'
   args.lnks_summary_filename = args.HST_path + args.base_file_name+'_lnks_summary.csv'

   args.date_second_epoch = Time('%4i-%02i-%02iT00:00:00.000'%(args.date_second_epoch[2], args.date_second_epoch[0], args.date_second_epoch[1])).mjd
   args.date_reference_second_epoch = Time(args.date_reference_second_epoch)
   
   # Set up the number of processors:
   args.n_processes = use_processors(args.n_processes)
   
   print('\n')
   print('-'*42)
   print('Search information')
   print('-'*42)
   print('- Object name:', object_name)
   print('- (ra, dec) = (%s, %s) deg.'%(round(args.ra, 5), round(args.dec, 5)))
   print('- Search radius = %s deg.'%args.search_radius)
   print('-'*42+'\n')

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
   

def get_real_error(table):
   """
   This routine calculates the excess of error that should be added to the listed Gaia errors. The lines are based on Fabricius 2021. Figure 21.
   """

   p5 = table.astrometric_params_solved == 31 # 5p parameters solved Brown et al. (2020)
   
   table.loc[:, ['parallax_error_old', 'pmra_error_old', 'pmdec_error_old']] = table.loc[:, ['parallax_error', 'pmra_error', 'pmdec_error']].values
   
   table.loc[p5, ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']] = table.loc[p5, ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']].values * 1.05
   table.loc[~p5, ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']] = table.loc[~p5, ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']].values * 1.22

   return table
