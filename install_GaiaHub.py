# This program will install all the necessary files to run StellarTeam

import os
import sys
import itertools
import subprocess
import re
import shutil

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


def cli_progress_test(current, end_val, bar_length=50):
   """
   Just a progress bar.
   """

   percent = float(current) / end_val
   hashes = '#' * int(round(percent * bar_length))
   spaces = ' ' * (bar_length - len(hashes))
   sys.stdout.write("\rProcessing: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
   sys.stdout.flush()


def listFD(url, ext=''):
   """
   This list all files with extension = ext in a url
   """
   import requests
   from bs4 import BeautifulSoup

   page = requests.get(url).text
   soup = BeautifulSoup(page, 'html.parser')
   return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]


def get_hst1pass(installation_folder, repo = 'https://www.stsci.edu/~jayander'):
   """
   This routine finds the HST1PASS
   """
   
   url = repo+'/HST1PASS'
   
   file_urls = listFD(url, ext='F')
   output_files = ['%s/fortran_codes/%s'%(installation_folder, os.path.split(file_url)[-1]) for file_url in file_urls]

   download_files(file_urls, output_files)
   print('\n')


def get_psfs_gd_libraries(installation_folder, repo = 'https://www.stsci.edu/~jayander', libs = ['STDPSFs', 'STDGDCs'], filters = ['F555W','F606W','F775W','F814W','F850LP'], instrument = ['ACSWFC', 'ACSHRC', 'WFC3UV', 'WFC3IR']):
   """
   This routine finds the required PSF and geometric distorsion libraries.
   """
   import requests
   from bs4 import BeautifulSoup

   def listFD(url, ext=''):
      page = requests.get(url).text
      soup = BeautifulSoup(page, 'html.parser')
      return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
   
   files_urls = []
   output_files = []

   for comb in itertools.product(libs, instrument):
      url = '%s/%s/%s'%(repo, comb[0], comb[1])
      for file_url in listFD(url, ext='fits'):
         file_output = '%s/lib/%s/%s/%s'%(installation_folder, comb[0], comb[1], os.path.split(file_url)[-1])
         if (not os.path.isfile(file_output)) and (any(x in file_url for x in filters)):
            files_urls.append(file_url)
            output_files.append(file_output)
            create_dir(os.path.split(file_output)[0])
   
   print('\nDownloading the PSF and Geometric distorsion libraries. This may take a while...\n')
   download_files(files_urls, output_files)
   print('\n')
   

def download_files(files_urls, output_files):
   """
   This routine downloads data.
   """
   import requests

   for ii, (file_url, file_output) in enumerate(zip(files_urls, output_files)):
      cli_progress_test(ii+1, len(files_urls))
      r = requests.get(file_url)
      with open(file_output, 'wb') as f:
         f.write(r.content)


def compile_fortran(compiler, installation_folder, raise_exception = False):
   """
   This routine will compile Fortran routines
   """ 
   
   path = installation_folder+'/fortran_codes/'
   f_modules = ['%s%s'%(path, file) for file in os.listdir(path) if file.endswith('F')]
   
   for input_file in f_modules:
      if input_file.endswith('FF'):
         os.rename(input_file, input_file.split('.')[0]+'.F')
         input_file = input_file.split('.')[0]+'.F'

      output_file = input_file.split('.')[0]+'.e'
      bashCommand = "%s %s -o %s"%(compiler, input_file, output_file)
      process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      output, error = process.communicate()
      if not os.path.isfile(output_file):
         print('WARNING: There was a problem compiling %s\n'%input_file)
         if raise_exception:
            raise Exception()
      else:
         os.chmod(output_file, 0o755)


def install_python_dependencies(packages):
   """
   This will install the required python dependencies
   """

   subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", packages])
   
   print('\n')


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


def replace_text(file_in, text_to_replace, replace_by):
   """
   This routine hardcodes the installation folder in the python script.
   """

   #open the file
   f = open(file_in, "rt")
   data = f.read()
   #replace all occurrences of the required string
   data = data.replace(text_to_replace, replace_by)
   #close the input file
   f.close()
   #open the input file in write mode
   f = open(file_in, "wt")
   #overrite the input file with the resulting data
   f.write(data)
   #close the file
   f.close()


def make_alias(installation_folder, master):
   """
   This will add alliases at the bash_profile file or the bashrc
   """

   home = os.environ['HOME']

   func = '\n# Added by %s installer\n%s() {\npython "%s/python_codes/%s.py" "$@"\n}\nexport -f %s\n'''%(master, master.lower(), installation_folder, master, master.lower())

   if os.path.isfile(home+'/.bash_profile'):
      bash_file = home+'/.bash_profile'
   elif os.path.isfile(home+'/.bashrc'):
      bash_file = home+'/.bashrc'
 
   shutil.copy(bash_file, bash_file+'.bkp')

   blank_file = ''

   with open(bash_file, 'r') as file:
      data = file.read()
   
   if not func in data:
      f = open(bash_file, "a")
      f.write(func)
      f.close()


def move_files(source_dir, target_dir):
   """
   This routine moves files and directories.
   """

   file_names = os.listdir(source_dir)

   for file_name in file_names:
      shutil.move(os.path.join(source_dir, file_name)+'/', target_dir+file_name+'/')


def installation():
   """
   This routine will install all possible dependencies and codes.
   """ 

   master = 'GaiaHub'
   current_dir = os.getcwd()

   help =  '-----------------------------------------------------------------------------------\n'\
           'Welcome to the installation process of %s\n'\
           '-----------------------------------------------------------------------------------\n'\
           '- %s needs approximatelly 2.5 GB of space in your disk.\n'\
           '- The installation will also attempt to install several python packages.\n'\
           '- If you do not want to modify your current python enviroment \n'\
           '  we recommend creating a dedicated conda enviroment to install and run %s.\n'\
           '-----------------------------------------------------------------------------------\n'

   print('\n'+help%(master, master, master)+'\n')

   try:
      cont = str2bool( input('Do you wish to continue? (y,n): ') or 'y')
      print('\n')
   except:
      print('\nWARNING: Answer not understood!\n')
      print('\nINSTALLATION ABORTED!\n')
      sys.exit(1)

   if cont:
      print('By default, %s will be installed in the current directory:\n'%master)
      print(current_dir)
      installation_folder = input('\nPress enter to accept or introduce an alternative path if you wish.\n') or current_dir

      print('Installing Python dependencies...\n')
      install_python_dependencies(installation_folder+'/python_codes/python_dependencies.txt')

      if installation_folder != current_dir:
         shutil.move(current_dir, installation_folder, copy_function = shutil.copytree)
         os.chdir(installation_folder)

      # Add the installation folder
      replace_text(installation_folder+"/python_codes/%s.py"%master, "installation_path = ''", "installation_path = '%s'"%installation_folder)

      print('Downloading hst1pass...\n')
      get_hst1pass(installation_folder, repo = 'https://www.stsci.edu/~jayander')

      print('The installation needs to compile two Fortran routines.')
      print('By default, the command "gfortran" will be executed.')
      compiler = input('\nPress enter to accept or introduce an alternative Fortran compiler if you wish.\n') or "gfortran"

      print('\nCompiling Fortran modules...\n')

      try:
         compile_fortran(compiler, installation_folder, raise_exception = False)
      except:
         print('WARNING: Something went wrong when compiling the Fortran libraries using "%s".'%compiler)
         print('This installation was tested on GNU Fortran (GCC) (version 8.2.0).')
         print('Visit "https://gcc.gnu.org/wiki/GFortranBinaries" for more information.')
         print('\nINSTALLATION ABORTED!\n')
         sys.exit(1)

      get_psfs_gd_libraries(installation_folder, repo = 'https://www.stsci.edu/~jayander', libs = ['STDPSFs', 'STDGDCs'], filters = ['F555W','F606W','F775W','F814W','F850LP'], instrument = ['ACSWFC', 'ACSHRC', 'WFC3UV'])
      
      print('\n\nThe installation can set an alias in your .bash_profile or .bashrc file for %s.'%master)
      print('This would allow you to run %s from anywere in your computer.\n'%master)
      
      try:
         alias = str2bool(input('Would you like to create an alias for %s? (y,n): '%master) or 'y')
      except:
         print('WARNING: Answer not understood!\n')
         alias = False

      if alias:
            make_alias(installation_folder, master)
            print('You can now run %s from anywere in your computer by typing "%s" in the terminal.\n'%(master, master.lower()))
            print('Try "%s --help" to learn how to execute it.\n'%master.lower())
            print('(You may need to open a new terminal)\n')

      else:
            print('No modification was made to .bash_profile.\n')
            print('To run %s, you will have to execute the python script located here:\n'%master)
            print(installation_folder+'/python_codes/'+master+'.py')

      print('DONE!\n')

if __name__ == '__main__':
    installation()
    sys.exit(0)
