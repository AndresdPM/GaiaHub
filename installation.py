# This program will install all the necessary files to run StellarTeam

import os
import sys

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


def get_psfs_gd_libraries(installation_folder, repo = 'https://www.stsci.edu/~jayander', libs = ['STDPSFs', 'STDGDCs'], filters = ['F555W','F606W','F775W','F814W','F850LP'], instrument = ['ACSWFC', 'ACSHRC', 'WFC3UV', 'WFC3IR']):
   """
   This routine finds the required PSF and geometric distorsion libraries.
   """
   from bs4 import BeautifulSoup
   import requests
   import itertools

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

   for ii, (file_url, file_output) in enumerate(zip(files_urls, output_files)):
      cli_progress_test(ii+1, len(files_urls))
      r = requests.get(file_url)
      with open(file_output, 'wb') as f:
         f.write(r.content)


def clone_repo(installation_folder, repo = 'https://github.com/AndresdPM/StellarTeam'):
   """
   This routine finds the codes in Github.
   """
   try:
      import git
   except:
      import subprocess
      subprocess.check_call([sys.executable, "-m", "pip", "install", 'GitPython'])
      import git

   git.Git(installation_folder).clone(repo)


def compile_fortran(installation_folder):
   """
   This routine will compile Fortran routines
   """ 
   import subprocess
   
   path = installation_folder+'/fortran_codes/'
   f_modules = ['%s%s'%(path, file) for file in os.listdir(path) if file.endswith('.F')]
   
   for input_file in f_modules:
      output_file = os.path.splitext(input_file)[0]+'.e'
      bashCommand = "gfortran %s -o %s"%(input_file, output_file)
      process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      output, error = process.communicate()


def remove_file(file_name):
   """
   This routine removes files
   """

   try:
      os.remove(file_name)
   except:
      pass


def remove_files(folder):
   import os, shutil
   for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
      try:
         if os.path.isfile(file_path) or os.path.islink(file_path):
               os.unlink(file_path)
         elif os.path.isdir(file_path):
               shutil.rmtree(file_path)
      except Exception as e:
         print('Failed to delete %s. Reason: %s\n' % (file_path, e))

   
def install_python_dependencies(packages):
   """
   This will install the required python dependencies
   """
   import subprocess
   
   subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", packages])
   
   print('\n')


def make_alias(installation_folder):
   """
   This will add alliases at the bash_profile file or the bashrc
   """
   
   home = os.environ['HOME']

   alias =  "\n# Added by StellarTeam installer\n"\
            "alias StellarTeam='python %s'\n"%(installation_folder+'/python_codes/StellarTeam.py')

   if os.path.isfile(home+'/.bash_profile'):
      f = open(home+'/.bash_profile', "a")
      f.write(alias)
      f.close()
   elif os.path.isfile(home+'/.bashrc'):
      f = open(home+'/.bashrc', "a")
      f.write(alias)
      f.close()


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


def add_path(installation_folder):
   """
   This routine hardcodes the installation folder in the python script.
   """

   #open the file
   fin = open(installation_folder+"/python_codes/StellarTeam.py", "rt")
   data = fin.read()
   #replace all occurrences of the required string
   data = data.replace("installation_path = '/Users/user/StellarTeam'", "installation_path = '%s'"%installation_folder)
   #close the input file
   fin.close()
   #open the input file in write mode
   fin = open(installation_folder+"/python_codes/StellarTeam.py", "wt")
   #overrite the input file with the resulting data
   fin.write(data)
   #close the file
   fin.close()

   
def installation():
   """
   This routine will install all possible dependencies and codes.
   """ 
   
   print('---------------------------------------------------')
   print('Welcome to the installation process of StellarTeam!')
   print('---------------------------------------------------')
   
   master = 'StellarTeam'
   default_dir = os.environ['HOME']+'/'+master
   
   print('StellarTeam needs approximatelly 2.5 GB of space in your disk.\n')

   print('By default, StellarTeam will be installed in the home directory:\n')
   print(default_dir)
   print('\nThis installation will remove any previous installation in such folder.\n')
   installation_folder = input('Press enter to accept or introduce an alternative path if you wish.\n') or default_dir
   
   try:
      remove_files(installation_folder)
   except:
      pass
   
   create_dir(installation_folder)

   clone_repo(os.path.split(installation_folder)[0], repo = 'https://github.com/AndresdPM/%s'%master)

   print('Succesfully cloned the repository at %s\n'%installation_folder)

   # Add the installation folder
   add_path(installation_folder)

   print('Installing Python dependencies...\n')
   
   install_python_dependencies(default_dir+'/python_codes/python_dependencies.txt')

   get_psfs_gd_libraries(installation_folder, repo = 'https://www.stsci.edu/~jayander', libs = ['STDPSFs', 'STDGDCs'], filters = ['F555W','F606W','F775W','F814W','F850LP'], instrument = ['ACSWFC', 'ACSHRC', 'WFC3UV'])

   print('\n\nCompiling Fortran modules...\n')

   compile_fortran(installation_folder)

   alias = input('Would you like to include an alias in the .bash_profile file? (y,n) ')

   try:
      alias = str2bool(alias)
   except:
      print('\nExiting now. No modification was made to .bash_profile \n')
      sys.exit(1)

   if alias:
      make_alias(installation_folder)

   print('\nDone!!!')


if __name__ == '__main__':
    installation()
    sys.exit(0)


"""
Andres del Pino Molina
"""
