![gaiaHub_logo_crop](https://github.com/AndresdPM/GaiaHub/assets/63738265/baae8350-aa15-4aec-8325-fa8d6f7d965f)

Credit: Zili Shen, Astrobites. ESA.

# GaiaHub

GaiaHub is a Python/Fortran tool that computes proper motions combining data from Gaia and the Hubble Space Telescope.

## License and Referencing
GaiaHub and its documentation are released under a BSD 2-clause license. GaiaHub is freely available and you can freely modify, extend, and improve the GaiaHub source code. However, if you use it for published research, you are requested to cite [del Pino et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...933...76D/abstract)  where the method is described.


## Features

GaiaHub includes lots of useful features:

* Search of objects based on names.
* Automatic screening out of poorly measured stars.
* Interactive and automatic selection of member stars.
* Statistics about the systemic proper motions of the object.
* Automatic generation of figures.

## Installation

At the moment, GaiaHub is offered as a python code that runs locally in your machine. To install it, please clone this repository and run the installation script located inside the cloned folder. The installation will check your current python environment and attempt to install or update the required python packages to run GaiaHub. We recommend creating a dedicated conda enviroment to install and run GaiaHub if you do not want to modify your current python environment. More information can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). To proceed with the installation, open a terminal in the directory where you would like to install GaiaHub and type:

$ git clone https://github.com/AndresdPM/GaiaHub.git

$ cd GaiaHub

$ python install_GaiaHub.py

Please follow the instructions in the screen.

## Running GaiaHub

Once the installation is completed, the user can run GaiaHub from the terminal as:

$ gaiahub [options]

For example, to compute the proper motions of NGC 5053 using only member stars, GaiaHub should be called as:

$ gaiahub --name "NGC 5053" --use_members

In this example, the results produced by GaiaHub will be stored in a subfolder called "NGC_5053".

To know more about all GaiaHub options:

$ gaiaHub --help

For more examples please see [del Pino et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...933...76D/abstract) and the Documentation file (Documentation_v1.pdf).

## Notes on Version 1.1.0

- Fixed bug: GaiaHub will now not download HAP images.
- New statistics added through the "---use_stat" option.
- New iterative sigma-clipped method to obtain the absolute reference frame.
