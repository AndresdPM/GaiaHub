# GaiaHub
GaiaHub is a Python/Fortran tool that computes proper motions combining data from Gaia and the Hubble Space Telescope.

## License and Referencing
This code is released under a BSD 2-clause license.

## Features

GaiaHub includes lots of useful features:

* Search of objects based on names.
* Automatic screening out of poorly measured stars.
* Interactive and automatic selection of member stars.
* Statistics about the systemic proper motions of the object.
* Automatic generation of plots.

## Installation

At the moment, GaiaHub is offered as a python program that runs from your terminal. To install it, please clone this repository and run the installation script located inside the cloned folder. The installation will also attempt to install several python packages. If you do not want to modify your current python enviroment, we recommend creating a dedicated conda enviroment to install and run the program.

$ ./git clone https://github.com/AndresdPM/GaiaHub.git
$ ./python installation.py

Once the installation is completed, the user can run GaiaHub in any terminal as:

$ ./GaiaHub --name "Sculptor dSph"

Please read the help to know more about GaiaHub options.

$ ./GaiaHub --help

