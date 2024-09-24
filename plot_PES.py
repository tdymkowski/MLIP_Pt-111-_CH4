#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:07:28 2023

@author: floris

Modded by Nick on 26-03-2024
"""

import numpy as np
from scipy.constants import physical_constants as phc
from methane import set_CH4_position, align_CH4_to_surface_normal

from pyace import PyACECalculator

from ase.build import fcc111, molecule, add_adsorbate
from ase.io import write

import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# %% import surface from file
atmas = phc['atomic mass constant'][0]
eV2J  = phc['electron volt-joule relationship'][0]
Bc    = phc['Boltzmann constant'][0]
hbar  = phc['Planck constant'][0] / (2*np.pi)
H2J   = phc['Hartree energy'][0]
A2B   = 1e-10 / phc['Bohr radius'][0]
H2eV  = phc['Hartree energy in eV'][0]
eV2kJmol = phc['electron volt-joule relationship'][0]*phc['Avogadro constant'][0]/1000.

#### Cell #######################################
# Pt(111) - 500 K
Cell = np.array([[8.5609344674269,    0.0000000000000000,    0.0000000000000000],
     [-4.2804672337134,    7.4139867289255,    0.0000000000000000],
     [0.0000000000000000,    0.0000000000000000,   22.3408205496026]])
#################################################

#### Surface ####################################
atoms = fcc111(symbol='Pt', size=(3, 3, 5), a=4.02016156764*1.003855975, vacuum=6.5, orthogonal=False)

Surface = np.array(
[[  0.0000000000000,  0.0000000000000,  22.3408205496026  ],
[ -1.4268224112378,  2.4713289096418,  22.3408205496026  ],
[ -2.8536448224756,  4.9426578192836,  22.3408205496026  ],
[  2.8536448224756,  0.0000000000000,  22.3408205496026  ],
[  1.4268224112378,  2.4713289096418,  22.3408205496026  ],
[  0.0000000000000,  4.9426578192836,  22.3408205496026  ],
[  5.7072896449512,  0.0000000000000,  22.3408205496026  ],
[  4.2804672337134,  2.4713289096418,  22.3408205496026  ],
[  2.8536448224756,  4.9426578192836,  22.3408205496026  ],
[  1.4268224112378,  0.8237763032139,  19.9848525009637  ],
[  0.0000000000000,  3.2951052128558,  19.9848525009637  ],
[ -1.4268224112378,  5.7664341224976,  19.9848525009637  ],
[  4.2804672337134,  0.8237763032139,  19.9848525009637  ],
[  2.8536448224756,  3.2951052128558,  19.9848525009637  ],
[  1.4268224112378,  5.7664341224976,  19.9848525009637  ],
[  7.1341120561890,  0.8237763032139,  19.9848525009637  ],
[  5.7072896449512,  3.2951052128558,  19.9848525009637  ],
[  4.2804672337134,  5.7664341224976,  19.9848525009637  ],
[  2.8536448224756,  1.6475526064279,  17.6703744522008  ],
[  1.4268224112378,  4.1188815160697,  17.6703744522008  ],
[  0.0000000000000,  6.5902104257115,  17.6703744522008  ],
[  5.7072896449512,  1.6475526064279,  17.6703744522008  ],
[  4.2804672337134,  4.1188815160697,  17.6703744522008  ],
[  2.8536448224756,  6.5902104257115,  17.6703744522008  ],
[  8.5609344674269,  1.6475526064279,  17.6703744522008  ],
[  7.1341120561890,  4.1188815160697,  17.6703744522008  ],
[  5.7072896449512,  6.5902104257115,  17.6703744522008  ],
[  0.0000000000000,  0.0000000000000,  15.3559354333583  ],
[ -1.4268224112378,  2.4713289096418,  15.3559354333583  ],
[ -2.8536448224756,  4.9426578192836,  15.3559354333583  ],
[  2.8536448224756,  0.0000000000000,  15.3559354333583  ],
[  1.4268224112378,  2.4713289096418,  15.3559354333583  ],
[  0.0000000000000,  4.9426578192836,  15.3559354333583  ],
[  5.7072896449512,  0.0000000000000,  15.3559354333583  ],
[  4.2804672337134,  2.4713289096418,  15.3559354333583  ],
[  2.8536448224756,  4.9426578192836,  15.3559354333583  ],
[  1.4268224112378,  0.8237763032139,  13.0000000000000  ],
[  0.0000000000000,  3.2951052128558,  13.0000000000000  ],
[ -1.4268224112378,  5.7664341224976,  13.0000000000000  ],
[  4.2804672337134,  0.8237763032139,  13.0000000000000  ],
[  2.8536448224756,  3.2951052128558,  13.0000000000000  ],
[  1.4268224112378,  5.7664341224976,  13.0000000000000  ],
[  7.1341120561890,  0.8237763032139,  13.0000000000000  ],
[  5.7072896449512,  3.2951052128558,  13.0000000000000  ],
[  4.2804672337134,  5.7664341224976,  13.0000000000000  ]])

atoms.positions = Surface
atoms.set_cell( Cell )
atoms.set_pbc( [True, True, True])
unit_cell = atoms.cell / 3
z_toplayer = atoms.positions[-1][-1]

#################################################
#### Molecule ###################################

methane = molecule('CH4')
methane.set_masses([12., 2.014, 2.014, 2.014, 1.008])	# CHD3

add_adsorbate(atoms, methane, 2, 'ontop', offset=1)
atoms.wrap()
file_name = 'pt111_ch4.xyz'
write(file_name, atoms, format='xyz')
#################################################


# %% define calculator
ace_calc = PyACECalculator('output_potential.yaml')

# %% compute PES

N = 50  # grid spacing

r_range, dr = np.linspace(1., 2., N, retstep=True)
z_range, dz = np.linspace(2., 3., N, retstep=True)

r_grid, z_grid = np.meshgrid(r_range, z_range)
E_grid_ace = np.zeros_like(r_grid)

print('Calculating PES...')

for i, r in enumerate(r_range):
    for j, z in enumerate(z_range):
        set_CH4_position(atoms, r=r, z=z)
        atoms.calc = ace_calc
        E_grid_ace[j, i] = atoms.get_potential_energy()
        write(file_name, atoms, format='xyz', append=True)


# GASPHASE
print('Calculating reference energy...')
set_CH4_position(atoms, r=1.09, z=6.5)
atoms.calc = ace_calc
write(file_name, atoms, format='xyz', append=True)
E_min_ace = atoms.get_potential_energy()

# TRANSITION STATE J. Chem. Phys. 130, 054701
print('Calculating transition state energy...')
align_CH4_to_surface_normal(atoms, theta=132.60)
set_CH4_position(atoms, r=1.493, z=2.193)
E_TS_ace = atoms.get_potential_energy()
write(file_name, atoms, format='xyz', append=True)

E_grid_ace -= E_min_ace
E_TS_ace -= E_min_ace


# %% plot PES
print('Plotting PES...')
levels = np.arange(0.1, 4, 0.05)

# interpolate
n = 1
r_zoom = zoom(r_grid, n, order=1, mode='nearest')
z_zoom = zoom(z_grid, n, order=1, mode='nearest')
E_zoom_ace = zoom(E_grid_ace, n)
# Compute curvature
# Er, Ez = np.gradient(E_zoom, dr/n, dz/n)
# Err, Erz = np.gradient(Er, dr/n, dz/n)
# Ezr, Ezz = np.gradient(Ez, dr/n, dz/n)
# K_zoom = (Err * Ezz - (Erz ** 2)) /  (1 + (Er ** 2) + (Ez **2)) ** 2


fig, ax = plt.subplots()

# k_max = 188
# ax.pcolormesh(r_zoom, z_zoom, K_zoom, cmap='bwr', vmin=-k_max, vmax=k_max)

# plot PES level lines
lines = ax.contour(r_zoom, z_zoom, E_zoom_ace, levels=levels)

plt.colorbar(lines, label='Potential energy (eV)')

# plot transition states point
#ax.scatter([1.738], [1.312], c='r', zorder=10, marker='+', s=40)

ax.set_xlabel(r'separation $r$ ($\mathrm{\AA}$)')
ax.set_ylabel(r'height $Z$ ($\mathrm{\AA}$)')

#plt.savefig('./PES.pdf')
plt.show()
