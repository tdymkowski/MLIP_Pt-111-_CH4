#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:07:28 2023

@author: floris

Modded by Nick on 26-03-2024
"""

import numpy as np
from scipy.constants import physical_constants as phc

from ase import Atoms
from ase.build import fcc111, molecule, add_adsorbate
from ase.io import write


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
atoms.set_cell(Cell)
atoms.set_pbc([True, True, True])
z_toplayer = atoms.positions[-1,-1]
#write('surface.xyz', atoms, format='xyz')
#################################################
# Molecule ###################################

methane = molecule('CH4')
methane.set_masses([12., 2.014, 2.014, 2.014, 1.008])  # CHD3
add_adsorbate(atoms, methane, z_toplayer - 6.5, 'ontop', offset=1)
atoms.wrap()

file_name = 'pt111_ch4.xyz'
write(file_name, atoms, format='xyz')
#################################################


def rotate_vector(axis, vector, theta):
    theta = np.radians(theta)
    axis = axis / np.linalg.norm(axis)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    term1 = vector * cos_theta
    term2 = np.cross(axis, vector) * sin_theta
    term3 = axis * np.dot(axis, vector) * (1 - cos_theta)

    return term1 + term2 + term3


def center_of_mass(atoms, molecule_formula: str):
    symbols, positions, masses = [], [], []
    ziped = zip(atoms.symbols, atoms.positions, atoms.get_masses())
    for symbol, position, mass in ziped:
        if symbol in molecule_formula:
            symbols.append(symbol)
            positions.append(position)
            masses.append(mass)

    molecule = Atoms(symbols=symbols, positions=positions,
                     masses=masses)

    com = molecule.get_center_of_mass()
    return com


def align_CH4_to_surface_normal(atoms, theta: float):
    com = center_of_mass(atoms, 'CH4')
    methane = atoms.positions[-5:]
    # Rotate the molecule along Z-axis
    surface_normal = np.array([0, 0, 1])
    rCH = methane[-5] - methane[-4]
    rCH_unit = rCH / np.linalg.norm(rCH)

    cos_theta = np.dot(rCH_unit, surface_normal) / (np.linalg.norm(rCH_unit) * np.linalg.norm(surface_normal))
    current_angle = np.degrees(np.arccos(cos_theta))

    methane -= com

    rotation_angle = theta - current_angle
    rotation_axis = np.cross(rCH_unit, surface_normal)

    for i in range(len(methane)):
        R = rotate_vector(rotation_axis, methane[i], rotation_angle)
        methane[i] = R

    methane += com
    atoms.positions[-5:] = methane

    return


def set_CH4_position(atoms, r: float, z: float):
    methane = atoms.positions[-5:]
    top_surface = atoms.positions[-14:-5]

    # Set z component of CH4
    z_com = center_of_mass(atoms, 'CH4')[2]
    z_ = top_surface[-1, 2] - z
    methane[:, 2] += z_ - z_com # methane[-5, 2]

    atoms.positions[-5:] = methane

    # Set C-H bond length
    rCH = methane[-5] - methane[-4]
    dir_rCH = rCH / np.linalg.norm(rCH)

    new_H = methane[-5] - dir_rCH * r
    atoms.positions[-4] = new_H

    return


if __name__ == '__main__':
    theta = 132.60
    N = 50
    r_range, dr = np.linspace(1., 2., N, retstep=True)
    z_range, dz = np.linspace(1., 3., N, retstep=True)
    align_CH4_to_surface_normal(atoms, theta=theta)
    for i, r in enumerate(r_range):
        for j, z in enumerate(z_range):
            set_CH4_position(atoms, r=r, z=z)
            write(file_name, atoms, format='xyz', append=True)
