#! /home/tomasz/anaconda3/bin/python3

from ase import Atoms
import pandas as pd
from ase.io import write


def process_data_ase(file_path: str):
    """
    Extract data from a RuNNer file

    """

    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    energy = None
    lattice_vectors = []
    atom_positions = []
    atom_symbols = []
    forces = []
    energy_corrected = None
    reference_energy = 0

    for line in lines:
        split_line = line.split()

        if line.startswith("energy"):
            energy = float(split_line[1])
            energy_corrected = energy - reference_energy
            # Convert units
            energy, energy_corrected = energy*27.21, energy_corrected*27.21  # Hartree -> eV
        elif line.startswith("lattice"):
            lattice_vectors.append([float(x) * 0.529 for x in split_line[1:]]) # Bohr to Angs
        elif line.startswith("atom"):
            atom_position = [float(split_line[1]),
                             float(split_line[2]),
                             float(split_line[3])]
            # Convert units
            atom_position = [x*0.529 for x in atom_position]  # Bohr to Angs
            atom_symbol = split_line[4]
            atom_positions.append(atom_position)
            atom_symbols.append(atom_symbol)
            force = [float(split_line[-3]),
                     float(split_line[-2]),
                     float(split_line[-1])]
            # Convert units
            force = [x*51.42 for x in force]  # Ha/Bohr -> eV/A
            forces.append(force)
        elif line.startswith("end"):
            if energy is not None and lattice_vectors and atom_positions:
                atom_number = len(atom_symbols)
                energy_corrected_per_atom = energy_corrected / atom_number
                ase_atoms = Atoms(symbols=atom_symbols,
                                  positions=atom_positions,
                                  cell=lattice_vectors,
                                  pbc=True)
                row = [ase_atoms, energy, forces,
                       energy_corrected, energy_corrected_per_atom]
                data.append(row)
                write('runner.xyz', ase_atoms, format='xyz', append=True)
            energy, lattice_vectors, atom_positions, atom_symbols, forces = None, [], [], [], []
    df = pd.DataFrame(data, columns=["ase_atoms", "energy",
                                     "forces", "energy_corrected",
                                     "energy_corrected_per_atom"])
    return df


# Prepare data frame
df = process_data_ase('/home/tdymkowski/Dokumenty/master_project/new_input.data')
print(df)
print(f'Df shape: {df.shape}')
df_test = df.sample(n=250, random_state=42)
df_train = df.sample(n=1000, random_state=42)
# df_train = df.drop(index=df_test.index)
print(f'Df test shape: {df_test.shape}')
print(f'Df train shape: {df_train.shape}')

# Save data frame as pickle
# df_train.to_excel("pt_train_set.xlsx")
df_test.to_pickle("df_test.pckl.gzip", compression='gzip', protocol=4)
df_train.to_pickle("df_full.pckl.gzip", compression='gzip', protocol=4)
