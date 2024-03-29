import ovito
from ovito.io import import_file
from ovito.modifiers import *
from ovito.data import *
import glob as glob
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import tarfile

from clusterOVITO.basic.io import *
from clusterOVITO.basic.xyz_to_FEFF import *
from clusterOVITO.ovito_tools import *  
from ovito.modifiers import ReplicateModifier


from clusterOVITO.basic.xyz_to_FEFF import *


from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def make_rdf_feff(distances, rmeshPrime):
    # Use np.histogram to directly compute the count within each bin
    counts, _ = np.histogram(distances, bins=np.append(rmeshPrime, rmeshPrime[-1]+(rmeshPrime[1]-rmeshPrime[0])))
    # Normalize the counts by the width of the bins to get the density
    rdf_normalized = counts / (rmeshPrime[1] - rmeshPrime[0])
    return rdf_normalized



test = {"Pt":1, "C":2, "O":3}
class MD_EXAFS_Binary_Pipeline_General:
    def __init__(self, pipeline, file_params, custom_analysis_func=None):
        self.pipeline = pipeline
        self.framerange = file_params["framerange"]
        self.atoms = file_params["atoms"]
        self.cutoff = file_params["cutoff"]
        self.custom_analysis_func = custom_analysis_func
        self.b_line_table = []

    def run(self):
        for frame in self.framerange:
            self.frame = frame
            data = self.pipeline.compute(frame)
            finder = CutoffNeighborFinder(self.cutoff, data)

            atomic_position_index_list = self.get_atom_indices(data)
            all_positions = data.particles.positions[...]

            if self.custom_analysis_func:
            
                custom_result = self.custom_analysis_func(
                    atomic_position_index_list, finder
                    )
                self.interesting_results = []
                # add metal first
                self.interesting_results.append(all_positions[atomic_position_index_list[0]])
                # add other atoms
                for result in custom_result:
                    self.interesting_results.append(all_positions[result])
            # just add metal
            else:
                self.interesting_results = [all_positions[atomic_position_index_list[0]]]

            self.add_sep_line_start()

            self.make_files()

        self.finish()

    def get_atom_indices(self, data):

        atomic_position_index_list = []
        for atom in self.atoms.values():
            atomic_position_index_list.append(np.where(data.particles['Particle Type'].array == atom)[0])

        return atomic_position_index_list
    

    def make_lattice(self, n_atoms, n_pot, atom_coords):
            n_pots = np.full(n_atoms, n_pot).reshape(-1, 1)
            lattice = np.concatenate((atom_coords, n_pots), axis=1)
            return lattice

    def make_files(self):
        
        n_atoms = [atom.shape[0] for atom in self.interesting_results]
        
        atom_pots = [pot for pot in self.atoms.values()]

        lattices = [self.make_lattice(n_atom, n_pot, atom_coords) for n_atom, n_pot, atom_coords in zip(n_atoms, atom_pots, self.interesting_results)]

        b_line = np.concatenate(lattices, axis=0)
        b_line = np.around(b_line, decimals=4)
        self.b_line_table.append(b_line)


    def add_sep_line_start(self):
        self.b_line_table.append(np.array([[0,0,0,0]]))

    def add_sep_line_end(self):
        self.b_line_table.append(np.array([1,1,1,1]))

    def finish(self):
        self.data = np.concatenate(self.b_line_table, axis=0)


def find_adsorbed_CO(atomic_position_index_list, finder):
        ptatom_index, catom_index, oatom_index = atomic_position_index_list
        interesting_c, interesting_pt, interesting_o = [], [], []
        for i in catom_index:
            neigh = [neigh.index for neigh in finder.find(i)]
            if len(neigh) > 0:
                n_count = 0
                for l in neigh:
                    if l in ptatom_index:
                        n_count += 1
                        interesting_pt.append(l)
                if n_count > 0:
                    interesting_c.append(i)

        interesting_c = np.unique(np.array(interesting_c))
        interesting_pt = np.unique(np.array(interesting_pt))

        for i in interesting_c:
            neigh = [neigh.index for neigh in finder.find(i)]
            if len(neigh) > 0:
                for l in neigh:
                    if l in oatom_index:
                        interesting_o.append(l)

        interesting_o = np.unique(np.array(interesting_o))

        return interesting_c, interesting_o


test = {"Pt":1, "C":2, "O":3}
class MD_EXAFS_Analysis_Pipeline_PtCO:
    def __init__(self, pipeline, file_params, custom_analysis_func=None):
        self.pipeline = pipeline
        self.framerange = file_params["framerange"]
        self.atoms = file_params["atoms"]
        self.cutoff = file_params["cutoff"]
        self.custom_analysis_func = custom_analysis_func
        self.b_line_table = []
        self.n_atoms_list = []

    def run(self):
        for frame in self.framerange:
            self.frame = frame
            data = self.pipeline.compute(frame)
            finder = CutoffNeighborFinder(self.cutoff, data)

            atomic_position_index_list = self.get_atom_indices(data)
            all_positions = data.particles.positions[...]

            if self.custom_analysis_func:
            
                custom_result = self.custom_analysis_func(
                    atomic_position_index_list, finder
                    )
                self.interesting_results = []
                # add metal first
                self.interesting_results.append(all_positions[atomic_position_index_list[0]])
                # add other atoms
                for result in custom_result:
                    self.interesting_results.append(all_positions[result])
            # just add metal
            else:
                self.interesting_results = [all_positions[atomic_position_index_list[0]]]


            self.make_output()

        self.finish()

    def get_atom_indices(self, data):

        atomic_position_index_list = []
        for atom in self.atoms.values():
            atomic_position_index_list.append(np.where(data.particles['Particle Type'].array == atom)[0])

        return atomic_position_index_list
    

    def make_lattice(self, n_atoms, n_pot, atom_coords):
            n_pots = np.full(n_atoms, n_pot).reshape(-1, 1)
            lattice = np.concatenate((atom_coords, n_pots), axis=1)
            return lattice

    def make_output(self):
        
        n_atoms = [atom.shape[0] for atom in self.interesting_results]
        
        atom_pots = [pot for pot in self.atoms.values()]

        lattices = [self.make_lattice(n_atom, n_pot, atom_coords) for n_atom, n_pot, atom_coords in zip(n_atoms, atom_pots, self.interesting_results)]

        b_line = np.concatenate(lattices, axis=0)
        b_line = np.around(b_line, decimals=4)
        self.b_line_table.append(b_line)
        self.n_atoms_list.append(n_atoms)

    def finish(self):
        #self.data = np.concatenate(self.b_line_table, axis=0)
        self.data = self.b_line_table

def find_adsorbed_CO(atomic_position_index_list, finder):
        ptatom_index, catom_index, oatom_index = atomic_position_index_list
        interesting_c, interesting_pt, interesting_o = [], [], []
        for i in catom_index:
            neigh = [neigh.index for neigh in finder.find(i)]
            if len(neigh) > 0:
                n_count = 0
                for l in neigh:
                    if l in ptatom_index:
                        n_count += 1
                        interesting_pt.append(l)
                if n_count > 0:
                    interesting_c.append(i)

        interesting_c = np.unique(np.array(interesting_c))
        interesting_pt = np.unique(np.array(interesting_pt))

        for i in interesting_c:
            neigh = [neigh.index for neigh in finder.find(i)]
            if len(neigh) > 0:
                for l in neigh:
                    if l in oatom_index:
                        interesting_o.append(l)

        interesting_o = np.unique(np.array(interesting_o))

        return interesting_c, interesting_o


def find_atom_pairs_distances(array, print_indices=False):
    # Isolate atoms of type 2 and 3, keeping track of their original indices
    indices_type_2 = np.where(array[:, 3] == 2)[0]
    indices_type_3 = np.where(array[:, 3] == 3)[0]
    atoms_type_2 = array[indices_type_2, :3]  # x, y, z coordinates
    atoms_type_3 = array[indices_type_3, :3]  # x, y, z coordinates
    
    # Calculate pairwise distances between atoms of type 2 and 3
    distances = cdist(atoms_type_2, atoms_type_3)
    
    # Use the Hungarian algorithm to find optimal pairing based on distances
    row_ind, col_ind = linear_sum_assignment(distances)
    
    # Calculate distances for the optimal pairs
    optimal_distances = distances[row_ind, col_ind]
    
    if print_indices:
        # Print indices of paired atoms in the original array
        for i, j in zip(row_ind, col_ind):
            print(f"Atom of type 2 at index {indices_type_2[i]} is paired with Atom of type 3 at index {indices_type_3[j]}, Distance: {optimal_distances[i]}")
    
    return optimal_distances

# Example usage:
# array = np.array([...])  # Your Nx4 array
# distances = find_atom_pairs_distances(array, print_indices=True)
# print(distances

def find_atom_pairs_with_indices(array):
    indices_type_2 = np.where(array[:, 3] == 2)[0]
    indices_type_3 = np.where(array[:, 3] == 3)[0]
    atoms_type_2 = array[indices_type_2, :3]
    atoms_type_3 = array[indices_type_3, :3]
    
    distances = cdist(atoms_type_2, atoms_type_3)
    row_ind, col_ind = linear_sum_assignment(distances)
    optimal_distances = distances[row_ind, col_ind]

    # Return paired indices and their distances
    paired_indices = [(indices_type_2[i], indices_type_3[j], optimal_distances[i]) for i, j in zip(row_ind, col_ind)]
    return paired_indices

# Example usage:
# array = np.array([...])  # Your Nx4 array
# pairs = find_atom_pairs_with_indices(array)
# for pair in pairs:
#     print(f"Pair: {pair[0]} (Type 2) and {pair[1]} (Type 3), Distance: {pair[2]}")


def verify_unique_pairs(pairs):
    # Extract atom indices from pairs
    type_2_indices = [pair[0] for pair in pairs]
    type_3_indices = [pair[1] for pair in pairs]
    
    # Check for uniqueness
    unique_type_2 = len(type_2_indices) == len(set(type_2_indices))
    unique_type_3 = len(type_3_indices) == len(set(type_3_indices))
    
    if unique_type_2 and unique_type_3:
        print("All pairs are unique.")
        return True
    else:
        print("There are non-unique pairs.")
        return False

# Example usage:
# unique = verify_unique_pairs(pairs)
    
def classify_and_find_distances(array, cutoff_distance=3.0):
    # Isolate atoms of each type and their indices
    indices_type_1 = np.where(array[:, 3] == 1)[0]
    atoms_type_1 = array[indices_type_1, :3]
    indices_type_2 = np.where(array[:, 3] == 2)[0]
    atoms_type_2 = array[indices_type_2, :3]
    indices_type_3 = np.where(array[:, 3] == 3)[0]
    atoms_type_3 = array[indices_type_3, :3]
    
    # Calculate distances for bonding classification (type 1 and type 2 atoms)
    distances_1_2 = cdist(atoms_type_2, atoms_type_1)
    bonded_counts = np.sum(distances_1_2 < cutoff_distance, axis=1)
    
    # Calculate distances for pairing (type 2 and type 3 atoms)
    distances_2_3 = cdist(atoms_type_2, atoms_type_3)
    row_ind, col_ind = linear_sum_assignment(distances_2_3)
    paired_distances = distances_2_3[row_ind, col_ind]
    
    # Compile results into a dictionary
    results = {}
    for i, idx2 in enumerate(indices_type_2[row_ind]):
        idx3 = indices_type_3[col_ind[i]]
        classification = bonded_counts[i]
        pair_distance = paired_distances[i]
        results[idx2] = (classification, pair_distance, idx3)
    
    return results

# Example usage:
# array = np.array([...])  # Your Nx4 array
# results = classify_and_find_distances(array)
# for idx2, (classification, distance, idx3) in results.items():
#     print(f"Type 2 atom {idx2} (classified with {classification} type 1 bonds) is paired with Type 3 atom {idx3}, Distance: {distance}")

def classify_and_find_distances_array(array, cutoff_distance=3.0):
    # Isolate atoms of each type and their indices
    indices_type_1 = np.where(array[:, 3] == 1)[0]
    atoms_type_1 = array[indices_type_1, :3]
    indices_type_2 = np.where(array[:, 3] == 2)[0]
    atoms_type_2 = array[indices_type_2, :3]
    indices_type_3 = np.where(array[:, 3] == 3)[0]
    atoms_type_3 = array[indices_type_3, :3]
    
    # Calculate distances for bonding classification (type 1 and type 2 atoms)
    distances_1_2 = cdist(atoms_type_2, atoms_type_1)
    bonded_counts = np.sum(distances_1_2 < cutoff_distance, axis=1)
    
    # Calculate distances for pairing (type 2 and type 3 atoms)
    distances_2_3 = cdist(atoms_type_2, atoms_type_3)
    row_ind, col_ind = linear_sum_assignment(distances_2_3)
    paired_distances = distances_2_3[row_ind, col_ind]
    
    # Compile results into an array
    results_array = np.zeros((len(row_ind), 4))
    for i, idx2 in enumerate(indices_type_2[row_ind]):
        classification = bonded_counts[i]
        distance = paired_distances[i]
        idx3 = indices_type_3[col_ind[i]]
        results_array[i] = [idx2, classification, distance, idx3]
    
    return results_array

# Example usage:
# array = np.array([...])  # Your Nx4 array
# results_array = classify_and_find_distances_array(array)
# print(results_array)