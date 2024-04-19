class MD_EXAFS_Analysis_Pipeline_PtCO_v2:
    def __init__(self, pipeline, file_params, custom_analysis_func=None):
        self.pipeline = pipeline
        self.framerange = file_params["framerange"]
        self.atoms = file_params["atoms"]
        self.cutoff = file_params["cutoff"]
        self.custom_analysis_func = custom_analysis_func
        self.b_line_table = []
        self.n_atoms_list = []
        self.surface_pt = [] # v2 adds the ability to classify Pt atom

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