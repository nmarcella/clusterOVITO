def write_feff_dir(feff_inp, directory):
    os.makedirs(directory, exist_ok=True)
    with open(directory + "feff.inp", "w") as f:
        f.write(feff_inp)
    

def write_feff_dir_from_xyz(
    xyz, directory, absorber=0, edge="K", title="test", xmu_path=None, feff_inp_path=None, feff_template=feff_template
):
    potentials, atoms = make_potential_atoms_from_xyz(xyz, absorber=absorber)
    feff_inp = feff_template.format(
        title=title, edge=edge, potentials=potentials, atoms=atoms
    )
    write_feff_dir(feff_inp, directory)

    if xmu_path is not None:
        os.makedirs(os.path.dirname(xmu_path), exist_ok=True)
        shutil.copy(directory + "xmu.dat", xmu_path)
        
    if feff_inp_path is not None:
        os.makedirs(os.path.dirname(feff_inp_path), exist_ok=True)
        shutil.copy(directory + "feff.inp", feff_inp_path)

def create_tar_gz_of_directory(directory_path, output_filename, root_dir_name):
    """
    Create a tar.gz archive of the specified directory.

    :param directory_path: Path of the directory to be archived.
    :param output_filename: Name of the output tar.gz file.
    :param root_dir_name: Name of the root directory in the archive.
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                arcname = os.path.join(root_dir_name, os.path.relpath(filepath, directory_path))
                tar.add(filepath, arcname=arcname)

def create_slurm_scripts(dir_list, script_prefix, file_path):
    """
    Create SLURM batch scripts from a list of directories.

    :param dir_list: List of directories to process
    :param script_prefix: Prefix for the output SLURM script files
    """
    for block_num, i in enumerate(range(0, len(dir_list), 48), start=1):
        block = dir_list[i:i + 48]
        script_name = f"{script_prefix}_block{block_num}.sh"

        with open(file_path+script_name, 'w') as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --account=cfn310033\n")
            file.write("#SBATCH --nodes=1\n")
            file.write("#SBATCH --partition=cfn\n")
            file.write("#SBATCH --job-name=feff\n")
            file.write("#SBATCH --ntasks=48\n")  # Request 48 tasks
            file.write("#SBATCH --cpus-per-task=1\n")  # 1 CPU per task
            file.write("#SBATCH --time=01:00:00\n")  # 24 hour time limit
            file.write("#SBATCH --mail-user=nmarcella@bnl.gov\n")
            file.write("#SBATCH --mail-type=ALL\n")
            file.write("\n")

            for dir in block:
                dir_modified = dir.replace('/mnt/sdcc/', '/')
                file.write(f"srun --exclusive -n 1 --cpus-per-task=1 --mem=4G --time=00:10:00 --chdir={dir_modified} ~/feff_code/feff.sh &\n")
            file.write("wait\n")

    print(f"Created {block_num} SLURM script(s) with prefix '{script_prefix}'.")



def move_file(source_path, destination_path):
    """
    Move a file from source_path to destination_path.

    :param source_path: The path of the file to be moved
    :param destination_path: The path where the file will be moved
    """
    shutil.move(source_path, destination_path)


# make a command to run the scripts. sbatch run_block1.sh; sbatch run_bl;ock2.sh; etc
def make_command_to_run_scripts(script_prefix, start_script, end_script):
    """
    Create a command to run a series of SLURM scripts within a specified range.

    :param script_prefix: Prefix for the SLURM script files
    :param start_script: The starting script number in the range
    :param end_script: The ending script number in the range (inclusive)
    :return: A string containing the command to run the specified range of SLURM scripts
    """
    return "; ".join([f"sbatch {script_prefix}_block{i}.sh" for i in range(start_script, end_script + 1)])



def generate_sbatch_command(script_dir, output_file):
    """
    Generate a single sbatch command with semicolons to submit all script files in a specified directory.

    :param script_dir: Directory containing the SLURM script files
    :param output_file: Output file to write the sbatch command
    """
    script_files = [f for f in os.listdir(script_dir) if f.endswith('.sh')]
    with open(output_file, 'w') as file:
        sbatch_commands = '; '.join(f"sbatch {os.path.join(script_dir, script)}" for script in script_files)
        file.write(sbatch_commands + '\n')

    print(f"Generated sbatch command in '{output_file}'.")
    
    
    
def get_variance(rdf, rmeshPrime=rmeshPrime, rrange=[1,3]):
    bins,counts = np.array([i for i in np.array([rmeshPrime, rdf]).T if rrange[0]<i[0]<rrange[1]]).T
    weighted_avg = np.average(bins, weights=counts)
    variance = np.average((bins - weighted_avg)**2, weights=counts)
    return variance



def feff_to_xyz(input_file_path, output_file_path):
    # Reading the input file
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # Initialize flags for recording POTENTIALS and ATOMS
    record_potentials = False
    record_atoms = False

    potentials = {}  # To store potential mappings
    atoms = []  # To store atom details

    for line in lines:
        # Extracting potential mappings
        if line.strip() == 'POTENTIALS':
            record_potentials = True
            continue

        if record_potentials:
            if line.strip() == 'ATOMS':
                record_potentials = False
                record_atoms = True
                continue

            parts = line.split()
            if len(parts) >= 2:
                pot = int(parts[0])
                atom_symbol = parts[2]
                potentials[pot] = atom_symbol

        # Extracting atom coordinates and types
        if record_atoms:
            if line.strip() == 'END':
                break

            parts = line.split()
            if len(parts) >= 4:
                x, y, z, pot = parts[:4]
                atom_symbol = potentials[int(pot)]
                atoms.append((atom_symbol, x, y, z))
                
                
def feff_to_rdf(input_file_path, partial_1, partial_2, return_distances = False):
    # Reading the input file
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # Initialize flags for recording POTENTIALS and ATOMS
    record_potentials = False
    record_atoms = False

    potentials = {}  # To store potential mappings
    atoms = []  # To store atom details
    xyz = [] # To store xyz coordinates

    for line in lines:
        # Extracting potential mappings
        if line.strip() == 'POTENTIALS':
            record_potentials = True
            continue

        if record_potentials:
            if line.strip() == 'ATOMS':
                record_potentials = False
                record_atoms = True
                continue

            parts = line.split()
            if len(parts) >= 2:
                pot = int(parts[0])
                atom_symbol = parts[2]
                potentials[pot] = atom_symbol

        # Extracting atom coordinates and types
        if record_atoms:
            if line.strip() == 'END':
                break

            parts = line.split()
            if len(parts) >= 4:
                x, y, z, pot = parts[:4]
                atom_symbol = potentials[int(pot)]
                atoms.append((atom_symbol, x, y, z))
                xyz.append((x, y, z))