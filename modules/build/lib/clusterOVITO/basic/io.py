
import numpy as np
def read_lines(filename):

    with open(filename, "rt") as fobject:
                f = fobject.readlines()
                lines_str = []

                for line in f:
                    lines_str.append(line.replace("\n", "").replace("\t", " "))
    return lines_str

def format_neg(number):
    """
    makes the columns in the feff.inp file look nice
    """
    if number < 0:
        return format(number, '.5f')
    else: 
        return format(number, '.6f')
    


"""
Reading EXAFS data files
"""
def get_temp_from_dir(file_path):
    return int(file_path.split("\\")[-1].split('_')[-2].replace('K',''))
def get_bulk_files(files_path):
    return glob.glob(files_path+"\\*\\*\\*\\*xmu.dat")
def read_chik2(file_path):
    return np.asarray([[float(s) for s in l.split()] for l in [l for l in read_lines(file_path) if l.split()[0] != '#'][1:]])[:,[0,2]]
def read_dat(file_path):
    return np.asarray([[float(s) for s in l.split()] for l in [l for l in read_lines(file_path) if l.split()[0] != '#']])[:,[0,1]]
def get_temp_from_exp(file_path):
    return file_path.split('\\')[-1].replace('.dat','')