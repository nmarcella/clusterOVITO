import numpy as np

mass_e = 9.1e-31;
hbar = 6.626e-34/(2*np.pi);

def intpol(data, energymesh):
    """
    data format is [[energies1, mus1]...[energy_n, mus_n]]
    """
    return np.interp(energymesh, xp=data[0:,0], fp=data[0:,1])

# s is S02
def correct1(data, de, s, kmesh):
    z = np.asarray(list(map(lambda n: [np.sqrt(n[0]**2+(2*mass_e*de*1.6e-19/hbar**2)*10e-20),n[1]/s], data)))
    d = z[[np.imag(0)==0 for s in z]]
    ipol = intpol(d,
                 kmesh)
    return np.asarray([kmesh,ipol]).transpose()

def k2(data):
    k, m = data.transpose()
    return np.asarray([k,k*k*m]).transpose()