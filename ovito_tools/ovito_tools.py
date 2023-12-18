import numpy as np
import ovito
from ovito.io import import_file
from ovito.modifiers import *
import glob as glob
import numpy as np
import matplotlib.pyplot as plt
from ovito.data import *
from scipy.integrate import trapz
from fastdist import fastdist
import pyperclip

import numpy as np


def center_nanoparticle(positions, box_size, pt_atom_indices):
    # Calculate the center of mass of the nanoparticle
    center_of_mass = np.mean(positions[pt_atom_indices], axis=0)
    
    # Wrap the center of mass to the simulation box
    center_of_mass %= box_size
    
    # Shift the positions so that the center of mass is at the center of the simulation box
    shift_vector = box_size / 2 - center_of_mass
    new_positions = positions + shift_vector
    new_positions %= box_size
    
    return new_positions


def integrate_mono(rdf ,rmesh, rrange):
    x,y = np.asarray([l for l in np.asarray([rmesh, rdf]).transpose() if rrange[0]<l[0]<rrange[1]]).transpose()
    return trapz(y,x)

def make_rdf_feff(distances, rmeshPrime):
    digitized =np.digitize(distances
        , rmeshPrime)
    unique, counts = np.unique(digitized, return_counts=True)
    counter = [0]*len(rmeshPrime)
    for i in range(len(unique)):
        counter[unique[i]-1] = counts[i]/(rmeshPrime[1]-rmeshPrime[0])
    return np.asarray(counter)


class AnalysisPipeline:
    def __init__(self, pipeline, framerange):
        self.pipeline = pipeline
        self.framerange = framerange
        self.results = {
            "t_v_ave": {},
            "t_r_ave": {},
            "t_all_positions": {},
            "t_c_ads": {},
            "t_cn_ptpt": {},
            "t_frame_rdf": {},
        }

    def run(self):
        for frame in self.framerange:
            data = self.pipeline.compute(frame)
            finder = CutoffNeighborFinder(3, data)
            finder2 = CutoffNeighborFinder(3.4, data)

            ptatom_index, catom_index, oatom_index = self.get_atom_indices(data)
            all_positions = data.particles.positions[...]

            interesting_c, interesting_pt, interesting_o = self.find_interesting_atoms(
                catom_index, ptatom_index, oatom_index, finder
            )

            ptptcn, ptpt_R, ptpt_msrd = self.compute_ptpt_metrics(
                ptatom_index, finder2
            )

            c_ads = len(interesting_c)
            pt_atom_coords = all_positions[ptatom_index]

            box_size = data.cell.matrix.diagonal()[:3]

            rdf = self.frame_rdf(pt_atom_coords, box_size, ptatom_index)

            self.update_results(frame, ptpt_msrd, ptpt_R, pt_atom_coords, c_ads, np.mean(ptptcn), rdf)

        self.process_results()

    def get_atom_indices(self, data):
        ptatom_index = np.where(data.particles['Particle Type'].array == 1)[0]
        catom_index = np.where(data.particles['Particle Type'].array == 2)[0]
        oatom_index = np.where(data.particles['Particle Type'].array == 3)[0]
        return ptatom_index, catom_index, oatom_index

    def find_interesting_atoms(self, catom_index, ptatom_index, oatom_index, finder):
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

        return interesting_c, interesting_pt, interesting_o

    def compute_ptpt_metrics(self, ptatom_index, finder2):
        ptptcn, ptpt_R, ptpt_msrd = [], [], []
        for i in ptatom_index:
            neigh = [neigh.index for neigh in finder2.find(i)]
            rs = [neigh.distance for neigh in finder2.find(i)]
            distances = [[l, rs[i]] for i, l in enumerate(neigh) if l in ptatom_index]

            distances_unique = np.unique(np.array(distances), axis=0)
            distance = np.mean(distances_unique[:, 1], axis=0)
            var = np.var(distances_unique[:, 1], axis=0)

            ptpt_msrd.append(var)
            ptptcn.append(len(distances))
            ptpt_R.append(distance)
        ptptcn = np.array(ptptcn)
        ptpt_R = np.array(ptpt_R)
        ptpt_msrd = np.array(ptpt_msrd)
        ptpt_msrd = np.mean(ptpt_msrd)
        ptpt_R = np.mean(ptpt_R)
        cn_ptpt = np.mean(ptptcn)

        return ptptcn, ptpt_R, ptpt_msrd

    def frame_rdf(self, pt_atom_coords, cell, ptatom_index):
        centered_coords = center_nanoparticle(pt_atom_coords, cell, ptatom_index)
        num_atoms = len(centered_coords)
        distances = np.linalg.norm(centered_coords[:, None, :] - centered_coords[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)  # set diagonal to infinity to exclude self-distances
        distances = distances.flatten()
        distances = distances[distances != np.inf]  # remove self-distances
        distances = distances[distances < 6]  # remove distances greater than 6
        rdf = make_rdf_feff(distances)/num_atoms
        return rdf

    def update_results(self, frame, ptpt_msrd, ptpt_R, pt_atom_coords, c_ads, cn_ptpt, rdf):
        self.results["t_v_ave"][frame] = ptpt_msrd
        self.results["t_r_ave"][frame] = ptpt_R
        self.results["t_all_positions"][frame] = pt_atom_coords
        self.results["t_c_ads"][frame] = c_ads
        self.results["t_cn_ptpt"][frame] = cn_ptpt
        self.results["t_frame_rdf"][frame] = rdf

    def process_results(self):
        t_cns = np.array(list(self.results["t_cn_ptpt"].values()))
        t_all_msrd = np.array(list(self.results["t_v_ave"].values()))
        t_all_r = np.array(list(self.results["t_r_ave"].values()))
        t_all_c_ads = np.array(list(self.results["t_c_ads"].values()))
        t_all_rdf = np.array(list(self.results["t_frame_rdf"].values()))

        self.results["t_cn_ptpt"] = t_cns
        self.results["t_v_ave"] = t_all_msrd
        self.results["t_r_ave"] = t_all_r
        self.results["t_c_ads"] = t_all_c_ads
        self.results["t_frame_rdf"] = t_all_rdf





import numpy as np
import plotly.graph_objs as go

class TrajectoryPlot:
    def __init__(self, t_cns, t_all_msrd, t_all_r, t_all_c_ads, axis_ranges):
        self.t_cns = t_cns
        self.t_all_msrd = t_all_msrd
        self.t_all_r = t_all_r
        self.t_all_c_ads = t_all_c_ads
        self.axis_ranges = axis_ranges

    def plot(self):
        time = np.arange(0, 100999, 1000)
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=time, y=self.t_cns, yaxis="y1", name="CNs", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=time, y=self.t_all_msrd, yaxis="y2", name="MSRD", line=dict(color='red')))
        fig.add_trace(go.Scatter(x=time, y=self.t_all_r, yaxis="y3", name="R", line=dict(color='green')))
        fig.add_trace(go.Scatter(x=time, y=self.t_all_c_ads, yaxis="y4", name="CO-Ads", opacity=0.5, line=dict(color='black', dash='dash')))

        fig.update_layout(
            height=600,
            width=800,
            xaxis=dict(title="Frame", range=self.axis_ranges["x"], domain=[.12, 0.85], showgrid=False),
            yaxis=dict(title="Pt-Pt Coordination number", side="left", range=self.axis_ranges["y1"], titlefont=dict(color='blue', size=20), tickfont=dict(color='blue'), showgrid=False),
            yaxis2=dict(title="Pt-Pt MSRD", side="right", range=self.axis_ranges["y2"], overlaying="y", titlefont=dict(color='red', size=20), tickfont=dict(color='red'), showgrid=False),
            yaxis3=dict(title="Pt-Pt R", autoshift=True, range=self.axis_ranges["y3"], side='right', overlaying="y", anchor='free', titlefont=dict(color='green', size=20), tickfont=dict(color='green'), position=0.99, showgrid=False),
            yaxis4=dict(title="# of adsorbed CO", autoshift=True, range=self.axis_ranges["y4"], side='left', overlaying="y", anchor='free', titlefont=dict(color='black', size=20), tickfont=dict(color='black'), position=0, showgrid=False),
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            title="Trajectory Analysis",
        )

        fig.show()

# # Example usage
# axis_ranges = {
#     "x": [0, 100999],
#     "y1": [0, 12],
#     "y2": [0.006, 0.06],
#     "y3": [2, 3],
#     "y4": [0, 40]
# }

# plotter = TrajectoryPlot(t_cns, t_all_msrd, t_all_r, t_all_c_ads, axis_ranges)
# plotter.plot()


import numpy as np
import matplotlib.pyplot as plt

class TrajectoryPlot_MPL:
    def __init__(self, t_cns, t_all_msrd, t_all_r, t_all_c_ads, axis_ranges, plot_name):
        self.t_cns = t_cns
        self.t_all_msrd = t_all_msrd
        self.t_all_r = t_all_r
        self.t_all_c_ads = t_all_c_ads
        self.axis_ranges = axis_ranges
        self.plot_name = plot_name

    def plot(self):
        plt.rcParams.update({'font.size': 6})
        time = self.axis_ranges['time_range']
        fig, ax1 = plt.subplots(figsize=(4, 3), dpi=300)

        ax1.plot(time, self.t_cns, color='blue', label='CNs')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Pt-Pt Coordination number', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xlim(self.axis_ranges['x'])
        ax1.set_ylim(self.axis_ranges['y1'])

        ax2 = ax1.twinx()
        ax2.plot(time, self.t_all_msrd, color='red', label='MSRD')
        ax2.set_ylabel('Pt-Pt MSRD', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(self.axis_ranges['y2'])

        ax3 = ax1.twinx()
        ax3.plot(time, self.t_all_r, color='green', label='R')
        ax3.set_ylabel('Pt-Pt R', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        ax3.spines['right'].set_position(('outward', 80))
        ax3.set_ylim(self.axis_ranges['y3'])

        ax4 = ax1.twinx()
        ax4.plot(time, self.t_all_c_ads, color='black', linestyle='--', label='CO-Ads', alpha=0.5)
        ax4.set_ylabel('# of adsorbed CO', color='black')
        ax4.tick_params(axis='y', labelcolor='black')
        ax4.spines['right'].set_position(('outward', 40))
        ax4.set_ylim(self.axis_ranges['y4'])

        fig.tight_layout()
        plt.title(self.plot_name)
        plt.show()



