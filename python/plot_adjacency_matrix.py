"""
plot_adjacency_matrix.py

Plot all neuron interactions as adjacency matrices. R neurons are excluded due
to the plastic nature of those connections. See the main paper and
extended_ring_model.py for anatomical justification.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from extended_ring_model import *

if __name__ == "__main__":
    rm = RingModel()

    mpl.rcParams["axes.labelpad"] = -10
    mosaic = [
        ["epg-d7", "d7-pen", "d7-pen", "epg-pen", "epg-pen", "pen-epg", "pen-epg"],
        ["d7-d7", "d7-peg", "d7-peg", "epg-peg", "epg-peg", "peg-epg",  "peg-epg"]

    ]

    vmin = -1.5
    vmax = 1.5
    cmap = "coolwarm"
    fig, axs = plt.subplot_mosaic(mosaic, figsize=(10,5))

    axs["epg-d7"].pcolormesh(rm.w_epg_d7, vmin=vmin, vmax=vmax, cmap=cmap)  # 8x8
    axs["epg-d7"].set_title(r"E-PG $\rightarrow \Delta7$")
    axs["epg-d7"].set_xlabel("E-PG")
    axs["epg-d7"].set_ylabel(r"$\Delta 7$")
    axs["epg-d7"].set_xticks([0,8])

    axs["d7-d7"].set_title(r"$\Delta7 \rightarrow \Delta7$")
    axs["d7-d7"].pcolormesh(rm.w_d7_d7,  vmin=vmin, vmax=vmax, cmap=cmap) # 8x8
    axs["d7-d7"].set_xlabel(r"$\Delta 7$")
    axs["d7-d7"].set_ylabel(r"$\Delta 7$")
    axs["d7-d7"].set_xticks([0,8])

    axs["d7-pen"].set_title(r"$\Delta7 \rightarrow$ P-EN")
    axs["d7-pen"].pcolormesh(rm.w_d7_pen.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs["d7-pen"].set_xticks([0,16])
    axs["d7-pen"].set_ylabel(r"$\Delta 7$")
    axs["d7-pen"].set_xlabel("P-EN")

    axs["d7-peg"].set_title(r"$\Delta7 \rightarrow$ P-EG")
    axs["d7-peg"].pcolormesh(rm.w_d7_peg.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs["d7-peg"].set_xticks([0,16])
    axs["d7-peg"].set_ylabel(r"$\Delta 7$")
    axs["d7-peg"].set_xlabel("P-EG")

    axs["epg-pen"].set_title(r"E-PG $\rightarrow$ P-EN")
    axs["epg-pen"].pcolormesh(rm.w_epg_pen.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs["epg-pen"].set_xticks([0,16])
    axs["epg-pen"].set_ylabel("E-PG")
    axs["epg-pen"].set_xlabel("P-EN")

    axs["epg-peg"].set_title(r"E-PG $\rightarrow$ P-EG")
    axs["epg-peg"].pcolormesh(rm.w_epg_peg.T, vmin=vmin, vmax=vmax, cmap=cmap)
    axs["epg-peg"].set_xticks([0,16])
    axs["epg-peg"].set_ylabel("E-PG")
    axs["epg-peg"].set_xlabel("P-EG")

    axs["pen-epg"].set_title(r"P-EN $\rightarrow$ E-PG")
    axs["pen-epg"].pcolormesh(rm.w_pen_epg, vmin=vmin, vmax=vmax, cmap=cmap)
    axs["pen-epg"].set_xticks([0,16])
    axs["pen-epg"].set_ylabel("E-PG")
    axs["pen-epg"].set_xlabel("P-EN")

    axs["peg-epg"].set_title(r"P-EG $\rightarrow$ E-PG")
    wmap = axs["peg-epg"].pcolormesh(rm.w_peg_epg, vmin=vmin, vmax=vmax, cmap=cmap)
    axs["peg-epg"].set_xticks([0,16])
    axs["peg-epg"].set_ylabel("E-PG")
    axs["peg-epg"].set_xlabel("P-EG")

    for k in axs.keys():
        axs[k].set_aspect('equal')
        axs[k].set_yticks([0,8])

    cbar_ax = fig.add_axes([0.4, 0.05, 0.5, 0.02])
    cbar_ax.set_ylabel("synapse strength",rotation='horizontal',labelpad=4,ha='right',va='top')
    fig.colorbar(wmap, cax=cbar_ax, orientation='horizontal')

    fig.suptitle("Neuron connectivity (R neurons not shown)")
    fig.tight_layout()
    fig.savefig("plots/adjacency.pdf",bbox_inches="tight")
#    plt.show()
