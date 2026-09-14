
import matplotlib
import seaborn as sns
FONTSIZE = 11
PAGEWIDTH = 11
def init_plt():
    sns.set_style("whitegrid")
    matplotlib.rcParams.update(
        {
            "font.size": FONTSIZE,
            "axes.titlesize": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "legend.fontsize": FONTSIZE,
            "figure.titlesize": FONTSIZE,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "figure.figsize": (PAGEWIDTH / 2, PAGEWIDTH / 2),
            "lines.markeredgewidth": 0.8,
            "axes.edgecolor": "black",
            "axes.grid": False,
            "grid.color": "0.9",
            "axes.grid.which": "both",
            "xtick.bottom": True,
            "xtick.direction": "out",
            "xtick.color": "black",
            "xtick.major.bottom": True,
            "xtick.major.size": 4,
            "xtick.minor.bottom": True,
            "xtick.minor.size": 2,
            "ytick.left": True,
            "ytick.direction": "out",
            "ytick.color": "black",
            "ytick.major.left": True,
            "ytick.major.size": 4,
            "ytick.minor.left": True,
            "ytick.minor.size": 2,
        }
    )






