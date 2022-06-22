import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hcluster


def plot_dendrogram(Z, labels, threshold, filename):
    """
    Plots hierarchical clustering dendrogram given the linkage matrix Z.

    Args:
        Z (matrix): Linkage matrix.
        labels (array): Sample labels.
        threshold (float): Hierarchical clustering threshold.
        filename (str): Where to save the plot.
    """

    fc = hcluster.fcluster(Z, t=threshold, criterion='distance')
    cmap = matplotlib.cm.rainbow(np.linspace(0, 1, len(set(fc))))
    hcluster.set_link_color_palette([matplotlib.colors.rgb2hex(rgb[:3]) for rgb in cmap])

    plt.figure(figsize=(125, 5))
    ax = plt.gca()
    plt.hlines(threshold, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors="red", linestyles="dashed")
    lab_coords = ax.get_xticks()

    prev_size = 0
    for i in range(len(set(fc)) - 1):
        cluster_size = len(np.where(fc == i+1)[0])
        current = lab_coords[prev_size:(prev_size + cluster_size)]
        prev_size += cluster_size
        middle = int(len(current)/2)
        if middle % 2 == 0:
            x = current[middle]
        else:
            x = current[middle - 1]
        y = -0.035
        plt.text(x, y, str(i+1), fontsize=5)
        plt.text(current[-1], y, "|", fontsize=5)

    plt.savefig(filename, dpi=250)
    plt.close()


def plot_heatmap_with_dendrogram(Z, dists, labels, totals, ctype_cols,
                                 threshold, filename, cluster_labels=False,
                                 text_y=-0.075):
    """
    Plots a hierarchically clustered heatmap and a corresponding dendrogram,
    together with cancer type color bar and a 1D mutation count heatmap.

    Args:
        Z (matrix): Linkage matrix.
        dists (matrix): Pairwise distances between data entries.
        labels (array): Sample labels.
        totals (array): Total mutations per sample.
        threshold (float): Hierarchical clustering threshold.
        filename (str): Where to save the plot.
        text_y (float): y-position adjustment for cluster labels. Defaults to -0.075.
    """

    fc = hcluster.fcluster(Z, t=threshold, criterion='distance')
    cmap = matplotlib.cm.rainbow(np.linspace(0, 1, len(set(fc))))
    hcluster.set_link_color_palette([matplotlib.colors.rgb2hex(rgb[:3]) for rgb in cmap])

    plt.figure(figsize=(125, 125))
    gs = matplotlib.gridspec.GridSpec(nrows=32, ncols=10, hspace=1)

    plt.subplot(gs[:5, :8])
    dn = hcluster.dendrogram(Z, color_threshold=0)
    idx = dn['leaves']
    plt.hlines(threshold, xmin=plt.gca().get_xlim()[0], xmax=plt.gca().get_xlim()[1], colors="red", linestyles="dashed")

    lab_coords = range(5, 27800, 10)

    if cluster_labels:
        prev_size = 0
        for i in range(len(set(fc))):
            cluster_size = len(np.where(fc == i+1)[0])
            current = lab_coords[prev_size:(prev_size + cluster_size)]
            prev_size += cluster_size
            if len(current) > 1:
                middle = int(len(current)/2)
                if middle % 2 == 0:
                    x = current[middle]
                else:
                    x = current[middle - 1]
            else:
                x = current[0]

            y = text_y
            plt.text(x, y, str(i+1), fontsize=8, rotation=90)
            plt.text(current[-1], y, "|", fontsize=20)

    plt.subplot(gs[5:30, :8])
    # plt.subplot(212)
    plt.imshow(dists[idx, :][:, idx], cmap='bwr', aspect="auto")
    plt.axis('off')

    cbar_ax = plt.subplot(gs[30:, :8])
    cbar_cmap = matplotlib.cm.bwr
    cbar_norm = matplotlib.colors.Normalize(vmin=np.min(dists), vmax=np.max(dists))
    cbar_h = plt.colorbar(matplotlib.cm.ScalarMappable(norm=cbar_norm, cmap=cbar_cmap),
                          cax=cbar_ax, orientation="horizontal")
    cbar_h.ax.tick_params(labelsize=40)
    cbar_h.set_label(label="Correlation", size=50)

    plt.subplot(gs[5:30, 8:9])
    plt.imshow(totals, aspect="auto")
    plt.title("log10 mutations", fontdict={'fontsize': 50})
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=40)
    plt.axis('off')

    ctype_to_num = {}
    i = 1
    for key in ctype_cols.keys():
        ctype_to_num[key] = i
        i += 1
    cnum_list = np.array([ctype_to_num[x] for x in labels])
    cm = matplotlib.colors.LinearSegmentedColormap.from_list('newlist', list(ctype_cols.values()), N=len(cnum_list))

    plt.subplot(gs[5:30, 9:])
    plt.imshow(cnum_list[idx].reshape(1, -1).T, cmap=cm, aspect="auto")
    plt.title("Cancer type", fontdict={'fontsize': 50})
    plt.axis('off')

    legend_elements = []
    for ctype in ctype_cols.keys():
        legend_elements.append(matplotlib.patches.Patch(facecolor=ctype_cols[ctype], label=ctype))
    plt.subplot(gs[:5, 8:])
    legend = plt.legend(handles=legend_elements, loc="center", fontsize=35, ncol=2, title="Cancer type")
    legend.get_title().set_fontsize('50')
    plt.axis('off')

    # plt.tight_layout()
    plt.subplots_adjust(hspace=1)
    plt.savefig(filename, dpi=100)
    plt.close()


def plot_heatmap_with_dendrogram_tri(Z, dists1, dists2, labels, totals,
                                     ctype_cols, threshold, filename, dpi,
                                     cluster_labels=False, text_y=-0.075,
                                     vmin=-1, vmax=1):
    """
    Plots a hierarchically clustered heatmap and a corresponding dendrogram,
    together with cancer type color bar and a 1D mutation count heatmap.
    This version uses two distance matrices to plot two triangular matrices
    for direct comparison (samples are ordered by the clustering of the first,
    i.e. primary matrix).

    Args:
        Z (matrix): Linkage matrix (primary).
        dists1 (matrix): Pairwise distances between data entries (primary).
        dists2 (matrix): Pairwise distances between data entries (secondary).
        labels (array): Sample labels.
        totals (array): Total mutations per sample.
        ctype_cols (dict): A dictionary assigning colors to each label (i.e. cancer type).
        threshold (float): Hierarchical clustering threshold (primary).
        filename (str): Where to save the plot.
        cluster_labels (bool): Boolean indicating whether to show cluster labels. Defaults to False.
        text_y (float): y-position adjustment for cluster labels. Defaults to -0.075.
    """

    fc = hcluster.fcluster(Z, t=threshold, criterion='distance')
    cmap = matplotlib.cm.rainbow(np.linspace(0, 1, len(set(fc))))
    hcluster.set_link_color_palette([matplotlib.colors.rgb2hex(rgb[:3]) for rgb in cmap])

    plt.figure(figsize=(125, 125))
    gs = matplotlib.gridspec.GridSpec(nrows=32, ncols=10, hspace=1)

    plt.subplot(gs[:5, :8])
    dn = hcluster.dendrogram(Z, color_threshold=0, no_labels=True)
    ax = plt.gca()
    ax.tick_params(axis='y', which='major', labelsize=100)
    idx = dn['leaves']
    mat = np.tril(dists2[idx][:, idx]) + np.triu(dists1[idx][:, idx])
    np.fill_diagonal(mat, 0)
    plt.hlines(threshold, xmin=plt.gca().get_xlim()[0], xmax=plt.gca().get_xlim()[1],
               colors="red", linestyles="dashed")

    lab_coords = range(5, 27800, 10)

    if cluster_labels:
        prev_size = 0
        for i in range(len(set(fc))):
            cluster_size = len(np.where(fc == i+1)[0])
            current = lab_coords[prev_size:(prev_size + cluster_size)]
            prev_size += cluster_size
            if len(current) > 1:
                middle = int(len(current)/2)
                if middle % 2 == 0:
                    x = current[middle]
                else:
                    x = current[middle - 1]
            else:
                x = current[0]

            y = text_y
            plt.text(x, y, str(i+1), fontsize=8, rotation=90)
            plt.text(current[-1], y, "|", fontsize=20)

    plt.subplot(gs[5:30, :8])
    # plt.subplot(212)
    plt.imshow(mat,
               cmap='bwr', aspect="auto",
               vmin=vmin, vmax=vmax)
    plt.axis('off')

    cbar_ax = plt.subplot(gs[30:, :8])
    cbar_cmap = matplotlib.cm.bwr
    cbar_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar_h = plt.colorbar(matplotlib.cm.ScalarMappable(norm=cbar_norm, cmap=cbar_cmap),
                          cax=cbar_ax, orientation="horizontal")
    cbar_h.ax.tick_params(labelsize=150)
    cbar_h.set_label(label="Correlation", size=150)

    """
    plt.subplot(gs[5:30, 8:9])
    plt.imshow(totals, aspect="auto")
    plt.title("log10 mutations", fontdict={'fontsize': 50})
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=40)
    plt.axis('off')
    """

    ctype_to_num = {}
    i = 1
    for key in ctype_cols.keys():
        ctype_to_num[key] = i
        i += 1
    cnum_list = np.array([ctype_to_num[x] for x in labels])
    cm = matplotlib.colors.LinearSegmentedColormap.from_list('newlist', list(ctype_cols.values()), N=len(cnum_list))

    plt.subplot(gs[5:30, 8:9])
    plt.imshow(cnum_list[idx].reshape(1, -1).T, cmap=cm, aspect="auto")
    plt.title("Cancer type", fontdict={'fontsize': 125}, pad=100)
    plt.axis('off')

    """
    legend_elements = []
    for ctype in ctype_cols.keys():
        legend_elements.append(matplotlib.patches.Patch(facecolor=ctype_cols[ctype], label=ctype))
    plt.subplot(gs[:5, 8:])
    legend = plt.legend(handles=legend_elements, loc="center", fontsize=35, ncol=2, title="Cancer type")
    legend.get_title().set_fontsize('50')
    plt.axis('off')
    """

    # plt.tight_layout()
    plt.subplots_adjust(hspace=1)
    plt.savefig(filename, dpi=dpi, pad_inches=0)
    plt.close()
