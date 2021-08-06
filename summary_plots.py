import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import umap
import umap.plot


# Various global definitions
contour_cmaps = [sns.light_palette('dodgerblue', as_cmap=True),
                 sns.light_palette('black', as_cmap=True),
                 sns.light_palette('red', as_cmap=True),
                 sns.light_palette('grey', as_cmap=True),
                 sns.light_palette('forestgreen', as_cmap=True),
                 sns.light_palette('hotpink', as_cmap=True)]
contour_colors = ['dodgerblue', 'black', 'red', 'grey', 'forestgreen',
                  'hotpink']
mut_titles = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
mut_types_c = []
mut_types_t = []
nucl = ["A", "C", "G", "T"]
m1 = ["[C>A]", "[C>G]", "[C>T]"]
m2 = ["[T>A]", "[T>C]", "[T>G]"]
for m in range(3):
    for n1 in nucl:
        for n2 in nucl:
            mut_types_c.append(n1 + m1[m] + n2)
            mut_types_t.append(n1 + m2[m] + n2)
mut_types = np.array(mut_types_c + mut_types_t)
mut_type_colors = (["dodgerblue"]*16 + ["black"]*16 + ["red"]*16 +
                   ["grey"]*16 + ["forestgreen"]*16 + ["hotpink"]*16)
mut_group_patches = [mpatches.Patch(color='dodgerblue', label="C>A"),
                     mpatches.Patch(color='black', label="C>G"),
                     mpatches.Patch(color='red', label="C>T"),
                     mpatches.Patch(color='grey', label="T>A"),
                     mpatches.Patch(color='forestgreen', label="T>C"),
                     mpatches.Patch(color='hotpink', label="T>G")]
mut_xlabs = ['ACA', 'ACC', 'ACG', 'ACT', 'CCA', 'CCC', 'CCG', 'CCT',
             'GCA', 'GCC', 'GCG', 'GCT', 'TCA', 'TCC', 'TCG', 'TCT',
             'ACA', 'ACC', 'ACG', 'ACT', 'CCA', 'CCC', 'CCG', 'CCT',
             'GCA', 'GCC', 'GCG', 'GCT', 'TCA', 'TCC', 'TCG', 'TCT',
             'ACA', 'ACC', 'ACG', 'ACT', 'CCA', 'CCC', 'CCG', 'CCT',
             'GCA', 'GCC', 'GCG', 'GCT', 'TCA', 'TCC', 'TCG', 'TCT',
             'ATA', 'ATC', 'ATG', 'ATT', 'CTA', 'CTC', 'CTG', 'CTT',
             'GTA', 'GTC', 'GTG', 'GTT', 'TTA', 'TTC', 'TTG', 'TTT',
             'ATA', 'ATC', 'ATG', 'ATT', 'CTA', 'CTC', 'CTG', 'CTT',
             'GTA', 'GTC', 'GTG', 'GTT', 'TTA', 'TTC', 'TTG', 'TTT',
             'ATA', 'ATC', 'ATG', 'ATT', 'CTA', 'CTC', 'CTG', 'CTT',
             'GTA', 'GTC', 'GTG', 'GTT', 'TTA', 'TTC', 'TTG', 'TTT']


def sample_residues(x, y, title, text_annotations=None, plot_names=False, figsize=(20, 4*3)):
    """
    Plots the mutation count summary of the given sample (observed and
    predicted counts, additive and multiplicative residues). If matrices
    are supplied, calculates and plots mean statistics.

    Args:
        x (array): Observed mutation counts.
        y (array): Predicted mutation count.
        title (string): Main plot title.
        plot_names (bool): Indicates whether to plot titles of the subplots. Defaults to False.
        text_annotations (array): Annotations for each plot. Defaults to None.
        figsize (tuple): Indicates plot dimensions.
    """

    width = figsize[0]
    height = figsize[1]
    sns.set(rc={"figure.figsize": (width, height)})
    sns.set(style="whitegrid", color_codes=True, rc={"grid.linewidth": 0.2, 'grid.color': '.7', 'ytick.major.size': 2,
                                                     'axes.edgecolor': '.3', 'axes.linewidth': 1.35})

    channel6 = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    col_list = ["dodgerblue", "black", "red", "grey", "forestgreen", "hotpink"]
    names = ["Observed number of mutations",
             "Predicted number of mutations",
             "Additive residue",
             "Multiplicative residue"]

    if x.shape[0] > 1:
        plot_data = np.array([np.mean(x, axis=0),
                              np.mean(y, axis=0),
                              np.mean(x - y, axis=0),
                              np.mean(np.log2((x + 1)/(y + 1)), axis=0)]).reshape(4, x.shape[1])
    else:
        plot_data = np.array([x,
                              y,
                              x - y,
                              np.log2((x + 1)/(y + 1))]).reshape(4, x.shape[1])

    ymax = np.max((np.max(plot_data[0]), np.max(plot_data[1])))
    plt.subplots(plot_data.shape[0], 1, figsize=figsize)
    plt.suptitle(title, va="bottom", y=1, fontsize=25)
    for i in range(plot_data.shape[0]):
        plt.subplot(plot_data.shape[0], 1, i+1)
        plt.bar(range(len(plot_data[i])), plot_data[i], color=mut_type_colors)
        # plt.xticks(range(len(plot_data[i][0])), mut_xlabs, rotation=90)
        plt.xticks([], [])

        if plot_names:
            plt.title(names[i], fontsize=20)

        if i == 0:
            plt.ylabel("Observed SBSs", fontsize=20)
            text_col = ["w", "w", "w", "black", "black", "black"]
            for j in range(6):
                left, width = 0 + 1/6 * j + 0.001, 1/6 - 0.002
                bottom, height = 1.003, 0.15
                right = left + width
                top = bottom + height
                ax = plt.gca()
                p = plt.Rectangle((left, bottom), width, height, fill=True, color=col_list[j])
                p.set_transform(ax.transAxes)
                p.set_clip_on(False)
                ax.add_patch(p)
                ax.text(0.5 * (left + right), 0.5 * (bottom + top), channel6[j],
                        color=text_col[j], weight='bold', size=20,
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
                ax.margins(x=0.002, y=0.002)

            plt.ylim(0, ymax + ymax*0.15)
        if i == 1:
            plt.ylabel("Predicted SBSs", fontsize=20)
            plt.ylim(0, ymax + ymax*0.15)
        if i == 2:
            plt.ylabel("Difference", fontsize=20)
            plt.axhline(0, linestyle="--", c="grey")
            ymin2 = np.min(plot_data[2])
            ymax2 = np.max(plot_data[2])
            plt.ylim(ymin2 + ymin2*0.15, ymax2 + ymax2*0.15)
        if i == 3:
            ymin3 = np.min(plot_data[3])
            ymax3 = np.max(plot_data[3])
            # ylim3 = np.max([np.abs(ymin3), np.abs(ymax3)])
            plt.axhline(0, linestyle="--", c="grey")
            plt.ylabel("Ratio (log2)", fontsize=20)
            # plt.ylim(ylim3 - 0.5, ylim3 + 0.5)
            plt.ylim(ymin3 + ymin3*0.15, ymax3 + ymax3*0.15)
            ax = plt.gca()
            ax.xaxis.grid(False)
            for k in range(len(mut_xlabs)):
                plt.text(k/len(mut_xlabs) + 0.0025, - 0.1, mut_xlabs[k][2],
                         color=mut_type_colors[k], rotation=90, fontsize=15, transform=ax.transAxes)
                plt.text(k/len(mut_xlabs) + 0.0025, - 0.175, mut_xlabs[k][1],
                         color='black', rotation=90, fontsize=15, transform=ax.transAxes)
                plt.text(k/len(mut_xlabs) + 0.0025, - 0.25, mut_xlabs[k][0],
                         color=mut_type_colors[k], rotation=90, fontsize=15, transform=ax.transAxes)

        if text_annotations:
            plt.text(1, ymax, text_annotations[i], fontsize=20)

        plt.margins(x=0.002, y=0.002)
        plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.show()


def plot_loglikelihood(log_liks, hline=None):
    """
    Plots the log-likelihood curve given the log-likelihood history.

    Args:
        log_liks (list): History of log-likelihoods.
        hline (float, optional): Y-axis value to draw a horizontal line from.
                                 Defaults to None.
    """

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(log_liks)), log_liks)
    if hline:
        plt.hlines(hline, 0, len(log_liks), linestyles="dashed", color="red")
    plt.title("Log-likelihood change per iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.savefig('loglik.png')
    plt.show()


def plot_signatures(x, title=None, names=None, xlab="Mutation type",
                    ylab="Probability"):
    """
    Plots multiple barplots of mutational signatures.

    Args:
        x (array-like): A matrix of mutational signatures,
                        where each row is a signature.
        title (str, optional): Main plot title. Defaults to None.
        names (array-like, optional): Titles for each plot. Defaults to None.
        xlab (str, optional): X-axis label. Defaults to "Mutation type".
        ylab (str, optional): Y-axis label. Defaults to "Probability".
    """

    plt.subplots(x.shape[0], 1, figsize=(20, x.shape[0]*3))
    plt.suptitle(title, va="bottom", y=0.9, size="x-large")
    for i in range(x.shape[0]):
        plt.subplot(x.shape[0], 1, i+1)
        plt.bar(range(len(x[i])), x[i], color=mut_type_colors)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.xticks(range(len(x[i])), mut_types, rotation=90)

        xlabs = plt.gca().get_xticklabels()
        for j in range(len(xlabs)):
            xlabs[j].set_color(mut_type_colors[j])

        if names is not None:
            plt.title(names[i])
        else:
            plt.title("Signature " + str(i))
        if i == 0:
            plt.legend(handles=mut_group_patches)
    plt.tight_layout()
    plt.show()


def mut_barplot(data, title, ylabel="Mutations", figsize=(20, 4), hline=None,
                ylim=None, filename=None, annotate_types=False):
    """
    Plots a barplot of channel-wise mutation counts.

    Args:
        data (array): Mutation count vector.
        title (str): Plot title.
        ylabel (str): Y-axis label. Defaults to "Mutations".
        figsize (tuple, optional): Figure dimensions. Defaults to (20, 3).
        hline (float, optional): Horizontal line to draw. Defaults to None.
        ylim (float, optional): Y-axis limits. Defaults to None.
        filename(str, optional): File path to save the figure to (only if specified). Defaults to None.
    """

    sns.set(rc={"figure.figsize": figsize})
    sns.set(style="whitegrid", color_codes=True, rc={"grid.linewidth": 0.2, 'grid.color': '.7',
                                                     'ytick.major.size': 2,
                                                     'axes.edgecolor': '.3', 'axes.linewidth': 1.35})

    channel6 = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    col_list = ["dodgerblue", "black", "red", "grey", "forestgreen", "hotpink"]

    fig = plt.figure(figsize=figsize)
    plt.bar(range(96), data, color=mut_type_colors)
    plt.title(title, fontsize=20, pad=50)
    plt.xticks(range(96), mut_types, rotation=90)

    xlabs = plt.gca().get_xticklabels()
    for j in range(len(xlabs)):
        xlabs[j].set_color(mut_type_colors[j])
    plt.xticks([], [])

    if hline is not False:
        plt.hlines(hline, 0, 96, linestyles="dashed", color="grey")

    if ylim:
        plt.ylim(ylim)

    text_col = ["w", "w", "w", "black", "black", "black"]
    for j in range(6):
        left, width = 0 + 1/6 * j + 0.001, 1/6 - 0.002
        bottom, height = 1.003, 0.2
        right = left + width
        top = bottom + height
        ax = plt.gca()
        p = plt.Rectangle((left, bottom), width, height, fill=True, color=col_list[j])
        p.set_transform(ax.transAxes)
        p.set_clip_on(False)
        ax.add_patch(p)
        ax.text(0.5 * (left + right), 0.5 * (bottom + top), channel6[j],
                color=text_col[j], weight='bold', size=20,
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        ax.margins(x=0.002, y=0.002)

    ax = plt.gca()
    ax.xaxis.grid(False)
    if annotate_types:
        for k in range(len(mut_xlabs)):
            plt.text(k/len(mut_xlabs) + 0.0025, - 0.1, mut_xlabs[k][2],
                     color=mut_type_colors[k], rotation=90, fontsize=15, transform=ax.transAxes)
            plt.text(k/len(mut_xlabs) + 0.0025, - 0.175, mut_xlabs[k][1],
                     color='black', rotation=90, fontsize=15, transform=ax.transAxes)
            plt.text(k/len(mut_xlabs) + 0.0025, - 0.25, mut_xlabs[k][0],
                     color=mut_type_colors[k], rotation=90, fontsize=15, transform=ax.transAxes)

    # plt.xlabel("Mutation type")
    plt.ylabel(ylabel, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(ylim)
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()


def count_scatterplot(x, y, title, figsize=(8, 8)):
    """
    Plots scatterplot of observed vs. predicted mutation counts,
    colored by base mutation type.

    Args:
        x (matrix): Observed counts.
        y (matrix): Predicted counts.
        title (str): Plot title.
        figsize (tuple, optional): Figure dimensions. Defaults to (8, 8).
    """

    plt.figure(figsize=figsize)
    for i in range(x.shape[0]):
        plt.scatter(x[i], y[i], color=mut_type_colors, marker='.')

    xmax = np.max([np.max(x), np.max(y)]) + 1
    xmin = -0.5
    ymax = xmax
    ymin = xmin

    plt.plot([xmin, xmax], [ymin, ymax], linestyle='dashed', color='black',
             alpha=0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.legend(mut_group_patches, mut_titles,
               title="Channels")
    plt.show()


def plot_summary_statistics(x, y, labels, title, filename):
    """
    Plot the summary statistics in log10 scale (where applicable):
     - Label distribution;
     - Cluster/control channel-wise mutation count median;
     - Cluster/control channel-wise additive residue median;
     - Cluster/control channel-wise multiplicative residue median;

     Args:
        x (array-like): Matrix of observed cluster mutation counts.
        y (array-like): Matrix of predicted cluster mutation counts.
        labels (array): Label for each sample in the cluster.
        title (str): title for the plot.
        filename (str): where to save the plot.
    """

    fig = plt.figure(figsize=(15*4, 2*4), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=4)

    ax11 = fig.add_subplot(gs[0, :2])
    ax21 = fig.add_subplot(gs[1, :2], sharey=ax11)
    ax31 = fig.add_subplot(gs[0, 2:])
    ax51 = fig.add_subplot(gs[1, 2:])

    # Observed median channel-wise mutation counts in log10 in cluster
    ax11.bar(range(96), np.median(x, axis=0),
             color=mut_type_colors)
    ax11.set_ylabel("Mutations", fontsize=15)
    ax11.set_title("Cluster median observed mutation counts\n", fontsize=20)
    ax11.set_xticks(range(96))
    ax11.set_xticklabels(mut_types, rotation=90)
    ax11.set_yticklabels(ax11.get_yticks(), fontsize=15)
    for xtick, color in zip(ax11.get_xticklabels(), mut_type_colors):
        xtick.set_color(color)

    # Predicted median channel-wise mutation counts in log10 in cluster
    ax21.bar(range(96), np.median(y, axis=0),
             color=mut_type_colors)
    ax21.set_ylabel("Mutations", fontsize=15)
    ax21.set_title("Cluster median predicted mutation counts\n", fontsize=20)
    ax21.set_xticks(range(96))
    ax21.set_xticklabels(mut_types, rotation=90)
    ax21.set_yticklabels(ax21.get_yticks(), fontsize=15)
    for xtick, color in zip(ax21.get_xticklabels(), mut_type_colors):
        xtick.set_color(color)

    # Cluster KL residue median
    ax31.bar(range(96),
             np.median(x - y, axis=0),
             color=mut_type_colors)
    ax31.set_ylabel("Difference", fontsize=15)
    ax31.set_title("Cluster median additive residue\n", fontsize=20)
    ax31.set_xticks(range(96))
    ax31.set_xticklabels(mut_types, rotation=90)
    ax31.set_yticklabels(ax31.get_yticks(), fontsize=15)
    for xtick, color in zip(ax31.get_xticklabels(), mut_type_colors):
        xtick.set_color(color)

    # Cluster multiplicative residue median
    ax51.bar(range(96), np.median(x/(y + 1), axis=0), color=mut_type_colors)
    ax51.hlines(1, -1, 97, linestyles="dashed", color="grey")
    ax51.set_ylabel("Ratio", fontsize=15)
    ax51.set_title("Cluster median multiplicative residue\n", fontsize=20)
    ax51.set_xticks(range(96))
    ax51.set_xticklabels(mut_types, rotation=90)
    ax51.set_yticklabels(ax51.get_yticks(), fontsize=15)
    for xtick, color in zip(ax51.get_xticklabels(), mut_type_colors):
        xtick.set_color(color)

    plt.suptitle(title, fontsize=25)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def residue_umaps(x, y, labels, ctype_cols, data_title, show_legend=False):
    """[summary]

    Args:
        x ([type]): [description]
        y ([type]): [description]
        labels ([type]): [description]
        ctype_cols ([type]): [description]
        data_title ([type]): [description]
        show_legend (bool, optional): [description]. Defaults to False.
    """

    params = {'legend.fontsize': 5,
              'legend.handlelength': 2}
    plt.rcParams.update(params)
    fig = plt.figure(figsize=(8*2, 8), constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    mapper = umap.UMAP(metric="correlation").fit(x - y)
    umap.plot.points(mapper, labels=labels, color_key=ctype_cols,
                     width=800, height=800, show_legend=show_legend, ax=ax1)
    ax1.set_title(data_title + " additive residue correlation", fontsize=15)

    mapper = umap.UMAP(metric="correlation").fit(x/(y + 1))
    umap.plot.points(mapper, labels=labels, color_key=ctype_cols,
                     width=800, height=800, show_legend=show_legend, ax=ax2)
    ax2.set_title(data_title + " multiplicative residue correlation", fontsize=15)
    plt.tight_layout()
    plt.show()


def channel_scatterplot(obs, pred, title, filename=None, clim=0.5):
    """
    Plots 6 contour/scatter plots for each base mutation type.

    Args:
        obs (array-like): Observed mutation counts.
        pred (array-like): Predicted mutation counts.
        title (str): Plot title.
        filename (str): Where to save the plot.
        clim (float): A value for minimum level cutoff for contour plot (relevant to plotting colours).
                      Defaults to 0.5.
    """

    real = np.log10(obs + 1)
    fit = np.log10(pred + 1)

    xmax = np.max([np.max(real), np.max(fit)]) + 0.1
    xmin = -0.1
    ymax = xmax
    ymin = xmin

    fig = plt.figure(figsize=(8*6, 8*1), constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=6)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1], sharex=ax00, sharey=ax00)
    ax10 = fig.add_subplot(gs[0, 2], sharex=ax00, sharey=ax00)
    ax11 = fig.add_subplot(gs[0, 3], sharex=ax00, sharey=ax00)
    ax20 = fig.add_subplot(gs[0, 4], sharex=ax00, sharey=ax00)
    ax21 = fig.add_subplot(gs[0, 5], sharex=ax00, sharey=ax00)
    axes = [ax00, ax01, ax10, ax11, ax20, ax21]
    axc = 0
    for i in range(0, 6):
        bar_colors = [contour_colors[i]]*16
        alphas = [0.25]*16
        markers = ['.']*16

        markers = markers*real.shape[0]
        alphas = alphas*real.shape[0]
        bar_colors = bar_colors*real.shape[0]

        x = real[:, i*16:i*16+16].flatten()
        y = fit[:, i*16:i*16+16].flatten()

        for xi in range(len(x)):
            sns.scatterplot(x=[x[xi]], y=[y[xi]], ax=axes[axc],
                            color=bar_colors[xi], alpha=alphas[xi],
                            marker=markers[xi], s=250)
        axes[axc].plot([xmin, xmax], [ymin, ymax], linestyle='dashed',
                       color='black', alpha=0.5)
        axes[axc].set_xlim(xmin, xmax)
        axes[axc].set_ylim(ymin, ymax)

        # axes[axc].set_title(mut_titles[i], fontsize=20)
        axes[axc].text(0.05, 0.9, mut_titles[i], fontsize=50,
                       transform=axes[axc].transAxes)
        axes[axc].get_xaxis().set_visible(False)
        if axc == 0:
            # axes[axc].set_xlabel("Observed", fontsize=20)
            axes[axc].set_ylabel("Predicted", fontsize=50)
            # plt.setp(axes[axc].get_xticklabels(), fontsize=20)
            plt.setp(axes[axc].get_yticklabels(), fontsize=40)
            axes[axc].get_yaxis().set_visible(True)
        else:
            axes[axc].get_yaxis().set_visible(False)

        axes[axc].hlines(y=axes[axc].get_yticks(),
                         xmin=xmin, xmax=xmax, colors="grey", linestyles='-',
                         alpha=0.15)
        axc += 1
        print(mut_titles[i], "done.")

    plt.suptitle("Observed", fontsize=50, y=0)
    if title:
        plt.suptitle(title, fontsize=25)

    plt.tight_layout(pad=0)

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def channel_scatterplot_joint(obs, pred, r, c, title, filename=None, clim=0.5):
    """
    Plots 6 contour/scatter plots for each base mutation type.

    Args:
        obs (array-like): Observed mutation counts.
        pred (array-like): Predicted mutation counts.
        title (str): Plot title.
        filename (str): Where to save the plot.
        clim (float): A value for minimum level cutoff for contour plot (relevant to plotting colours).
                      Defaults to 0.5.
    """

    real = np.log10(obs + 1)
    fit = np.log10(pred + 1)
    repaired = np.log10((1 + c.reshape(-1, 1)*r.reshape(1, -1))*pred + 1)

    xmax = np.max([np.max(real), np.max(fit)]) + 0.1
    xmin = -0.1
    ymax = xmax
    ymin = xmin

    fig = plt.figure(figsize=(8*6, 8*1), constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=6)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1], sharex=ax00, sharey=ax00)
    ax10 = fig.add_subplot(gs[0, 2], sharex=ax00, sharey=ax00)
    ax11 = fig.add_subplot(gs[0, 3], sharex=ax00, sharey=ax00)
    ax20 = fig.add_subplot(gs[0, 4], sharex=ax00, sharey=ax00)
    ax21 = fig.add_subplot(gs[0, 5], sharex=ax00, sharey=ax00)
    axes = [ax00, ax01, ax10, ax11, ax20, ax21]
    axc = 0

    for i in range(0, 6):
        bar_colors = [contour_colors[i]]*16
        alphas = [0.5]*16
        markers = ['X']*16
        markers_alt = ['D']*16

        markers = markers*real.shape[0]
        markers_alt = markers_alt*real.shape[0]
        alphas = alphas*real.shape[0]
        bar_colors = bar_colors*real.shape[0]

        x = real[:, i*16:i*16+16].flatten()
        y = fit[:, i*16:i*16+16].flatten()
        z = repaired[:, i*16:i*16+16].flatten()

        for xi in range(len(x)):
            sns.scatterplot(x=[x[xi]], y=[y[xi]], ax=axes[axc],
                            color="red", alpha=alphas[xi],
                            marker=markers[xi], s=250)
            sns.scatterplot(x=[x[xi]], y=[z[xi]], ax=axes[axc],
                            color="cornflowerblue", alpha=alphas[xi],
                            marker=markers_alt[xi], s=250)
        axes[axc].plot([xmin, xmax], [ymin, ymax], linestyle='dashed',
                       color='black', alpha=0.5)
        axes[axc].set_xlim(xmin, xmax)
        axes[axc].set_ylim(ymin, ymax)

        channel6 = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
        col_list = ["dodgerblue", "black", "red", "grey", "forestgreen", "hotpink"]
        text_col = ["w", "w", "w", "black", "black", "black"]
        left, width = 0, 1
        bottom, height = 1.003, 0.15
        right = left + width
        top = bottom + height
        p = plt.Rectangle((left, bottom), width, height, fill=True, color=col_list[i])
        p.set_transform(axes[axc].transAxes)
        p.set_clip_on(False)
        axes[axc].add_patch(p)
        axes[axc].text(0.5 * (left + right), 0.5 * (bottom + top), channel6[i],
                       color=text_col[i], weight='bold', size=50,
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[axc].transAxes)
        axes[axc].margins(x=0.002, y=0.002)

        # axes[axc].set_title(mut_titles[i], fontsize=20)
        error_1 = np.round(np.mean((fit[:, i*16:i*16+16] - real[:, i*16:i*16+16])**2), 5)
        error_2 = np.round(np.mean((repaired[:, i*16:i*16+16] - real[:, i*16:i*16+16])**2), 5)
        axes[axc].text(0.05, 0.9, "MSE = " + str(error_1), fontsize=40,
                       transform=axes[axc].transAxes, color="red")
        axes[axc].text(0.05, 0.8, "MSE = " + str(error_2), fontsize=40,
                       transform=axes[axc].transAxes, color="cornflowerblue")
        axes[axc].get_xaxis().set_visible(False)
        if axc == 0:
            # axes[axc].set_xlabel("Observed", fontsize=20)
            axes[axc].set_ylabel("Predicted", fontsize=50, labelpad=25)
            # plt.setp(axes[axc].get_xticklabels(), fontsize=20)
            plt.setp(axes[axc].get_yticklabels(), fontsize=40)
            axes[axc].get_yaxis().set_visible(True)
        else:
            axes[axc].get_yaxis().set_visible(False)

        axes[axc].hlines(y=axes[axc].get_yticks(),
                         xmin=xmin, xmax=xmax, colors="grey", linestyles='-',
                         alpha=0.15)
        axc += 1
        print(mut_titles[i], "done.")

    plt.suptitle("Observed", fontsize=50, y=0)
    if title:
        plt.suptitle(title, fontsize=25)

    plt.tight_layout(pad=0)

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
