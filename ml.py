import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
from summary_plots import (mut_barplot, channel_scatterplot_joint)
from utilities import unique

# Below we define the log-likelihood function of the multiplicative (extended
# additive) mutational signature model and the derivatives with respect to the
# inferred parameters (modulatory process r and its activities per sample c).


# The main log-likelihood function that is to be minimized.
def f(rc, X, lam):
    r = rc[:96]
    c = rc[96:]
    result = 0
    for k in range(X.shape[0]):
        for j in range(X.shape[1]):
            result += X[k, j]*np.log((1 + c[k]*r[j])*lam[k, j]) - (1 + c[k]*r[j])*lam[k, j]
    return -result


# Alternative log-likelihood function for log-likelihood change analyses.
def sample_f(rc, X, lam):
    r = rc[:96]
    c = rc[96:]
    result = np.zeros(len(X))
    for k in range(X.shape[0]):
        for j in range(X.shape[1]):
            result[k] += X[k, j]*np.log((1 + c[k]*r[j])*lam[k, j]) - (1 + c[k]*r[j])*lam[k, j]
    return -result


# Element-wise derivatives for r and c
def dfdrc(rc, X, lam):
    r = rc[:96]  # Get the modulatory process r
    c = rc[96:]  # Get the activity vector c
    dr = np.zeros(len(r))
    dc = np.zeros(len(X))
    for k in range(X.shape[0]):
        for j in range(X.shape[1]):
            dr[j] += (X[k, j]*c[k])/(1 + c[k]*r[j]) - c[k]*lam[k, j]
            dc[k] += (X[k, j]*r[j])/(1 + c[k]*r[j]) - r[j]*lam[k, j]
    return np.hstack([-dr.flatten(), -dc.flatten()])


def ml_inference(counts, signatures, activities, full_labels, path, plot=True, rdf_out="r_all.csv"):
    fit = np.dot(activities, signatures)
    labels = np.array([x.split("::")[0] for x in full_labels])

    r_ctype_df = pd.DataFrame(data=np.zeros((len(unique(labels)), 97)))
    r_ctype_df[0] = sorted(unique(labels))
    r_ctype_df = r_ctype_df.set_index(0)

    opts = {'eta': 0.5,
            'stepmx': 200,
            'disp': True
            }

    for ctype in sorted(unique(labels)):
        print(ctype)
        ctidx = np.where(labels == ctype)[0]

        X = counts[ctidx] + 1
        x = activities[ctidx]
        mu = signatures
        lam = np.dot(x, mu) + 1

        bounds = ()
        for i in range(X.shape[1]):
            bounds += ((-1, 1), )
        for i in range(X.shape[0]):
            bounds += ((0, 10), )
        bounds

        base_logliks = sample_f(np.hstack([np.zeros(96).flatten(), np.zeros(len(X)).flatten()]), X, lam)

        success = False
        n = 1
        while not success:
            print("Run " + str(n) + "...")
            r0 = np.random.uniform(low=-1, high=1, size=96)
            c0 = np.random.rand(len(X))
            x0 = np.hstack([r0.flatten(), c0.flatten()])
            res = scipy.optimize.minimize(f, x0, jac=dfdrc, args=(X, lam), method="TNC",
                                          bounds=bounds, options=opts)
            success = res.success
            if success:
                print("Success at run " + str(n))

            print(res.message)
            n += 1

        r = res.x[:96]
        r = r/np.max(np.abs(r))
        c = res.x[96:]
        c = c/np.max(np.abs(r))
        print("Inference done.\n")
        new_logliks = sample_f(res.x, X, lam)
        loglik_change = pd.DataFrame(data=(base_logliks, new_logliks), index=("Before", "After")).T
        loglik_change.to_csv(path + ctype + "_ll_change.csv", sep=",")

        c_ctype_df = pd.DataFrame(data=np.zeros((c.shape[0], 1)))
        c_ctype_df[0] = full_labels[ctidx]
        c_ctype_df = c_ctype_df.set_index(0)
        c_ctype_df[0] = c
        c_ctype_df.to_csv(path + ctype + "_c.csv", sep=",")
        r_ctype_df.loc[ctype] = r

        if plot:
            print("Plotting the inferred modulatory process...")
            mut_barplot(r, title="", ylabel="Modulation", hline=0.0,
                        filename=path + ctype + "_mp.png", ylim=(-1.1, 1.1))
            print("Done.\n")

            print("Plotting the channel-wise scatter-plots...")
            channel_scatterplot_joint(X, lam, r, c, title="",
                                      filename=path + ctype + "_repaired_scatter.png")
            print("Done.\n")

            print("Plotting the activity histogram...")
            fig = plt.figure(figsize=(8, 5))
            plt.hist(res.x[96:], bins=20)
            plt.title(ctype + " multiplicative process activity histogram")
            plt.xlabel("Activity")
            plt.ylabel("Frequency")
            plt.savefig(path + ctype + "_activity_hist.png")
            plt.close(fig)
            print("Done.\n")

    r_ctype_df.to_csv(path + rdf_out, sep=",")
