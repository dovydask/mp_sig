import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
from summary_plots import (mut_barplot, channel_scatterplot_joint)
from utilities import unique

# %%
# Below we define the log-likelihood function of the multiplicative (extended
# additive) mutational signature model and the derivatives with respect to the
# inferred parameters (modulatory process r and its activities per sample c).

# The main log-likelihood function that is to be minimized.
def fa(a, X, mu, r, c):
    nsamp = X.shape[0]
    nch = X.shape[1]
    nsig = mu.shape[0]
    a = a.reshape(nsamp, nsig)
    lam = np.dot(a, mu) + 1
    result = 0
    for k in range(nsamp):
        for j in range(nch):
            result += X[k, j]*np.log((1 + c[k]*r[j])*lam[k, j]) - (1 + c[k]*r[j])*lam[k, j]
    return -result


def fr(r, X, mu, a, c):
    nsamp = X.shape[0]
    nch = X.shape[1]
    nsig = mu.shape[0]
    a = a.reshape(nsamp, nsig)
    lam = np.dot(a, mu) + 1
    result = 0
    for k in range(nsamp):
        for j in range(nch):
            result += X[k, j]*np.log((1 + c[k]*r[j])*lam[k, j]) - (1 + c[k]*r[j])*lam[k, j]
    return -result


def fc(c, X, mu, a, r):
    nsamp = X.shape[0]
    nch = X.shape[1]
    nsig = mu.shape[0]
    a = a.reshape(nsamp, nsig)
    lam = np.dot(a, mu) + 1
    result = 0
    for k in range(nsamp):
        for j in range(nch):
            result += X[k, j]*np.log((1 + c[k]*r[j])*lam[k, j]) - (1 + c[k]*r[j])*lam[k, j]
    return -result


def frc(rc, X, mu, a):
    nsamp = X.shape[0]
    nch = X.shape[1]
    nsig = mu.shape[0]
    a = a.reshape(nsamp, nsig)
    lam = np.dot(a, mu) + 1
    r = rc[:nch]
    c = rc[nch:]
    result = 0
    for k in range(nsamp):
        for j in range(nch):
            result += X[k, j]*np.log((1 + c[k]*r[j])*lam[k, j]) - (1 + c[k]*r[j])*lam[k, j]
    return -result


# Alternative log-likelihood function for log-likelihood change analyses.
def sample_f(X, mu, a, r, c):
    nsamp = X.shape[0]
    nch = X.shape[1]
    nsig = mu.shape[0]
    a = a.reshape(nsamp, nsig)
    lam = np.dot(a, mu) + 1
    result = np.zeros(nsamp)
    for k in range(nsamp):
        for j in range(nch):
            result[k] += X[k, j]*np.log((1 + c[k]*r[j])*lam[k, j]) - (1 + c[k]*r[j])*lam[k, j]
    return -result


def dfda(a, X, mu, r, c):
    nsamp = X.shape[0]
    nch = X.shape[1]
    nsig = mu.shape[0]
    a = a.reshape(nsamp, nsig)
    lam = np.dot(a, mu) + 1
    da = np.zeros((nsamp, nsig))
    for k in range(nsamp):
        for i in range(nsig):
            for j in range(nch):
                da[k, i] += X[k, j]*(mu[i, j]/lam[k, j]) - (1 + c[k]*r[j])*mu[i, j]
    return -da.reshape(-1,)


def dfdr(r, X, mu, a, c):
    nsamp = X.shape[0]
    nch = X.shape[1]
    nsig = mu.shape[0]
    a = a.reshape(nsamp, nsig)
    lam = np.dot(a, mu) + 1
    dr = np.zeros(len(r))
    for k in range(nsamp):
        for j in range(nch):
            dr[j] += (X[k, j]*c[k])/(1 + c[k]*r[j]) - c[k]*lam[k, j]
    return -dr


def dfdc(c, X, mu, a, r):
    nsamp = X.shape[0]
    nch = X.shape[1]
    nsig = mu.shape[0]
    a = a.reshape(nsamp, nsig)
    lam = np.dot(a, mu) + 1
    dc = np.zeros(len(c))
    for k in range(nsamp):
        for j in range(nch):
            dc[k] += (X[k, j]*r[j])/(1 + c[k]*r[j]) - r[j]*lam[k, j]
    return -dc


def dfdrc(rc, X, mu, a):
    nsamp = X.shape[0]
    nch = X.shape[1]
    nsig = mu.shape[0]
    a = a.reshape(nsamp, nsig)
    lam = np.dot(a, mu) + 1
    r = rc[:nch]
    c = rc[nch:]
    dr = np.zeros(len(r))
    dc = np.zeros(len(c))
    for k in range(nsamp):
        for j in range(nch):
            dr[j] += (X[k, j]*c[k])/(1 + c[k]*r[j]) - c[k]*lam[k, j]
            dc[k] += (X[k, j]*r[j])/(1 + c[k]*r[j]) - r[j]*lam[k, j]
    return -np.hstack([dr, dc])


def ml_inference_joint(counts, signatures, activities, full_labels, path, infer_activities=False, plot=True, rdf_out="r_all.csv"):
    labels = np.array([x.split("::")[0] for x in full_labels])

    r_ctype_df = pd.DataFrame(data=np.zeros((len(unique(labels)), 97)))
    r_ctype_df[0] = sorted(unique(labels))
    r_ctype_df = r_ctype_df.set_index(0)
    gauges = np.zeros(len(unique(labels)))

    opts = {'eta': 0.5,
            'stepmx': 500,
            'disp': True
            }

    for ctype in sorted(unique(labels)):
        print(ctype)
        ctidx = np.where(labels == ctype)[0]

        X = counts[ctidx] + 1
        mu = signatures
        acts = activities[ctidx]

        nsamp = X.shape[0]
        nch = X.shape[1]
        nsig = mu.shape[0]

        base_logliks = sample_f(X, mu, acts, np.zeros(nch), np.zeros(nsamp))

        if infer_activities:
            bounds_a = ()
            for i in range(1*nsig):
                bounds_a += ((0, 1000000), )

            inf_a = np.zeros((nsamp, nsig))
            new_logliks = np.zeros(nsamp)
            r0 = np.zeros(96)
            c0 = np.zeros(1)
            print("Inferring a...")
            for i in range(nsamp):
                # print(i)
                success = False
                Xs = X[i].reshape(1, -1)
                while not success:
                    a = np.random.uniform(size=(1, nsig), low=0, high=1000).reshape(1, -1)
                    res_a = scipy.optimize.minimize(fa, a, jac=dfda, args=(Xs, mu, r0, c0), method="TNC",
                                                    bounds=bounds_a, options=opts)
                    success = res_a.success
                    if success:
                        inf_a[i, :] = res_a.x.reshape(1, nsig)
                    else:
                        print("a inf:", str(i), res_a.message + ".", "Retrying...")

                loglik = sample_f(Xs, mu, inf_a[i], r0, c0)
                new_logliks[i] = loglik
            
            pd.DataFrame(inf_a).to_csv(path + ctype + "_a.csv", sep=",")
        else:
            inf_a = acts


        bounds_rc = ()
        for i in range(nch):
            bounds_rc += ((-1, 1), )
        for i in range(nsamp):
            bounds_rc += ((0, 10), )


        def rc_callback(rc):
            r = rc[:nch]
            c = rc[nch:]
            r = r/np.max(np.abs(r))
            c = c*np.max(np.abs(r))
            return np.hstack([r, c])


        a = inf_a
        success = False
        while not success:
            r = np.random.uniform(low=-1, high=1, size=nch)
            c = np.random.rand(nsamp)*np.max(np.abs(r))
            r = r/np.max(np.abs(r))
            rc = np.hstack([r, c])
            res_rc = scipy.optimize.minimize(frc, rc, jac=dfdrc, args=(X, mu, a), method="TNC",
                                            bounds=bounds_rc, options=opts, callback=rc_callback)
            success = res_rc.success
            if success:
                inf_r = res_rc.x[:nch]
                inf_c = res_rc.x[nch:]
            else:
                print("cr inf: " + res_rc.message + ".", "Retrying...")


        corr_logliks = sample_f(X, mu, inf_a, inf_r, inf_c)

        loglik_change = pd.DataFrame(data=(base_logliks, corr_logliks), index=("Before", "After")).T
        loglik_change.to_csv(path + ctype + "_ll_change.csv", sep=",")

        c_ctype_df = pd.DataFrame(data=np.zeros((inf_c.shape[0], 1)))
        c_ctype_df[0] = full_labels[ctidx]
        c_ctype_df = c_ctype_df.set_index(0)
        c_ctype_df[0] = inf_c
        c_ctype_df.to_csv(path + ctype + "_c.csv", sep=",")
        r_ctype_df.loc[ctype] = inf_r
        np.savetxt(path + ctype + "_gauges.txt", gauges, fmt='%.5f')

        if plot:
            print("Plotting the inferred modulatory process...")
            mut_barplot(r, title="", ylabel="Modulation", hline=0.0,
                        filename=path + ctype + "_mp.png", ylim=(-1.1, 1.1), annotate_types=True)
            print("Done.\n")

            print("Plotting the channel-wise scatter-plots...")
            channel_scatterplot_joint(X, np.dot(inf_a, mu).astype(int) + 1, r, c, title="",
                                      filename=path + ctype + "_repaired_scatter.png")
            print("Done.\n")

            print("Plotting the activity histogram...")
            fig = plt.figure(figsize=(8, 5))
            plt.hist(c, bins=20)
            plt.title(ctype + " multiplicative process activity histogram")
            plt.xlabel("Activity")
            plt.ylabel("Frequency")
            plt.savefig(path + ctype + "_activity_hist.png")
            plt.close(fig)
            print("Done.\n")

    r_ctype_df.to_csv(path + rdf_out, sep=",")
