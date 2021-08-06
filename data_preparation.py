import numpy as np
import pandas as pd


def prepare_data(file_path="./alexandrov_data/", save_to_path="./data/"):
    """
    Prepares all necessary datasets for analyses.

    Args:
        file_path (str): path where the original data is. The files are accessed by the names
                         as they appear in original study Synapse repository.
        save_to_path (str): where to save the processed data.
    """

    # PCAWG mutation counts
    pcawg_data = pd.read_csv(file_path + "WGS_PCAWG.96.csv")
    pcawg_data = pcawg_data.iloc[:, 2:].sort_index(axis=1)

    # Non-PCAWG mutation counts
    other_data = pd.read_csv(file_path + "WGS_Other.96.csv")
    other_data = other_data.iloc[:, 2:].sort_index(axis=1)

    # SigProfiler mutational signatures
    sp_signatures = pd.read_csv(file_path + "sigProfiler_SBS_signatures_2019_05_22.csv")
    sp_signatures['ID'] = sp_signatures.apply(lambda a: '{}_{}'.format(a['Type'], a['SubType']), axis=1)
    sp_signatures = sp_signatures.set_index('ID').drop(['Type', 'SubType'], axis=1)
    sp_signatures = sp_signatures.T

    # PCAWG SigProfiler activities
    sp_activities = pd.read_csv(file_path + "PCAWG_sigProfiler_SBS_signatures_in_samples.csv")
    sp_activities["ID"] = sp_activities["Cancer Types"] + "::" + sp_activities["Sample Names"]
    sp_activities = sp_activities.set_index("ID").drop(["Cancer Types", "Sample Names", "Accuracy"], axis=1)
    sp_activities.sort_index(inplace=True)

    # Non-PCAWG SigProfiler activities
    other_sp_activities = pd.read_csv(file_path + "nonPCAWG_WGS_sigProfiler_SBS_signatures_in_samples_2018_04_13.csv")
    other_sp_activities["ID"] = other_sp_activities["Cancer Types"] + "::" + other_sp_activities["Sample Names"]
    other_sp_activities = other_sp_activities.set_index("ID").drop(["Cancer Types", "Sample Names", "Accuracy"], axis=1)
    other_sp_activities.sort_index(inplace=True)

    # SignatureAnalyzer mutational signatures
    sa_signatures = pd.read_csv(file_path + "SignatureAnalyzer_SBS_W96.signature.031918.txt", sep="\t")
    sa_signatures.set_index("feature", inplace=True)
    sa_signatures.sort_index(inplace=True)
    sa_signatures = sa_signatures.T

    # PCAWG SignatureAnalyzer activities
    sa_activities = pd.read_csv(file_path + "SignatureAnalyzer_SNV.activity.FULL_SET.031918.txt", sep="\t").T
    new_columns = sa_activities.iloc[0]
    sa_activities = sa_activities[1:]
    sa_activities.columns = new_columns
    sa_activities.sort_index(inplace=True)
    sa_activities.index = [x.split("_")[0] + "-" + x.split("_")[1] + "::" + x.split("_")[3] for x in sa_activities.index]
    sa_activities.index.names = ["ID"]

    # Alexandrov fits
    sp_signatures = sp_signatures.loc[sp_activities.columns.values, ]
    pcawg_counts = pcawg_data.to_numpy().T
    other_counts = other_data.to_numpy().T
    sp_pcawg_fit = np.dot(sp_activities.to_numpy(), sp_signatures).astype(int)
    sp_other_fit = np.dot(other_sp_activities.to_numpy(), sp_signatures).astype(int)
    sa_pcawg_fit = np.dot(sa_activities.to_numpy(), sa_signatures).astype(int)
    pcawg_labels = [x.split("::")[0] for x in pcawg_data.columns.to_numpy()]
    pcawg_full_labels = pcawg_data.columns.to_numpy()
    other_labels = [x.split("::")[0] for x in other_data.columns.to_numpy()]
    other_full_labels = other_data.columns.to_numpy()

    # Save to files
    np.savetxt(save_to_path + "sigProfiler_signatures.txt", sp_signatures, fmt="%f")
    np.savetxt(save_to_path + "signatureAnalyzer_signatures.txt", sa_signatures, fmt="%f")
    np.savetxt(save_to_path + "sigProfiler_activities.txt", sp_activities, fmt="%f")
    np.savetxt(save_to_path + "signatureAnalyzer_activities.txt", sa_activities, fmt="%f")
    np.savetxt(save_to_path + "sigProfiler_other_activities.txt", other_sp_activities, fmt="%f")
    np.savetxt(save_to_path + "sigProfiler_signature_names.txt", sp_signatures.index.values, fmt="%s")
    

    np.savetxt(save_to_path + "real_pcawg_mutation_counts.txt", pcawg_counts, fmt='%i')
    np.savetxt(save_to_path + "real_other_mutation_counts.txt", other_counts, fmt='%i')
    np.savetxt(save_to_path + "alexandrov_sigProfiler_PCAWG_fitted_counts.txt", sp_pcawg_fit, fmt='%i')
    np.savetxt(save_to_path + "alexandrov_sigProfiler_other_fitted_counts.txt", sp_other_fit, fmt='%i')
    np.savetxt(save_to_path + "alexandrov_signatureAnalyzer_PCAWG_fitted_counts.txt", sa_pcawg_fit, fmt='%i')
    np.savetxt(save_to_path + "pcawg_full_labels.txt", pcawg_full_labels, fmt='%s')
    np.savetxt(save_to_path + "pcawg_labels.txt", pcawg_labels, fmt='%s')
    np.savetxt(save_to_path + "other_full_labels.txt", other_full_labels, fmt='%s')
    np.savetxt(save_to_path + "other_labels.txt", other_labels, fmt='%s')
