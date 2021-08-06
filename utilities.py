import numpy as np

other_ctype_cols = {'Biliary-AdenoCa': '#00CD66',
                    'ALL': '#EEAD0E',
                    'Bone-cancer': '#F0EE60',
                    'AML': '#FFD700',
                    'Blood-CMDI': '#FFEC8B',
                    'Ewings': '#CDCB50',
                    'Eye-Melanoma': '#ADAC44',
                    'Breast-cancer': '#CD6090',
                    'Sarcoma': '#79CDCD',
                    'CNS-Medullo': '#D8BFD8',
                    'CNS-glioma-NOS': '#B0B0B0',
                    'CNS-GBM': '#3D3D3D',
                    'CNS-LGG': '#787878',
                    'ColoRect-AdenoCa': '#191970',
                    'Eso-AdenoCa': '#1E90FF',
                    'Sarcoma-bone': '#8B2323',
                    'Kidney-RCC': '#FF4500',
                    'Liver-HCC': '#006400',
                    'Lung-AdenoCa': '#FFFFFF',
                    'Lymph-BNHL': '#698B22',
                    'Lymph-CLL': '#F4A35D',
                    'Ovary-AdenoCa': '#008B8B',
                    'Panc-AdenoCa': '#7A378B',
                    'Panc-Endocrine': '#E066FF',
                    'Prost-AdenoCa': '#87CEFA',
                    'Skin-Melanoma': '#000000',
                    'Stomach-AdenoCa': '#BFEFFF'}

ctype_cols = {'Biliary-AdenoCA': '#00CD66',
              'Bladder-TCC': '#EEAD0E',
              'Bone-Benign': '#F0EE60',
              'Bone-Osteosarc': '#FFD700',
              'SoftTissue-Leiomyo': '#FFEC8B',
              'SoftTissue-Liposarc': '#CDCB50',
              'Bone-Epith': '#ADAC44',
              'Breast-AdenoCA': '#CD6090',
              'Cervix-SCC': '#79CDCD',
              'CNS-Medullo': '#D8BFD8',
              'CNS-PiloAstro': '#B0B0B0',
              'CNS-GBM': '#3D3D3D',
              'CNS-Oligo': '#787878',
              'ColoRect-AdenoCA': '#191970',
              'Eso-AdenoCA': '#1E90FF',
              'Head-SCC': '#8B2323',
              'Kidney-RCC': '#FF4500',
              'Kidney-ChRCC': '#B32F0B',
              'Liver-HCC': '#006400',
              'Lung-SCC': '#FDF5E6',
              'Lung-AdenoCA': '#FFFFFF',
              'Lymph-BNHL': '#698B22',
              'Lymph-CLL': '#F4A35D',
              'Myeloid-MPN': '#FFC100',
              'Myeloid-AML': '#CD6600',
              'Ovary-AdenoCA': '#008B8B',
              'Panc-AdenoCA': '#7A378B',
              'Panc-Endocrine': '#E066FF',
              'Prost-AdenoCA': '#87CEFA',
              'Skin-Melanoma': '#000000',
              'Stomach-AdenoCA': '#BFEFFF',
              'Thy-AdenoCA': '#9370DB',
              'Uterus-AdenoCA': '#FF8C69',
              'Breast-LobularCA': '#DDCDCD',
              'Breast-DCIS': '#DDCDCD',
              'Myeloid-MDS': '#DDCDCD',
              'Cervix-AdenoCA': '#DDCDCD'
              }


cancer_cols = {
    'Biliary': '#00CD66',
    'Bladder': '#EEAD0E',
    'Bone': '#F0EE60',
    'SoftTissue': '#FFEC8B',
    'Breast': '#CD6090',
    'Cervix': '#79CDCD',
    'CNS': '#D8BFD8',
    'ColoRect': '#191970',
    'Eso': '#1E90FF',
    'Head': '#8B2323',
    'Kidney': '#FF4500',
    'Liver': '#006400',
    'Lung': '#FDF5E6',
    'Lymph': '#698B22',
    'Myeloid': '#FFC100',
    'Ovary': '#008B8B',
    'Panc': '#7A378B',
    'Prost': '#87CEFA',
    'Skin': '#000000',
    'Stomach': '#BFEFFF',
    'Thy': '#9370DB',
    'Uterus': '#FF8C69'
}


cancer_organs_cols = {
    'Gastrointestinal': '#FF0000',
    'Hematologic': '#0000FF',
    'Endocrine': '#00FF00',
    'Musculosceletal': '#FFFF00',
    'Genitourinary': '#00FFFF',
    'Gynecologic': '#FF00FF',
    'Breast': '#C0C0C0',
    'Respiratory': '#800000',
    'Skin': '#000000',
    'Head and Neck': '#008000',
    'Neurologic': '#800080'
}


cancer_organs = {
    'Gastrointestinal': ['ColoRect', 'Biliary', 'Eso', 'Liver', 'Panc', 'Stomach'],
    'Hematologic': ['Lymph', 'Myeloid'],
    'Endocrine': ['Thy'],
    'Musculosceletal': ['Bone', 'SoftTissue'],
    'Genitourinary': ['Bladder', 'Kidney', 'Prost'],
    'Gynecologic': ['Cervix', 'Uterus', 'Ovary'],
    'Breast': ['Breast'],
    'Respiratory': ['Lung'],
    'Skin': ['Skin'],
    'Head and Neck': ['Head'],
    'Neurologic': ['CNS']
}


def is_number(x):
    """Check if x is a number."""
    try:
        float(x)
        return True
    except ValueError:
        return False


def unique(x):
    """Return unique elements in x."""
    return list(set(x))


def normalize_sum_1(x):
    """Normalize x so that all elements sum up to 1 (along rows if matrix)."""
    return x/x.sum(axis=1)[:, None]


def normalize_min_max(x):
    """Normalize x using min-max (along rows if matrix)."""
    return (x - np.min(x))/(np.max(x) - np.min(x))


def normalize_neg(x):
    """Normalize x to sum up to 1 on positive/negative signs separately."""
    x[x < 0] = x[x < 0]/(np.sum(np.abs(x[x < 0])))
    x[x >= 0] = x[x >= 0]/np.sum([x[x >= 0]])
    return x


def log10_abs_transform(x):
    """Transform x to log10 scale, keeping the sign."""
    log10_x = np.log10(np.abs(x))
    log10_x[x < 0] = log10_x[x < 0]*(-1)
    return log10_x


def get_idx(x):
    """Return ordered indices of each unique element in x."""
    idx = []
    for i in range(len(set(x))):
        cl = np.where(x == i)[0]
        for c in cl:
            idx.append(c)
    return idx
