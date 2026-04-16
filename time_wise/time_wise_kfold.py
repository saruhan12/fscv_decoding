from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
from tqdm import tqdm

sys.path.append('..')

import src_v1.utils as utils 

weights_paths = "/home/sgurbuz/nasShare/projects/sgurbuz/dynamix_tryout/data_1d_vxlbl/ALCIS_155_macroREF"

probes = ['ALCIS_155_macroREF__ALC1_INM001_01bW04R01M',
 'ALCIS_155_macroREF__ALC2_INM001_02bW02R01M',
 'ALCIS_155_macroREF__ALC3_INM001_03bW02R01M',
 'ALCIS_155_macroREF__ALC3_INM001_03bW06R02M',
 'ALCIS_155_macroREF__ALC4_INM001_04bW02R01M',
 'ALCIS_155_macroREF__ALC4_INM001_04bW06R02M',
 'ALCIS_155_macroREF__ALC4_INM001_04bW08R03M']

T = 1000

def logreg_classifier(train, test):
    
    acc_over_t = []
    for t in tqdm(range(T),unit='Time Step'):
        X_train = train[0][:,t,:]
        X_test = test[0][:,t,:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        reg = LogisticRegression(max_iter=10000, C=1.0, class_weight='balanced',solver='saga')
        reg.fit(X_train_s, train[1])

        acc_over_t.append(reg.score(X_test_s, test[1]))

    return acc_over_t


def k_fold_decoding(probe_list,model_dim='3d',collapsed=True):
    all_acc_curves = []
    for test_probe in probe_list:
        train, test, _, _, _, _ = utils.load_activation_data(weights_paths,
                    test_probe=test_probe,
                    ret_np=True,
                    model_dim=model_dim,
                    collapsed=collapsed
                    )
        acc_t = logreg_classifier(train, test)
        all_acc_curves.append(acc_t)

    return np.array(all_acc_curves)

#acc_over_t_collapsed_over_electrode = k_fold_decoding(probe_list=probes, model_dim=None)
#np.save('k_fold_acc_over_t_collapsed_over_electrode_6d.npy',acc_over_t_collapsed_over_electrode)
acc_over_t_full =  k_fold_decoding(probe_list=probes,collapsed=False)
np.save('k_fold_acc_over_t_full_3d.npy',acc_over_t_full)
