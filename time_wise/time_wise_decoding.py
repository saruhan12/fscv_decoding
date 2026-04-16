from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
from tqdm import tqdm

sys.path.append('..')

import src_v1.utils as utils 

weights_paths = "/home/sgurbuz/nasShare/projects/sgurbuz/dynamix_tryout/data_1d_vxlbl/ALCIS_155_macroREF"


#train_collapsed_per_electrode, test_collapsed_per_electrode, y_mean_collapsed_per_electrode, y_std_collapsed_per_electrode, mono_dom_list, test_probe_collapsed = utils.load_activation_data(weights_paths,test_probe="ALCIS_155_macroREF__ALC1_INM001_01bW04R01M",ret_np=True,model_dim=None)
train_full, test_full, y_mean_full, y_std_full, _, test_probe_full =  utils.load_activation_data(weights_paths,test_probe="ALCIS_155_macroREF__ALC1_INM001_01bW04R01M",collapsed=False,ret_np=True,model_dim=None)



T = train_full[0].shape[1]

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


def logreg_classifier_mean(train, test):
    
    X_train = train[0].mean(axis=1)
    X_test = test[0].mean(axis=1)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    reg = LogisticRegression(max_iter=1000, C=1.0,class_weight='balanced')
    reg.fit(X_train_s, train[1])

    return reg.score(X_test_s, test[1])



#acc_over_t_collapsed_over_electrode = logreg_classifier(train_collapsed_per_electrode,test_collapsed_per_electrode)
#np.save('acc_over_t_collapsed_over_electrode_6d.npy',np.array(acc_over_t_collapsed_over_electrode))
acc_over_t_full = logreg_classifier(train_full,test_full)
np.save('acc_over_t_full_6d.npy',np.array(acc_over_t_full))
