from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys

sys.path.append('..')

import src_v1.utils as utils 

weights_paths = "/home/sgurbuz/nasShare/projects/sgurbuz/dynamix_tryout/data_1d_vxlbl"


train_collapsed_per_electrode, test_collapsed_per_electrode, y_mean_collapsed_per_electrode, y_std_collapsed_per_electrode, mono_dom_list, test_probe_collapsed = utils.load_activation_data(weights_paths,ret_np=True)
train_full, test_full, y_mean_full, y_std_full, _, test_probe_full =  utils.load_activation_data(weights_paths,test_probe=test_probe_collapsed,collapsed=False,ret_np=True)

train_spec, test_spec, y_mean_spec, y_std_spec, mono_dom_list_spec, test_probe_spec =  utils.load_activation_data(
    weights_paths,
    collapsed=False,
    test_probe='BFvsALC__BFA_INM001_99bW06R09',
    ret_np=True)


T = train_full[0].shape[1]

def logreg_classifier(train, test):
    
    acc_over_t = []
    for t in range(T):
        X_train = train[0][:,t,:]
        X_test = test[0][:,t,:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        reg = LogisticRegression(max_iter=5000, C=1.0, class_weight='balanced',solver='saga')
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



acc_over_t_collapsed_over_electrode = logreg_classifier(train_collapsed_per_electrode,test_collapsed_per_electrode)
np.save('acc_over_t_collapsed_over_electrode.npy',np.array(acc_over_t_collapsed_over_electrode))
acc_over_t_full = logreg_classifier(train_full,test_full)
np.save('acc_over_t_full.npy',np.array(acc_over_t_full))
acc_full_controlled = logreg_classifier(train_spec,test_spec)
np.save('acc_full_controlled_BFvsALC__BFA_INM001_99bW06R09.npy',np.array(acc_full_controlled))
