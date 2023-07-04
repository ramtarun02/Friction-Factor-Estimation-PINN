import torch 
import scipy.io 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, dataset

import src.HyperParameters as hp


# data = scipy.io.loadmat('/Users/ramtarun/Desktop/Cambridge/Indirect-Noise-in-Nozzles/Data/Data_PINN_subsonic_geom_linvelsup_f0-0.1.mat')

def DataPreprocessing(data, ff):

    fmat = {0.1:data['sbsl_PINN'][0][0], 0.08:data['sbsl_PINN'][1][0], 0.06:data['sbsl_PINN'][2][0], 0.04:data['sbsl_PINN'][3][0], 0.02:data['sbsl_PINN'][4][0], 0.01:data['sbsl_PINN'][5][0], 0.005:data['sbsl_PINN'][6][0], 0.0:data['sbsl_PINN'][7][0]}
    fval = [0.1  , 0.08 , 0.06 , 0.04 , 0.02 , 0.01 , 0.005, 0.000]
    i = ff
    N = fmat[i].shape[1]
    U = torch.zeros((N,4))
    U[:,0] = torch.from_numpy(data['eta'])
    U[:,1] = torch.from_numpy(fmat[i][0,:])
    U[:,2] = torch.from_numpy(fmat[i][1,:])
    U[:,3] = torch.from_numpy(fmat[i][2,:])
    #U[:,4] = torch.from_numpy(np.array([0.06]))

    D = torch.from_numpy(data['D'])
    M = torch.from_numpy(data['M'])
    u = torch.from_numpy(data['ubar'])
    dMdx = torch.from_numpy(data['dMdx'])
    dudx = torch.from_numpy(data['dudx'])
    alpha = torch.from_numpy(data['alp'])
    eta1 = torch.from_numpy(data['eta'])   

    meanflow = torch.zeros((D.shape[1],7))
    meanflow[:,0] = D
    meanflow[:,1] = M
    meanflow[:,2] = u
    meanflow[:,3] = dMdx
    meanflow[:,4] = dudx
    meanflow[:,5] = alpha
    meanflow[:,6] = eta1
    #meanflow = torch.tile(meanflow, (8,1))

    he = torch.tensor([0.00])
    HE = torch.tile(he, (N,1)) 
    #He = HE.flatten()[:,None]

    uu = U[:, 1:4]
    uu0 = U[:,0]
    uu0 = uu0.flatten()[:, None]

    input_set = torch.cat([HE, uu0],1)

    return input_set, uu, meanflow

def DataTransformer(X, Y, Z, TrainingSet = True):
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(X, Y, Z,  train_size = hp.train_size, shuffle = True, random_state=45)
    x_train.requires_grad = True
    y_train.requires_grad = True
    z_train.requires_grad = True
    train_dataset = TensorDataset(x_train, y_train, z_train)
    test_dataset = TensorDataset(x_test, y_test, z_test)
    train_set, val_set = torch.utils.data.random_split(train_dataset, hp.train_val)
    train_loader = DataLoader(train_set, batch_size = hp.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size = hp.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = hp.batch_size, shuffle=False)

    if TrainingSet == True:
        return train_loader, val_loader
    else:
        return test_loader
