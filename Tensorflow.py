import scipy.io
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras




def DataPreprocessing(data, ff):

    fmat = {0.1:data['sbsl_PINN'][0][0], 0.08:data['sbsl_PINN'][1][0], 0.06:data['sbsl_PINN'][2][0], 0.04:data['sbsl_PINN'][3][0], 0.02:data['sbsl_PINN'][4][0], 0.01:data['sbsl_PINN'][5][0], 0.005:data['sbsl_PINN'][6][0], 0.0:data['sbsl_PINN'][7][0]}
    fval = [0.1  , 0.08 , 0.06 , 0.04 , 0.02 , 0.01 , 0.005, 0.000]
    i = ff
    N = fmat[i].shape[1]
    U = np.zeros((N,4))
    U[:,0] = data['eta']
    U[:,1] = fmat[i][0,:]
    U[:,2] = fmat[i][1,:]
    U[:,3] = fmat[i][2,:]
    #U[:,4] = torch.from_numpy(np.array([0.06]))

    D = np.array(data['D'])
    M = np.array(data['M'])
    u = np.array(data['ubar'])
    dMdx = np.array(data['dMdx'])
    dudx = np.array(data['dudx'])
    alpha = np.array(data['alp'])
    eta1 = np.array(data['eta'])   

    meanflow = np.zeros((D.shape[1],7))
    meanflow[:,0] = D
    meanflow[:,1] = M
    meanflow[:,2] = u
    meanflow[:,3] = dMdx
    meanflow[:,4] = dudx
    meanflow[:,5] = alpha
    meanflow[:,6] = eta1
    #meanflow = torch.tile(meanflow, (8,1))

    he = np.array([0.00])
    HE = np.tile(he, (N,1)) 
    #He = HE.flatten()[:,None]

    uu = U[:, 1:4]
    uu0 = U[:,0]
    uu0 = uu0.flatten()[:, None]

    input_set = np.concatenate([HE, uu0],1)

    return input_set, uu, meanflow

data = scipy.io.loadmat('/Users/ramtarun/Desktop/Cambridge/Indirect-Noise-in-Nozzles/Data/Data_PINN_subsonic_geom_linvelsup_f0-0.1.mat')

inputs, targets, meanflow = DataPreprocessing(data, ff=0.01)


def RHS_ff_t(x,y, baseflow, f):
    eta = x
    pi_p, pi_m, sig = tf.unstack(y,axis=1)

    D = baseflow[:,0]
    M = baseflow[:,1]
    u = baseflow[:,2]
    dMdx = baseflow[:,3]
    dudx = baseflow[:,4]
    alpha = baseflow[:,5]
    eta1 = baseflow[:,6]    
    
    He = 0#a constant input

    gamma = 1.4#constant
    
    Msq = tf.math.square(M)

    Lambda = 1 + Msq * (gamma-1)/2
    zeta = f*gamma*Msq - 2*tf.math.tan(alpha)
    C1= ((gamma - 1)*(1-Msq)*f)/(2*Lambda*zeta)
    Ca = -C1*M*u*dMdx*(2-(2*Msq/(1-Msq)) - (2*gamma*Msq*(-2*f*Lambda - (gamma-1)/gamma *zeta)/(2*Lambda*zeta)))
    Ff = -(dudx + (4*f*u/(2*D)))

    denom = (M**2*u - u + C1*Msq *u + C1*Msq*Msq*gamma*u) #M**4
    vrh_p = M*(2*(1-M) + C1*M*(M-2+M*gamma*(1-2*M)))/(2*denom)
    vrh_m = -M*(2*(1+M) + C1*M*(M+2+M*gamma*(1+2*M)))/(2*denom)
    vkp_p = Msq*C1*(2+M*(gamma-1))/(2*denom)
    vkp_m = Msq*C1*(2-M*(gamma-1))/(2*denom)
    vth_p = C1*M*(M*(1+gamma) + M**2*(1-gamma) - 2)/denom
    vth_m = C1*M*(M*(1+gamma) - M**2*(1-gamma) + 2)/denom
    vsig = -(C1*Msq*(1+gamma*Msq) + Msq - 1)/denom
    calM = dMdx/(2*M)
    kp_p = (gamma - 1) + (2/M)
    kp_m = (gamma - 1) - (2/M)
    Gm_p = M*(Ca*(M + 1) + Ff*M*(C1*gamma*M*Msq + M + (1 - C1*Msq)))/(2*denom)
    Gm_m = M*(Ca*(M - 1) - Ff*M*(C1*gamma*M*Msq + M - (1 - C1*Msq)))/(2*denom)
    Ups = (Ca*(Msq - 1) - C1*Ff*Msq*Msq *(1+gamma))/denom
    
#     eq1 = - (2*np.pi*1j*He*vrh_p + Gm_m*kp_m + calM)*pi_p + (2*np.pi*1j*He*vkp_p + Gm_m*kp_p + calM)*pi_m - Gm_m*sig
#     eq2 = - (2*np.pi*1j*He*vkp_m + Gm_p*kp_m - calM)*pi_p - (2*np.pi*1j*He*vrh_m + Gm_p*kp_p + calM)*pi_m - Gm_m*sig
#     eq3 = - (2*np.pi*1j*He*vth_p + Ups*kp_m)*pi_p - (2*np.pi*1j*He*vth_m + Ups*kp_p)*pi_m - (2*np.pi*1j*He*vsig + Ups)*sig
    
#     eq1 = - tf.multiply((Gm_m*kp_m + calM),pi_p) + tf.multiply(( Gm_m*kp_p + calM),pi_m) - tf.multiply(Gm_m,sig)
    eq1 =  (Gm_m*kp_m + calM)*pi_p + ( Gm_m*kp_p - calM)*pi_m + Gm_m*sig
    eq2 =  ( Gm_p*kp_m - calM)*pi_p + ( Gm_p*kp_p + calM)*pi_m + Gm_m*sig
    eq3 =  ( Ups*kp_m)*pi_p + ( Ups*kp_p)*pi_m + (Ups)*sig
    



    return tf.stack([-eq1, -eq2, -eq3],axis=1)


def pde(r, n, baseflow, f):
    dr_dn = tf.gradients(r,n)
    # gr = RHS_ff_t(n, r, baseflow, f)
    # pde = dr_dn  + gr
    return dr_dn


def loss(inputs, outs, targets, baseflow, f):
    l1 = tf.keras.losses.MSE(targets, outs)
    l2 = pde(outs, inputs, baseflow, f)
    return l1+l2

### Model
model = keras.Sequential(name='PhyLSTM')
model.add(keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(None,2)))
model.add(keras.layers.Dense(3, activation='sigmoid')) 

# print(model.summary())
def transform_to_sequence(tensor, sequence_length):
    # Get the total number of samples in the tensor
    n = tf.shape(tensor)[0]

    # Calculate the batch size
    batch_size = n // sequence_length

    # Reshape the tensor to the desired shape
    transformed_tensor = tf.reshape(tensor[:batch_size * sequence_length], (batch_size, sequence_length, 2))

    return transformed_tensor

print(model.predict(transform_to_sequence(inputs, sequence_length=1)))