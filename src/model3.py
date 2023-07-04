import torch
import torch.nn as nn
import src.HyperParameters as hp

def RHS_f(y, baseflow, f):
    pi_p, pi_m, sig = torch.unbind(y,axis=1)

    D = baseflow[:,0]
    M = baseflow[:,1]
    u = baseflow[:,2]
    dMdx = baseflow[:,3]
    dudx = baseflow[:,4]
    alpha = baseflow[:,5]
    eta1 = baseflow[:,6]    
    
    He = 0 #a constant input

    gamma = 1.4 #constant
    
    Msq = torch.square(M)

    Lambda = 1 + Msq * (gamma-1)/2
    zeta = f*gamma*Msq - 2*torch.tan(alpha)
    C1= ((gamma - 1)*(1-Msq)*f)/(2*Lambda*zeta)
    Ca = -C1*M*u*dMdx*(2-(2*Msq/(1-Msq)) - (2*gamma*Msq*(-2*f*Lambda - (gamma-1)/gamma *zeta)/(2*Lambda*zeta)))
    Ff = -(dudx + (4*f*u/(2*D)))
    
    denom = (Msq*u - u + C1*Msq *u + C1*Msq*Msq*gamma*u) #M**4
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
    eq2 =  (Gm_p*kp_m - calM)*pi_p + ( Gm_p*kp_p + calM)*pi_m + Gm_p*sig
    eq3 =  (Ups*kp_m)*pi_p + ( Ups*kp_p)*pi_m + (Ups)*sig
    
    return torch.stack([-eq1, -eq2, -eq3], axis=1)


class PhyGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activations):
        super(PhyGRU, self).__init__()

        self.gru_layers = len(hp.hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.activation = nn.ModuleList(activations)
        self.gru = nn.GRU(input_size, self.hidden_sizes[0], batch_first=True)
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(self.hidden_sizes[0], self.hidden_sizes[0]))
        for i in range(1, self.gru_layers):
            self.hidden_layers.append(nn.Linear(self.hidden_sizes[i-1], self.hidden_sizes[i]))
        self.output_layer = nn.Linear(self.hidden_sizes[-1], output_size)

    def forward(self, x):
        gru_output, _ = self.gru(x)
        gru_output = self.activation[0](gru_output)
        
        for i, hidden_layer in enumerate(self.hidden_layers):
            gru_output = hidden_layer(gru_output)
            gru_output = self.activation[i](gru_output)
        
        output = self.output_layer(gru_output)
        return output
    
# Physics informed Neural Network 
class PINN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activations):   
        super(PINN, self).__init__()
        self.loss_function = nn.MSELoss()
        'Initialize our new parameters i.e.f (Inverse problem)' 
        # self.f = torch.tensor([f], requires_grad=True).type(torch.float32)
        # nn.init.zeros_(self.f)
        # 'Register f to optimize'
        # self.f = nn.Parameter(self.f)
        'Initialize iterator'
        self.iter = 0
        'Call our PIGRU'
        self.rnn = PhyGRU(input_size, output_size, hidden_sizes, activations)
        'Register our new parameter'
        # self.dnn.register_parameter('f', self.f)  

        'Regularisation Constant'
        self.lda = torch.tensor([hp.lda])

    def loss_data(self, X, UU):
        output = self.rnn(X)
        x_pred, y_pred, z_pred = output[:,0], output[:,1], output[:,2]
        loss_data = torch.mean((x_pred - UU[:,0]) ** 2 +
                          (y_pred - UU[:,1]) ** 2 +
                          (z_pred - UU[:,2]) ** 2)
                        
        return loss_data
    
    def loss_residual(self, X, meanflow):
        output = self.rnn(X)
        r = output[:,:3]
        f_pred = output[:,3]
        f = torch.mean(f_pred)
        GR = RHS_f(r, meanflow, f)
        dr_dn = torch.autograd.grad(r,X,torch.ones_like(r), retain_graph=True, create_graph=True)[0]
        pde = dr_dn[:, 1:] + GR
        return torch.mean(pde**2) 
    
    def Loss(self, X, UU, meanflow):
        loss_data = self.loss_data(X, UU)
        loss_residual = self.loss_residual(X, meanflow)
        return loss_data + hp.lda*loss_residual 
