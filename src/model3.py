import torch
import torch.nn as nn
import src.HyperParameters as hp


def RHS_f(y, baseflow, f):

    batch_len = y.shape[0]
    seq_len = y.shape[1]
    # pi_p, pi_m, sig = torch.unbind(y,axis=1)
    pi_p = y[:,:,0].reshape(y.shape[0]*y.shape[1], 1)
    pi_m = y[:,:,1].reshape(y.shape[0]*y.shape[1], 1)
    sig = y[:,:,2].reshape(y.shape[0]*y.shape[1], 1)

    D = baseflow[:,:,0]
    M = baseflow[:,:,1]
    u = baseflow[:,:,2]
    dMdx = baseflow[:,:,3]
    dudx = baseflow[:,:,4]
    alpha = baseflow[:,:,5]
    eta1 = baseflow[:,:,6]    
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
    calM = (dMdx/(2*M)).reshape(y.shape[0]*y.shape[1], 1)
    kp_p = ((gamma - 1) + (2/M)).reshape(y.shape[0]*y.shape[1], 1)
    kp_m = ((gamma - 1) - (2/M)).reshape(y.shape[0]*y.shape[1], 1)
    Gm_p = (M*(Ca*(M + 1) + Ff*M*(C1*gamma*M*Msq + M + (1 - C1*Msq)))/(2*denom)).reshape(y.shape[0]*y.shape[1], 1)
    Gm_m = (M*(Ca*(M - 1) - Ff*M*(C1*gamma*M*Msq + M - (1 - C1*Msq)))/(2*denom)).reshape(y.shape[0]*y.shape[1], 1)
    Ups = ((Ca*(Msq - 1) - C1*Ff*Msq*Msq *(1+gamma))/denom).reshape(y.shape[0]*y.shape[1], 1)
    
#     eq1 = - (2*np.pi*1j*He*vrh_p + Gm_m*kp_m + calM)*pi_p + (2*np.pi*1j*He*vkp_p + Gm_m*kp_p + calM)*pi_m - Gm_m*sig
#     eq2 = - (2*np.pi*1j*He*vkp_m + Gm_p*kp_m - calM)*pi_p - (2*np.pi*1j*He*vrh_m + Gm_p*kp_p + calM)*pi_m - Gm_m*sig
#     eq3 = - (2*np.pi*1j*He*vth_p + Ups*kp_m)*pi_p - (2*np.pi*1j*He*vth_m + Ups*kp_p)*pi_m - (2*np.pi*1j*He*vsig + Ups)*sig
#     eq1 = - tf.multiply((Gm_m*kp_m + calM),pi_p) + tf.multiply(( Gm_m*kp_p + calM),pi_m) - tf.multiply(Gm_m,sig)

    eq1 =  (Gm_m*kp_m + calM)*pi_p + ( Gm_m*kp_p - calM)*pi_m + Gm_m*sig
    eq2 =  (Gm_p*kp_m - calM)*pi_p + ( Gm_p*kp_p + calM)*pi_m + Gm_p*sig
    eq3 =  (Ups*kp_m)*pi_p + ( Ups*kp_p)*pi_m + (Ups)*sig
    
    gr = torch.cat((eq1, eq2, eq3), dim=1)
    GR = gr.unsqueeze(1).unsqueeze(2).expand(-1, batch_len, seq_len, -1)
    
    return GR


class PhyGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(PhyGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout= 0.5, batch_first = True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, x):

        out, _ = self.gru(x)
        out = self.activation(out)
        out = self.output_layer(out)

        return out
    
# Physics informed Neural Network 
class PINN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, lda):   
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
        self.rnn = PhyGRU(input_size, output_size, hidden_size, num_layers)
        'Register our new parameter'
        # self.dnn.register_parameter('f', self.f)  

        'Regularisation Constant'
        self.lda = lda

    def loss_data(self, X, UU):
        
        output = self.rnn(X)
        # x_pred, y_pred, z_pred = output[:,0], output[:,1], output[:,2]
        # loss_data = torch.mean((x_pred - UU[:,0]) ** 2 +
        #                   (y_pred - UU[:,1]) ** 2 +
        #                   (z_pred - UU[:,2]) ** 2)
        # loss_data = self.loss_function(output,UU)
        loss_data = torch.mean((output[:,:,0] - UU[:,:,0])**2 + (output[:,:,1] - UU[:,:,1])**2 + (output[:,:,2] - UU[:,:,2])**2)
        return loss_data
    
    def loss_residual(self, X, meanflow):
        output = self.rnn(X)
        r = output[:,:, :-1]
        f_pred = output[:,:,3]
        f = torch.mean(f_pred)
        GR = RHS_f(r, meanflow, f)
        dr_dn = torch.autograd.grad(r,X,torch.ones_like(r), retain_graph=True, create_graph=True)[0]
        pde = dr_dn[:,:, 1:] + GR
        
        return torch.mean(pde**2)
    
    def Loss(self, X, UU, meanflow):
        loss_data = self.loss_data(X, UU)
        loss_residual = self.loss_residual(X, meanflow)
        return loss_data + self.lda*loss_residual 

def transform_sequence(tensor, seq_length):
    # tensor shape: (N, 2)
    # seq_length: desired sequence length

    N, _ = tensor.shape
    transformed_sequences = []

    for i in range(N - seq_length + 1):
        sequence = tensor[i : i + seq_length]
        transformed_sequences.append(sequence)

    transformed_tensor = torch.stack(transformed_sequences)

    # transformed_tensor shape: (N - seq_length + 1, seq_length, 2)
    return transformed_tensor

# model = PINN(hp.input_size, hp.output_size, hp.hidden_size, hp.num_layers)
# a = torch.rand((10,2))
# b = torch.rand((10,4))
# c = torch.rand((10,7))
# a.requires_grad=True
# c.requires_grad=True
# # print(model.rnn(transform_sequence(a, seq_length=3)))
# # print(c)
# # tr = transform_sequence(a, seq_length=5)
# # print(tr)
# # print(tr[:,:,1])

# print(model.loss_data(transform_sequence(a, seq_length=4), transform_sequence(b, seq_length=4)))