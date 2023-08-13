import torch
import torch.nn as nn
import src.HyperParameters as hp
import src.model3 as model


# PINN_model = model.PINN()

# # Define your loss function and optimizer
# params = list(PINN_model.parameters())
# optimizer1 = torch.optim.Adam(params = params, lr = hp.learning_rate, amsgrad = True)   
def transform_sequence(tensor, seq_length):
    # tensor shape: (N, 2)
    # seq_length: desired sequence length
  

    # Add a new dimension of size 1 to the tensor
    tensor = torch.unsqueeze(tensor, dim=1)

    # Repeat the tensor along the added dimension to match the desired sequence length
    tensor = tensor.repeat(1, seq_length, 1)

    # tensor shape: (N, L, 2)
    return tensor

     

def train(train_loader, val_loader, epochs, optimizer, PINN_model, N):
    losses = {}
    f_train = {}
    vals = {}
    f_dist = {}
    f_test = {}
    # seq_len = N // hp.batch_size
    
    torch.autograd.set_detect_anomaly(True)
    for i in range(epochs):
        running_loss = 0.0
        for j, (X, Y, MF) in enumerate(train_loader):

<<<<<<< HEAD
                # X = transform_sequence(X, hp.seq_length)
                # Y = transform_sequence(Y, hp.seq_length)
                # MF = transform_sequence(MF, hp.seq_length)
=======
            X = transform_sequence(X, hp.seq_length)
            Y = transform_sequence(Y, hp.seq_length)
            MF = transform_sequence(MF, hp.seq_length)
>>>>>>> e4d3ce2871796c3baf74b5eb7a9b9028fc8e6fb8
    
            optimizer.zero_grad()    
                
<<<<<<< HEAD
                loss1 = PINN_model.Loss(X, Y, MF)
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(PINN_model.parameters(), max_norm=2.0, norm_type=2, error_if_nonfinite=False)
                optimizer.step()
                running_loss += loss1.item() 
                # ff = PINN_model.f.item()
                # ff = PINN_model.rnn(X)[:,:,3]
                ff = PINN_model.dnn(X)[:,3]
                f_dist[X[:,1]] = ff
=======
            loss1 = PINN_model.Loss(X, Y, MF)
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(PINN_model.parameters(), max_norm=2.0, norm_type=2, error_if_nonfinite=False)
            optimizer.step()
            running_loss += loss1.item() 
            # ff = PINN_model.f.item()
            ff = PINN_model.rnn(X)[:,:,3]
            f_dist[X[:,:,:-1]] = ff
>>>>>>> e4d3ce2871796c3baf74b5eb7a9b9028fc8e6fb8

            epoch_loss = running_loss/N 
            losses[i] = epoch_loss
            f_train[i] = torch.mean(ff)

            # Validation Step 
            with torch.no_grad():
                val_loss = 0.0
                for k, (A, B, C) in enumerate(val_loader):
                    # A = transform_sequence(A, seq_length=hp.seq_length)
                    # B = transform_sequence(B, seq_length=hp.seq_length)
                    # C = transform_sequence(C, seq_length=hp.seq_length)
                    u_pred = PINN_model.dnn(A)[:, :3]
                    
                    val_loss += torch.mean((u_pred-B)**2)
    
                    # f_pred = PINN_model.rnn(A)[:,:,3]
                    f_pred = PINN_model.dnn(X)[:,3]

                val_loss /= len(val_loader) # Average Validation Loss
                vals[i] = val_loss
                f_test[i] = torch.mean(f_pred)

        # Print the loss after each epoch
<<<<<<< HEAD
            if i == epochs-1:
                print(f"Epoch {i+1}/{epochs} - Train Loss: {losses[i]:.6f} Val Loss: {val_loss:.6f} f_train: {f_train[i]:.4f} f_test: {f_test[i]:.4f}")
            elif (i+1) % (epochs//5) == 0:
                print(f"Epoch {i+1}/{epochs} - Train Loss: {losses[i]:.6f} Val Loss: {val_loss:.6f} f_train: {f_train[i]:.4f} f_test: {f_test[i]:.4f}")
=======
        if i == epochs-1:
            print(f"Epoch {i+1}/{epochs} - Train Loss: {losses[i]:.6f} Val Loss: {val_loss:.6f} f_train: {f_train[i]:.4f} f_test: {f_test[i]:.4f}")
        elif (i+1) % (epochs//10) == 0:
            print(f"Epoch {i+1}/{epochs} - Train Loss: {losses[i]:.6f} Val Loss: {val_loss:.6f} f_train: {f_train[i]:.4f} f_test: {f_test[i]:.4f}")
    
>>>>>>> e4d3ce2871796c3baf74b5eb7a9b9028fc8e6fb8

    return losses, vals, f_train, f_test, f_dist

# train(train_loader, val_loader, hp.epochs)

