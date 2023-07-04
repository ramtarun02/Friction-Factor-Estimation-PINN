import torch
import src.HyperParameters as hp
import src.model as model


# PINN_model = model.PINN()

# # Define your loss function and optimizer
# params = list(PINN_model.parameters())
# optimizer1 = torch.optim.Adam(params = params, lr = hp.learning_rate, amsgrad = True)   




def train(train_loader, val_loader, epochs, optimizer, PINN_model, N):
    losses = {}
    f_train = {}
    vals = {}
    f_dist = {}
    f_test = {}
    # seq_len = N // hp.batch_size
    for i in range(epochs):
            running_loss = 0.0
            for j, (X, Y, MF) in enumerate(train_loader):
                optimizer.zero_grad()               
                loss1 = PINN_model.Loss(X, Y, MF)
                loss1.backward()
                optimizer.step()
                running_loss += loss1.item() 
                # ff = PINN_model.f.item()
                ff = PINN_model.rnn(X)[:,3]
                f_dist[X[:,1]] = ff

            epoch_loss = running_loss/N 
            losses[i] = epoch_loss
            f_train[i] = torch.mean(ff)

            # Validation Step 
            with torch.no_grad():
                val_loss = 0.0
                for k, (A, B, C) in enumerate(val_loader):
                    u_pred = PINN_model.rnn(A)[:, :3]
                    val_loss += torch.linalg.norm((B-u_pred),2)/torch.linalg.norm(B,2)
                #   val_loss += PINN_model.Loss(A, B, C).item()
                    f_pred = PINN_model.rnn(A)[:,3]

                val_loss /= len(val_loader) # Average Validation Loss
                vals[i] = val_loss
                f_test[i] = torch.mean(f_pred)

        # Print the loss after each epoch
            if i == epochs-1:
                print(f"Epoch {i+1}/{epochs} - Train Loss: {losses[i]:.6f} Val Loss: {val_loss:.6f} f_train: {f_train[i]:.4f} f_test: {f_test[i]:.4f}")
            elif (i+1) % (epochs//100) == 0:
                print(f"Epoch {i+1}/{epochs} - Train Loss: {losses[i]:.6f} Val Loss: {val_loss:.6f} f_train: {f_train[i]:.4f} f_test: {f_test[i]:.4f}")

    return losses, vals, f_train, f_test, f_dist

# train(train_loader, val_loader, hp.epochs)

