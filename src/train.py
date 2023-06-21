import torch



#PINN_model = model.PINN()

# Define your loss function and optimizer
#params = list(PINN_model.parameters())
#optimizer = torch.optim.Adam([{'params' : params[1::]},{'params' : params[0], 'lr': hp.ff_learning_rate}], lr = hp.learning_rate, amsgrad = True)   



def train(train_loader, val_loader, epochs, optimizer, PINN_model):
    losses = {}
    friction_f = {}
    vals = {}
    N = len(train_loader)
    for i in range(epochs):
            running_loss = 0.0
            for j, (X, Y, MF) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = PINN_model.Loss(X, Y, MF)
                loss.backward()
                optimizer.step(PINN_model.closure)
                running_loss += loss.item() 
                ff = PINN_model.f.item()
        
            epoch_loss = running_loss/N 
            losses[i] = epoch_loss
            friction_f[i] = ff

            # Validation Step 
            with torch.no_grad():
                val_loss = 0.0
                for k, (A, B, C) in enumerate(val_loader):
                    u_pred = PINN_model.dnn(A)
                    val_loss += torch.linalg.norm((B-u_pred),2)/torch.linalg.norm(B,2)
                #   val_loss += PINN_model.Loss(A, B, C).item()

                val_loss /= len(val_loader) # Average Validation Loss
                vals[i] = val_loss

        # Print the loss after each epoch
            if i == epochs-1:
                print(f"Epoch {i+1}/{epochs} - Train Loss: {losses[i]:.6f} Val Loss: {val_loss:.6f} f_PINN: {friction_f[i]}")
            elif (i+1) % (epochs//10) == 0:
                print(f"Epoch {i+1}/{epochs} - Train Loss: {losses[i]:.6f} Val Loss: {val_loss:.6f} f_PINN: {friction_f[i]}")


    return losses, vals, friction_f

# train(train_loader, val_loader, hp.epochs)