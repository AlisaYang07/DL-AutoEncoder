from timeit import default_timer as timer
import numpy as np
import torch.nn as nn
import torch
import pandas as pd

# BCE_loss = nn.BCELoss()
def reconstruction_loss(x_hat, x):
    #return nn.BCELoss(reduction='sum')(x_reconstructed, x) / x.size(0)
    return nn.MSELoss(reduction='sum')(x_hat, x)

def kl_divergence_loss(mean, logvar):
    return - 0.5 * torch.sum(1+ logvar - mean.pow(2) - logvar.exp())

def loss_function(x, x_hat, mean, log_var, beta):
    # reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    #reproduction_loss = ((x - x_hat) **2).sum()
    # reproduction_loss = MSE_loss(x_hat, x)
    # KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    reproduction_loss = reconstruction_loss(x_hat, x)
    KLD = beta * kl_divergence_loss(mean, log_var)
    # KLD = 0
    return reproduction_loss + KLD

#https://www.machinelearningnuggets.com/how-to-generate-images-with-variational-autoencoders-vae-and-keras/


# def gaussian_likelihood(self, x_hat, logscale, x):
#         scale = torch.exp(logscale)
#         mean = x_hat
#         dist = torch.distributions.Normal(mean, scale)

#         # measure prob of seeing image under p(x|z)
#         log_pxz = dist.log_prob(x)
#         return log_pxz.sum(dim=(1, 2, 3))


def train(model,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=2,
          beta=1):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        #train_acc = 0
        #valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if next(model.parameters()).is_cuda:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output, mean, var = model(data)
            
            # Loss and backpropagation of gradients
            # loss = criterion(output, data)
            loss = loss_function(data, output, mean, var, beta)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)


            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:
            model.epochs += 1

            total_recon_loss = 0
            total_KL_loss = 0

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    if next(model.parameters()).is_cuda:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output, mean, var = model(data)
                    
                    # Loss and backpropagation of gradients
                    # loss = criterion(output, data)
                    recon_loss = reconstruction_loss(output, data)
                    KL_loss = beta * kl_divergence_loss(mean, var)
                    loss = loss_function(data, output, mean, var, beta)

                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item()
                    total_recon_loss += recon_loss
                    total_KL_loss += KL_loss

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)
                avg_recon_loss = total_recon_loss/ len(valid_loader.dataset)
                avg_KL_loss = total_KL_loss/ len(valid_loader.dataset)
                time = timer() - start

                history.append([train_loss, valid_loss, avg_recon_loss.item(), avg_KL_loss.item(), time])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f} \tReconstruction Loss: {avg_recon_loss:.4f} \tKL Loss: {avg_KL_loss:.4f} '
                    )


                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f}'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'avg_recon_loss', 'avg_KL_loss', 'time'])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f}'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'avg_recon_loss', 'avg_KL_loss', 'time'])
    return model, history