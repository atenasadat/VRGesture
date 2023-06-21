
import time

from torch import optim

from data import Data

from model.speech2gesture import *

def train_loop(model_G,model_D,train_data, epoches, criterion,device='cpu'):
    for e in range(epoches):
        print("epoch {} of {}".format(e , epoches))
        for batch_idx, real_data in enumerate(train_data):
            real_data = real_data['audio/log_mel_512']
            print(real_data.shape)


            # Train the discriminator
            model_D.zero_grad()

            # Generate fake data and calculate discriminator loss on fake data
            noise = torch.randn(100, 64, 128)
            fake_data = model_G(noise)
            disc_fake_output = model_D(fake_data)
            disc_fake_loss = criterion(disc_fake_output, torch.zeros_like(disc_fake_output))

            # Calculate discriminator loss on real data
            real_data = real_data.to(device)
            disc_real_output = model_D(real_data)
            disc_real_loss = criterion(disc_real_output, torch.ones_like(disc_real_output))

            # Total discriminator loss and backpropagation
            disc_loss = disc_fake_loss + disc_real_loss
            disc_loss.backward()
            disc_optimizer.step()

            # Train the generator
            generator.zero_grad()

            # Generate fake data and calculate generator loss
            noise = torch.randn(batch_size, latent_dim, time, frequency)
            fake_data = model_G(noise)
            gen_output = model_D(fake_data)
            gen_loss = criterion(gen_output, torch.ones_like(gen_output))

            # Backpropagation
            gen_loss.backward()
            gen_optimizer.step()

            # Print the losses
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(data_loader)}], "
                      f"Discriminator Loss: {disc_loss.item():.4f}, Generator Loss: {gen_loss.item():.4f}")




data = Data(path2data='data/', speaker='minhaj')
data_train = data.train
data_dev = data.dev
data_test = data.test

print('Data Loaded')



generator = Speech2Gesture_G()
discriminator = Speech2Gesture_D()

# Define loss function and optimizers
criterion = nn.L1Loss()
gen_optimizer = optim.Adam(generator.parameters(), lr=0.001)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

train_loop(model_G=generator,model_D=discriminator,train_data=data_train,epoches=100,criterion=criterion)