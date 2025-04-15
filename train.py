import torch
import os
import torchvision.datasets as datasets 
from tqdm import tqdm
from torch import nn,optim
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader 


# Configurations

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM=784
H_DIM=512
Z_DIM=32
NUM_EPOCHS=30
BATCH_SIZE=64
LEARNING_RATE=1e-3
BETA=1.0
CHECKPOINT_DIR="checkpoints"
os.makedirs(CHECKPOINT_DIR,exist_ok=True)


# Dataset Loading

dataset=datasets.MNIST(root="dataset/",train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

model=VariationalAutoEncoder(INPUT_DIM,H_DIM,Z_DIM).to(DEVICE)
optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)
scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3,verbose=True)
loss_fn=nn.BCELoss(reduction="sum")

best_loss=float("inf")
best_model_path=os.path.join(CHECKPOINT_DIR,"best_model.pth")

# Training

for epoch in range(NUM_EPOCHS):
    epoch_loss=0.0
    loop=tqdm(enumerate(train_loader))
    for i,(x,_) in loop :
        # forward pass
        x=x.to(DEVICE).view(x.shape[0],INPUT_DIM)
        x_reconstructed,mu,log_var=model(x)

        # compute loss
        reconstruction_loss=loss_fn(x_reconstructed,x)
        kl_div=-0.5*torch.sum(1+log_var-mu.pow(2)-log_var.exp())
        loss=reconstruction_loss+BETA*kl_div


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()
        loop.set_postfix(loss=loss.item())
    avg_epoch_loss=epoch_loss/len(dataset)

    scheduler.step(avg_epoch_loss)

    if avg_epoch_loss<best_loss:
        best_loss=avg_epoch_loss
        torch.save(model.state_dict(),best_model_path)
        print(f"Saved best model at epoch {epoch+1} with loss: {avg_epoch_loss:.4f}")

print("Training complete!")


model.load_state_dict(torch.load(best_model_path))
model.eval()

with torch.no_grad():
    z=torch.randn(16,Z_DIM).to(DEVICE)
    generated=model.decode(z)
    generated=generated.view(-1,1,28,28)
    save_image(generated,"generated_images.png",nrow=4,normalize=True)



