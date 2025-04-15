import torch
from torch import nn
from torch.nn import functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self,input_dim,h_dim=200,z_dim=20):
        super().__init__()
        # encoder
        self.img_2hid=nn.Linear(input_dim,h_dim)
        self.hid_2mu=nn.Linear(h_dim,z_dim)
        self.hid_2logvar=nn.Linear(h_dim,z_dim)

        # decoder
        self.z_2hid=nn.Linear(z_dim,h_dim)
        self.hdim_2img=nn.Linear(h_dim,input_dim)
        self.relu=nn.ReLU()


    def encode(self,x):
        h=self.relu(self.img_2hid(x))
        mu,log_var=self.hid_2mu(h),self.hid_2logvar(h)
        return mu,log_var

    def decode(self,z):
        h=self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hdim_2img(h))
    
    def reparameterize(self,mu,log_var):
        std=torch.exp(0.5*log_var)
        epsilon=torch.randn_like(std)
        return mu+epsilon*std
    
    def forward(self,x):
        mu,log_var=self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed=self.decode(z)
        return x_reconstructed,mu,log_var
    

if __name__=="__main__":
    x=torch.randn(4,28*28) # batch of 4 images with 784 pixels each (28x28 flattened)
    vae=VariationalAutoEncoder(input_dim=784)
    x_reconstructed,mu,sigma=vae(x)
    print(x_reconstructed.shape) # should be (4, 784)
    print(mu.shape) # should be (4, 20)
    print(sigma.shape) # should be (4, 20)
