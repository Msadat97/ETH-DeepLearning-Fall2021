import torch 
import numpy as np
import dnnlib
import pathlib

from omegaconf import OmegaConf

from iin.networks import VectorTransformer, FactorTransformer
from iin.loss import Loss
from iin.dataset import TensorDataset
from get_latent_codes import tqdm_setup
import click


class UnsupervisedIINTrainer:
    def __init__(self, config):
        self.model = VectorTransformer(**config.model)
        self.dataset = dnnlib.util.construct_class_by_name(**config.dataset)
        self.loader = torch.utils.data.DataLoader(self.dataset, **config.data_loader)
        self.loss_fn = Loss()
        self.opt = torch.optim.Adam(self.model.parameters(), **config.optimizer)
        self.device = config.device
        self.model.to(self.device)
        self.base_dir = pathlib.Path(config.base_dir)
    
    def train_step(self, batch):
        zs = batch.to(self.device)
        output, logdet = self.model(zs)
        loss, loss_log = self.loss_fn(output, logdet)
            
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss, loss_log
    
    def train(self, n_epochs):
        for eopch in range(n_epochs):
            for idx, batch in tqdm_setup(self.loader, should_enumerate=True):
                loss, _ = self.train_step(batch)
            print(loss)
            torch.save(self.model.state_dict(), self.base_dir.joinpath("iin_state_dict.pt"))


class IINTrainer:
    def __init__(self, config):
        self.model: torch.nn.Module = dnnlib.util.construct_class_by_name(**config.model)
        self.dataset = dnnlib.util.construct_class_by_name(**config.dataset)
        self.loader = torch.utils.data.DataLoader(self.dataset, **config.data_loader)
        self.loss_fn = dnnlib.util.construct_class_by_name(**config.loss)
        self.opt = dnnlib.util.construct_class_by_name(self.model.parameters(), **config.optimizer)
        self.device = config.device
        self.model.to(self.device)
        self.base_dir = pathlib.Path(config.base_dir)
        if config.resume:
            self.model.load_state_dict(torch.load(config.checkpoint_path))
    
    def train_step(self, batch):
        factors, z1_vectors, z2_vectors = batch
        z1_vectors, z2_vectors = z1_vectors.to(self.device), z2_vectors.to(self.device)
        output1, logdet1 = self.model(z1_vectors)
        output2, logdet2 = self.model(z2_vectors)
        
        loss, loss_log = self.loss_fn(output1, output2, logdet1, logdet2, factors)
            
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss, loss_log
    
    def train(self, n_epochs):
        for eopch in range(n_epochs):
            for idx, batch in tqdm_setup(self.loader, should_enumerate=True):
                loss, _ = self.train_step(batch)
            print(loss)
            torch.save(self.model.state_dict(), self.base_dir.joinpath("iin_state_dict_factor_loss_wplus.pt"))


def main():
    config = OmegaConf.load("./iin/config.yaml")
    trainer = IINTrainer(config)
    trainer.train(**config.train_options)
  
  
if __name__ == "__main__":
    main()