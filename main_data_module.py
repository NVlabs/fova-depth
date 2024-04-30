import torch
import lightning as L
from torch.utils.data import DataLoader
from typing import Optional, List

class MainDataModule(L.LightningDataModule):
    def __init__(self, train_dataset: Optional[torch.utils.data.Dataset] = None, 
                       val_datasets: List[torch.utils.data.Dataset] = [],
                       test_datasets: List[torch.utils.data.Dataset] = [],
                       batch_size: int = 0, 
                       num_workers: int = 8,
                       num_validation_workers: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_validation_workers = num_validation_workers

        self.train_dataset = train_dataset
        self.val_datasets = val_datasets
        self.test_datasets = test_datasets

        if self.train_dataset is not None:
            print('train dataset size: ', len(train_dataset))
        for i, ds in enumerate(val_datasets):
            print('val dataset %d size: ' % i, len(ds))
        for i, ds in enumerate(test_datasets):
            print('test dataset %d size: ' % i, len(ds))

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=self.num_workers,shuffle=True,pin_memory=True)
        return loader
		
    def val_dataloader(self):
        loaders = []
        for ds in self.val_datasets:
             loaders.append( DataLoader(ds,batch_size=self.batch_size,num_workers=self.num_validation_workers,shuffle=False, pin_memory=True) )
        return loaders
    
    def test_dataloader(self):
        loaders = []
        for ds in self.test_datasets:
             loaders.append( DataLoader(ds,batch_size=self.batch_size,num_workers=self.num_validation_workers,shuffle=False, pin_memory=True) )
        return loaders
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
       
        if self.trainer.training and self.train_dataset.gpu_transforms is not None:
            batch = self.train_dataset.gpu_transforms(batch)
        elif self.trainer.validating and self.val_datasets[dataloader_idx].gpu_transforms is not None:
            batch = self.val_datasets[dataloader_idx].gpu_transforms(batch)
        elif self.trainer.sanity_checking and self.val_datasets[dataloader_idx].gpu_transforms is not None:
            batch = self.val_datasets[dataloader_idx].gpu_transforms(batch)
        elif self.trainer.testing and self.test_datasets[dataloader_idx].gpu_transforms is not None:
            batch = self.test_datasets[dataloader_idx].gpu_transforms(batch)
        
        return batch