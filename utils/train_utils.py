"""
Module containing utility classes and functions for training PyTorch models.
"""

from tqdm import tqdm

class Trainer:
    """
    Class to handle training and validation of a PyTorch model.
    Parameters:
    - model (torch.nn.Module): The PyTorch model to be trained.
    - optimizer (torch.optim.Optimizer): The optimizer used for training the model.
    - lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    - train_loader (torch.utils.data.DataLoader): DataLoader for training dataset.
    - val_loader (torch.utils.data.DataLoader): DataLoader for validation dataset.
    - device (torch.device): The device on which the model and data will be processed.
    Methods:
    - to_device(): Move the model to the specified device.
    - train(): Perform training of the model using the provided data.
    - valid(): Perform validation of the model using the provided data.
    """

    def __init__(self, model, optimizer, lr_scheduler, train_loader, val_loader, device):
        """
        Initialize the Trainer object.

        Parameters:
        - model (torch.nn.Module): The PyTorch model to be trained.
        - optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        - lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        - train_loader (torch.utils.data.DataLoader): DataLoader for training dataset.
        - val_loader (torch.utils.data.DataLoader): DataLoader for validation dataset.
        - device (torch.device): The device on which the model and data will be processed.
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.to_device()

    def to_device(self):
        """Move the model to the specified device."""    
        self.model.to(self.device)

    def train(self):
        """
        Perform training of the model using the provided data.

        Returns:
        - train_loss (float): Average training loss.
        """
        self.model.train()
        total_loss = []
        loop = tqdm(self.train_loader, leave=True)
        for images, targets in loop:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            #print("\n=====Loss_dict: ", loss_dict)
            losses = sum(loss for loss in loss_dict.values())
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            total_loss.append(losses.cpu().detach().numpy())
            # Update tqdm postfix with loss information
            loop.set_postfix(loss=losses.item())
        train_loss = sum(total_loss) / len(total_loss)
        return train_loss

    def valid(self):
        """
        Perform validation of the model using the provided data.
        
        Returns:
        - val_loss (float): Average validation loss.
        """
        self.model.eval()
        total_loss = []
        for _, (images, targets) in enumerate(self.val_loader):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            val_loss_dict, _ = self.model(images, targets)
            val_losses = sum(loss for loss in val_loss_dict.values())
            total_loss.append(val_losses.cpu().detach().numpy())
        val_loss = sum(total_loss) / len(total_loss)
        return val_loss
