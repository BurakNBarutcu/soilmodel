import torch
from pathlib import Path
from tqdm.auto import tqdm
from timeit import default_timer as timer


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        forward,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        print_every: int=10,
        schedular: bool = False,
        patience: int = 100,
    ) -> None:
        """
        Class that contains training functions of an ANN Model by using supervised learning.
        
        Parameters:
        -------------
            model: torch.nn.Module
                ANN Model defined as torch.nnn.Module
            forward:
                Forward method used to calculate the response of a system.
            optimizer: torch.optim.Optimizer
                Defined optimizer to calculate the gradients and update the parameters of the ANN model.
            loss_fn: torch.nn.Module
                Defined loss function to calculate the loss values.
            print_every: int = 10, Optional
                Number of eopchs to print the obtained loss values. Default value is 10.
            schedular: torch.optim.lr_schedular=None, Optional
                Boolean value to decide on whether to use "ReduceLROnPlateau" learning rate schedular or not. Default value is "False".
            patience: int=100, Optional
                Value for schedular to wait before updating the learning rate. If average loss value is not decreased after that number of epoch than learning rate is multiplied with 0.1. Default value is 100.
        """

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.print_every = print_every
        self.model = model
        self.forward = forward
        if schedular: self.schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',verbose=True,patience=patience)
        else: self.schedular = schedular

    def _run_batch(self,targets,labels):
        """
        Run one batch calculation and perform the trainig for the defined model by using the forward funtion defined on the Trainer class.
        
        Parameters:
        -------------
            targets:
                Inputs of the defined forward function.
            labels:
                Labeled data to use loss calculations of defined loss function. 
        
        Return:
        -------------
            loss: float
                Loss value obtained by defined loss function.
        """
        self.model.train()
        self.optimizer.zero_grad()
        output = self.forward(targets,self.model)
        loss = self.loss_fn(output,labels)
        loss.backward()
        self.optimizer.step()
        return loss

    def _run_epoch(self, data: torch.utils.data.DataLoader):
        """
        Run one epoch for the defined model by looping the batch calculations.
        
        Parameters:
        -------------
            data: torch.utils.data.DataLoader
                Dataloader that will be used as input to model as well as the input for the defined loss function.
        
        Return:
        -------------
            avg_loss: float
                Average loss value obtained by defined loss function.
        """
        avg_loss = 0
        for _, (targets, labels) in tqdm(enumerate(data),total=len(data),desc="Number of Batches", leave=True):
        # for batch, (targets, labels) in enumerate(data):
            avg_loss += self._run_batch(targets,labels)
        if self.schedular: self.schedular.step(avg_loss)
        avg_loss /= len(data)
        return avg_loss

    def _save_checkpoint(self,model_name:str, epoch:int=None, loss:float=None):
        """
        Saves the parameters of the ANN model defined on the Trainer Class.
        
        Parameters:
        -------------
            model_name: str
                Model name to use as the file name during the saving.
            epoch: int=None, Optional
                Epoch number of the training. Default value is "None". 
            loss: float=None, Optional
                Current loss value. Default value is "None".
        """
        print(f"Loss value decreased to {loss:.5f} at {epoch}.")
        best_model_info = {'model_state_dict': self.model.state_dict(),
                        'current_loss':loss,
                        'epoch': epoch}
        model_path = Path("Saved_Models")
        model_path.mkdir(parents=True, exist_ok=True)
        model_save_path = model_path / model_name
        print(f"Saving model to: {model_save_path}\n")
        torch.save(best_model_info,f=model_save_path)

    def train(self,train_data: torch.utils.data.DataLoader, max_epoch: float,
              terminate_loss: float = 1e-3, save_checkpoints: bool = False, model_name: str = None,
              evaluation: bool = False, eval_data: torch.utils.data.DataLoader = None, writer = None):
        """
        Setup the main training loops for any AI model for supervised learning.

        Parameters:
        -------------
            train_data: torch.utils.data.DataLoader
                Training dataloader
            max_epoch: float
                Maximum number of training loops
            terminate_loss: float = 0.001, Optional
                Minimum loss value as a goal to reach by the traning algorithm. Once it is reached training will terminated. Default is 0.001.
            save_checkpoints: bool = False, Optional
                Boolean value to decide on saving of the best model with the minimum loss value. Default is "False".
            model_name: str=None, Optional
                Name of the defined AI model as a file name of the checkpoint. Default value is "None".
            evaluation: bool = False, Optional
                Boolean value to decide on evaluation of the model with different data than the trainig data. Default is "False".
            eval_data: torch.utils.data.DataLoader = None, Optional
                Evaluation dataloader to calculate the evaluation loss. Default is "None".
            writer = None, Optional
                Summarywriter option to feed the data to Tensorboadr. Default is "None".
        
        Return:
        -------------
            Nothing return specifically. Sincel the ANN model is global. The trained model is the main return of the function.
        
        """
        max_epoch = int(max_epoch)+1
        train_start_time = timer()
        min_loss = float(1e36)
        for epoch in tqdm(range(max_epoch),desc=" Number of Epochs"):
            epoch_calc_start = timer()
            self.avg_loss = self._run_epoch(train_data)
            if evaluation: loss_eval = self.evaluation(eval_data)
            if writer:
                if evaluation: writer.add_scalars("Loss", {"Train Loss":self.avg_loss,"Test Loss":loss_eval},epoch)
                else: writer.add_scalars("Loss", {"Train Loss":self.avg_loss},epoch)
            if save_checkpoints:
                if self.avg_loss < min_loss:
                    min_loss = self.avg_loss
                    best_epoch = epoch
                    self._save_checkpoint(model_name,best_epoch,min_loss)
            if self.avg_loss <= terminate_loss:
                print(f"Epoch: {epoch} | Train Loss: {self.avg_loss:.3f}")
                print(f"Single Epoch Training Time: {epoch_calc_time:.3f} s")
                print(f"------------------------------------------------------------\n")
                print(f"Average Loss is lower than the defined Minimum Loss at {epoch}.!")
                print("Training is terminated.!")
                break
            epoch_calc_time = timer() - epoch_calc_start
            if epoch % self.print_every == 0:
                if evaluation: print(f"Epoch: {epoch} | Train Loss: {self.avg_loss:.3f} | Test Loss: {loss_eval:.3f}")
                else:
                    print(f"Epoch: {epoch} | Train Loss: {self.avg_loss:.3f}")
                    print(f"Single Epoch Training Time: {epoch_calc_time:.3f} s")
                    print(f"------------------------------------------------------------\n")
        print(f"Total Training Time: {timer()-train_start_time:.3f} s")
        if writer: writer.flush(), writer.close()

    def evaluation(self,eval_data: torch.utils.data.DataLoader,time_data = None):
        """
        Calculate the performance of the model without go into deep for the training loops.
        
        Parameters:
        -------------
            eval_data: torch.utils.data.DataLoader
                Evaluation Dataloader that contains inputs and the labels for the model.
        
        Return:
        -------------
            avg_loss: float
                Average loss value obtained by defined loss function.
        """
        self.model.eval()
        with torch.inference_mode():
            avg_loss = 0
            for _, (targets, labels) in tqdm(enumerate(eval_data),total=len(eval_data),desc="Number of Batches", leave=True):
            # for batch, (targets, labels) in enumerate(eval_data):
                output = self.forward(targets,self.model)
                loss_batch = self.loss_fn(output,labels)
                avg_loss += loss_batch
            avg_loss /= len(eval_data)
        return avg_loss




