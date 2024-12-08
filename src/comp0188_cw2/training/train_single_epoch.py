
import torch
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
from typing import Dict, Tuple
import logging
from tqdm import tqdm
from pymlrf.types import CriterionProtocol
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)
from pymlrf.types import (
    GenericDataLoaderProtocol
    )

class TrainSingleEpoch:
    
    def __init__(
        self, 
        half_precision:bool=False,
        cache_preds:bool=True
        ) -> None:
        """Class which runs a single epoch of training.

        Args:
            half_precision (bool, optional): Boolean defining whether to run in 
            half-precision. Defaults to False.
            cache_preds (bool, optional): Boolean defining whether to save the 
            prediction outputs. Defaults to True.
        """
        self.half_precision = half_precision
        self.cache_preds = cache_preds
        
    def __call__(
        self,
        model:torch.nn.Module,
        data_loader:GenericDataLoaderProtocol,
        gpu:bool,
        optimizer:torch.optim.Optimizer,
        criterion:CriterionProtocol,
        logger:logging.Logger
        )->Tuple[torch.Tensor, Dict[str,torch.Tensor]]:
        """ Call function which runs a single epoch of training
        Args:
            model (BaseModel): Torch model of type BaseModel i.e., it should
            subclass the BaseModel class
            data_loader (DataLoader): Torch data loader object
            gpu (bool): Boolean defining whether to use a GPU if available
            optimizer (torch.optim.Optimizer): Torch optimiser to use in training
            criterion (CriterionProtocol): Criterian to use for training or 
            for training and validation if val_criterion is not specified. I.e., 
            this could be nn.MSELoss() 
            logger (logging.Logger): Logger object to use for printing to terminal
        Raises:
            RuntimeError: Captures generic runtime errors that may occur during 
            training

        Returns:
            Tuple[torch.Tensor, Dict[str,torch.Tensor]]: Tuple defining the 
            final loss for the epoch and a dictionary of predictions. The keys 
            will be the same keys required by the criterion. 
        """
        losses = torch.tensor(0)
        denom = torch.tensor(0)
        if gpu:
            _device = "cuda"
        else:
            _device = "cpu"
            
        if self.half_precision:
            losses = losses.half()
            denom = denom.half()
        model.train()

        #addition to calculate RMSE and accuracy
        all_grp_pred = []
        all_grp_true = []
        all_pos_pred = []
        all_pos_true = []
        #addition to calculate RMSE and accuracy
            
        preds = []
            
        range_gen = tqdm(
            enumerate(data_loader),
            total=len(data_loader)
            #desc=f"Epoch {int(epoch)}/{epochs}",
            )
        for i, vals in range_gen:

            input_vals = vals.input
            output_vals = vals.output
            if gpu:
                input_vals = {
                    val:input_vals[val].cuda() for val in input_vals
                    }
                output_vals = {
                    val:output_vals[val].cuda() for val in output_vals
                    }
            else:
                input_vals = {val:Variable(input_vals[val]) for val in input_vals}
                output_vals = {val:Variable(output_vals[val])
                            for val in output_vals}

            optimizer.zero_grad()

            # Compute output
            if self.half_precision:
                with torch.autocast(device_type=_device):
                        output = model(**input_vals)
                        train_loss = criterion(output, output_vals)
            
            else:
                output = model(**input_vals)
                train_loss = criterion(output, output_vals)
                
            #addition to calculate RMSE and accuracy
             # --- Metric Calculation ---
            grp_pred = torch.argmax(output['grp'], dim=1)  # Get predicted class labels
            grp_true = torch.argmax(output_vals['actions'][:, 3:], dim=1).long()  # Get true class labels
            pos_pred = output['pos']  # Get predicted coordinates
            pos_true = output_vals['actions'][:, :3]  # Get true coordinates
            # --- End Metric Calculation ---

            # Accumulate predictions and ground truth
            all_grp_pred.extend(grp_pred.cpu().numpy())
            all_grp_true.extend(grp_true.cpu().numpy())
            all_pos_pred.extend(pos_pred.cpu().detach().numpy())
            all_pos_true.extend(pos_true.cpu().detach().numpy())
            #addition to calculate RMSE and accuracy

            
            if self.cache_preds:
                preds.append({k:output[k].detach().cpu() for k in output.keys()})
            losses += train_loss.detach().cpu()
            denom += 1
            # losses.update(train_loss.data[0], g.size(0))
            # error_ratio.update(evaluation(output, target).data[0], g.size(0))

            try:
                # compute gradient and do SGD step
                train_loss.backward()
                optimizer.step()
            except RuntimeError as e:
                logger.debug("Runtime error on training instance: {}".format(i))
        
                raise e
                
        #addition to calculate RMSE and accuracy
        # Calculate metrics after epoch
        acc = accuracy_score(all_grp_true, all_grp_pred)
        rmse = np.sqrt(mean_squared_error(all_pos_true, all_pos_pred))
        #addition to calculate RMSE and accuracy
            
        _prd_lst = {}
        if self.cache_preds:
            for k in preds[0].keys():
                _prd_lst[k] = torch.concat([t[k] for t in preds],dim=0)
                
        #addition to calculate RMSE and accuracy
        metrics = {'accuracy': acc, 'rmse': rmse} 
        #addition to calculate RMSE and accuracy

            
        losses = losses/denom
            
        return losses, _prd_lst, metrics
