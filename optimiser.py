import torch
from model import NumpyModel


class ClientOpt():
    """
    Client optimiser base class for use with FedAvg/AdaptiveFedOpt.
    """
    
    def get_params(self):
        """
        Returns:
            (NumpyModel) copy of all optimiser parameters.
        """
        raise NotImplementedError()
    
    def set_params(self, params):
        """
        Set all optimiser parameters.
        
        Args:
            - params: (NumpyModel) values to set
        """
        raise NotImplementedError()

    def get_bn_params(self, setting=0):
        """
        Return only BN parameters. Setting can be one of the following 
        {0: usyb, 1: yb, 2: us, 3: none} to get different types of parameters.
        
        Args:
            - setting (int) param types to get
            
        Returns:
            list of numpy.ndarrays
        """
        raise NotImplementedError()
        
    def set_bn_params(self, params, setting=0):
        """
        Set only BN parameters. Setting can be one of the following 
        {0: usyb, 1: yb, 2: us, 3: none} to get different types of parameters.
        
        Args:
            - params  (list) of numpy.ndarray values to set
            - setting (int) param types to get
        """
        raise NotImplementedError()


class ClientSGD(torch.optim.SGD, ClientOpt):
    """
    Client SGD optimizer for FedAvg and AdaptiveFedOpt.
    """

    def __init__(self, params, lr):
        super(ClientSGD, self).__init__(params, lr)
        
    def get_params(self):
        """
        Returns:
            (NumpyModel) copy of all optimiser parameters.
        """
        return NumpyModel([])
        
    def set_params(self, params):
        """
        Set all optimiser parameters.
        
        Args:
            - params: (NumpyModel) values to set
        """
        pass
        
    def get_bn_params(self, model, setting=0):
        """
        Vanilla SGD has no optimisation parameters. Returns empty list.
        
        Returns:
            [] empty list.
        """
        return []
        
    def set_bn_params(self, params, model, setting=0):
        """
        Vanilla SGD has no optimisation parameters. Does nothing.
        """
        pass
        
    def step(self, closure=None, beta=None):
        """
        SGD step. 
        
        Args: 
            - beta: (float) optional different learning rate.
        """
        loss = None
        if closure is not None:
            loss = closure

        # apply SGD update rule
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if beta is None:
                    p.data.add_(d_p, alpha=-group['lr'])
                else:     
                    p.data.add_(d_p, alpha=-beta)
        
        return loss