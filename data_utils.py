import gzip
import numpy as np
import torch

class PyTorchDataFeeder():
    """
    Contains data as torch.tensors. Allows easy retrieval of data batches and
    in-built data shuffling.
    """

    def __init__(   self, x, x_dtype, y, y_dtype, device, 
                    cast_device=None, transform=None):
        """
        Return a new data feeder with copies of the input x and y data. Data is 
        stored on device. If the intended model to use with the input data is 
        on another device, then cast_device can be passed. Batch data will be 
        sent to this device before being returned by next_batch. If transform is
        passed, the function will be applied to x data returned by next_batch.
        
        Args:
        - x:            x data to store
        - x_dtype:      torch.dtype or 'long' 
        - y:            y data to store 
        - y_dtype:      torch.dtype or 'long'
        - device:       torch.device to store data on 
        - cast_device:  data from next_batch is sent to this torch.device
        - transform:    function to apply to x data from next_batch
        """
        if x_dtype == 'long':
            self.x = torch.tensor(  x, device=device, 
                                    requires_grad=False, 
                                    dtype=torch.int32).long()
        else:
            self.x = torch.tensor(  x, device=device, 
                                    requires_grad=False, 
                                    dtype=x_dtype)
        
        if y_dtype == 'long':
            self.y = torch.tensor(  y, device=device, 
                                    requires_grad=False, 
                                    dtype=torch.int32).long()
        else:
            self.y = torch.tensor(  y, device=device, 
                                    requires_grad=False, 
                                    dtype=y_dtype)
        
        self.idx = 0
        self.n_samples = x.shape[0]
        self.cast_device = cast_device
        self.transform = transform
        self.shuffle_data()
        
    def shuffle_data(self):
        """
        Co-shuffle x and y data.
        """
        ord = torch.randperm(self.n_samples)
        self.x = self.x[ord]
        self.y = self.y[ord]
        
    def next_batch(self, B):
        """
        Return batch of data If B = -1, the all data is returned. Otherwise, a 
        batch of size B is returned. If the end of the local data is reached, 
        the contained data is shuffled and the internal counter starts from 0.
        
        Args:
        - B:    size of batch to return
        
        Returns (x, y) tuple of torch.tensors. Tensors are placed on cast_device
                if this is not None, else device.
        """
        if B == -1:
            x = self.x
            y = self.y
            self.shuffle_data()
            
        elif self.idx + B > self.n_samples:
            # if batch wraps around to start, add some samples from the start
            extra = (self.idx + B) - self.n_samples
            x = torch.cat((self.x[self.idx:], self.x[:extra]))
            y = torch.cat((self.y[self.idx:], self.y[:extra]))
            self.shuffle_data()
            self.idx = extra
            
        else:
            x = self.x[self.idx:self.idx+B]
            y = self.y[self.idx:self.idx+B]
            self.idx += B
            
        if not self.cast_device is None:
            x = x.to(self.cast_device)
            y = y.to(self.cast_device)

        if not self.transform is None:
            x = self.transform(x)

        return x, y


def data_split(x, y, W):
    """
    Shuffle x and y using the same random order, split into W parts.
    
    Args:
        - x: (np.ndarray) samples
        - y: (np.ndarray) labels
        - W: (int)        num parts to split into
            
    Returns:
        Tuple containing (list of x arrays, list of y arrays)
    """
    order = np.random.permutation(x.shape[0])
    x_split = np.array_split(x[order], W)
    y_split = np.array_split(y[order], W)
    
    return x_split, y_split


def load_dataset(data_dir, W):
    """
    Load the MNIST dataset.

    Args:
        - data_dir:  (str)  path to data folder
        - W:         (int)  number of workers to split dataset into
        
    Returns:
        Tuple containing ((x_train, y_train), (x_test, y_test)). The training 
        variables are both lists of length W, each element being a 2D numpy 
        array.
    """
    train_x_fname = data_dir + '/train-images-idx3-ubyte.gz'
    train_y_fname = data_dir + '/train-labels-idx1-ubyte.gz'
    test_x_fname = data_dir + '/t10k-images-idx3-ubyte.gz'
    test_y_fname = data_dir + '/t10k-labels-idx1-ubyte.gz'

    # load MNIST files
    with gzip.open(train_x_fname) as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0

    with gzip.open(train_y_fname) as f:
        y_train = np.copy(np.frombuffer(f.read(), np.uint8, offset=8))

    with gzip.open(test_x_fname) as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0  

    with gzip.open(test_y_fname) as f:
        y_test = np.copy(np.frombuffer(f.read(), np.uint8, offset=8))
    
    # split into iid/non-iid and users
    x_train, y_train = data_split(x_train, y_train, W)
    x_test, y_test = data_split(x_test, y_test, W)

    
    return (x_train, y_train), (x_test, y_test)

