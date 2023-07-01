from data_utils import *
from model import *
from optimiser import *
import numpy as np
from tqdm import tqdm
import pickle
import torch
import gzip

SEED = 10
DEVICE = 'gpu'

def run_fedavg( data_feeders, test_data, model, client_opt,  
                T, M, K, B, test_freq=1, bn_setting=0, noisy_idxs=[]):
    """
    Run Federated Averaging (FedAvg) algorithm from 'Communication-efficient
    learning of deep networks from decentralized data', McMahan et al., AISTATS 
    2021. In this implementation, the parameters of the client optimisers are 
    also averaged (gives FedAvg-Adam when client_opt is ClientAdam). Runs T 
    rounds of FedAvg, and returns the training and test results.
    
    Args:
        - data_feeders: (list)      of PyTorchDataFeeders
        - test_data:    (tuple)     of (x, y) testing data as torch.tensors
        - model:        (FLModel)   to run SGD on 
        - client_opt:   (ClientOpt) distributed client optimiser
        - T:            (int)       number of communication rounds
        - M:            (int)       number of clients sampled per round
        - K:            (int)       number of local training steps
        - B:            (int)       client batch size
        - test_freq:    (int)       how often to test UA
        - bn_setting:   (int)       private: 0=ybus, 1=yb, 2=us, 3=none
        - noisy_idxs:   (iterable)  indexes of noisy clients (ignore their UA)
        
    Returns:
        Tuple containing (train_errs, train_accs, test_errs, test_accs) as 
        Numpy arrays of length T. If test_freq > 1, non-tested rounds will 
        contain 0's.
    """
    W = len(data_feeders)
    
    train_errs, train_accs, test_errs, test_accs = tuple(np.zeros(T, dtype=np.float32) for i in range(4))
    
    # contains private model and optimiser BN vals (if bn_setting != 3)
    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for w in range(W)]
    user_bn_optim_vals = [client_opt.get_bn_params(model) for w in range(W)]
    
    # global model/optimiser updated at the end of each round
    round_model = model.get_params()
    round_optim = client_opt.get_params()
    
    # stores accumulated client models/optimisers each round
    round_agg   = model.get_params()
    round_opt_agg = client_opt.get_params()
    
    for t in tqdm(range(T)):
        round_agg = round_agg.zeros_like()
        round_opt_agg = round_opt_agg.zeros_like()
        
        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)        
        weights   = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights   = weights.astype(np.float32)
        weights   /= np.sum(weights)
        
        round_n_test_users = 0
        
        for (w, user_idx) in zip(weights, user_idxs):
            # download global model/optim, update with private BN params
            model.set_params(round_model)
            client_opt.set_params(round_optim)
            model.set_bn_vals(user_bn_model_vals[user_idx], setting=bn_setting)
            client_opt.set_bn_params(user_bn_optim_vals[user_idx], 
                                        model, setting=bn_setting)
            
            # test local model if not a noisy client
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                err, acc = model.test(  test_data[0][user_idx], 
                                        test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1
            
            # perform local SGD
            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                err, acc = model.train_step(x, y)
                train_errs[t] += err
                train_accs[t] += acc

            # upload local model/optim to server, store private BN params
            round_agg = round_agg + (model.get_params() * w)
            round_opt_agg = round_opt_agg + (client_opt.get_params() * w)
            user_bn_model_vals[user_idx] = model.get_bn_vals(setting=bn_setting)
            user_bn_optim_vals[user_idx] = client_opt.get_bn_params(model,
                                                setting=bn_setting)
            
        # new global model is weighted sum of client models
        round_model = round_agg.copy()
        round_optim = round_opt_agg.copy()
        
        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
    
    train_errs /= M * K
    train_accs /= M * K
    
    print(f'train accuracy = {train_accs}\ntrain loss = {train_errs}')
    print("-----------------------------------------------------------")
    print(f'test accuracy = {test_accs}\ntest loss = {test_errs}')

    return train_errs, train_accs, test_errs, test_accs



def main(data_dir, worker_num=200, batch_size=16, learning_rate=0.1, epoch=50, client_selection_rate=0.5, fname='model.pkl'):
    """
    Run experiment specified by command-line args.
    """
        
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda:0' if DEVICE =='gpu' else 'cpu')

    # load data 
    print('Loading data...')
    train, test = load_dataset(data_dir=data_dir, W=worker_num)
    model       = MNISTModel(device)
    steps_per_E = int(np.round(60000 / (worker_num * batch_size)))
    
 
    # convert to pytorch tensors
    feeders   = [   PyTorchDataFeeder(x, torch.float32, y, 'long', device) for (x, y) in zip(train[0], train[1])]
    
    test_data = (   [torch.tensor(x, device=device, requires_grad=False, dtype=torch.float32) for x in test[0]],
                    [torch.tensor(y, device=device, requires_grad=False, dtype=torch.int32).long() for y in test[1]])
    

    # miscellaneous settings
    M                 = int(worker_num * client_selection_rate)
    K                 = steps_per_E
    
    # run experiment
    print('Starting experiment...')
    client_optim = ClientSGD(model.parameters(), lr=learning_rate)
    model.set_optim(client_optim)
    train_errs, train_accs, test_errs, test_accs = run_fedavg(  feeders, test_data, model, client_optim, epoch, M, 
                                                                K, batch_size, bn_setting=3)    
    output = {
        'model': model,
        'train_res': {
            'errs': train_errs,
            'accs': train_accs
        },
        'test_res': {
            'errs': test_errs,
            'accs': test_accs
        }
    }
    with open(fname, 'wb') as f:
        pickle.dump(output, f)

    print('Data saved to: {}'.format(fname))
    

if __name__ == '__main__':
    main(data_dir='../MNIST_data',
         worker_num=200,
         batch_size=16,
         learning_rate=0.1,
         epoch=100,
         client_selection_rate=0.5,
         fname='model.pkl')