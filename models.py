# Create classes for models with best tuned parameters

import math
import time, datetime
import gc, sys, copy
import numpy as np
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn import feature_extraction
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.tree import DecisionTreeRegressor

import torch
import torch.utils.data as thdata

from data import all_fit_cols
from tools import now

####################
# Base class
####################

class base_model(object):
    '''
    Base class, just to be subclassed
    manages fit_cols selection and target clipping
    
    WARNING: it is assumed that X is a pandas dataframe with fitting colums 
    and y is a numpy ndarray with target values
    '''
    
    def __init__(self):
        self.name = 'Base'
        self.postclip = False
        self.target_range = None
        self.fit_cols = None
        self.trained = False
        self.model = None

    def _filter_cols(self, X):
        if self.fit_cols is not None:
            print('  ** fit columns selection',len(self.fit_cols),'vs',len(all_fit_cols(X)))
            return X[self.fit_cols]
        else:
            return X
        
    def _clip(self, preds):
        if self.target_range is not None:
            print(' ** predictions clipping to',self.target_range)
            return preds.clip(*self.target_range)
        else:
            return preds

    def _raw_fit(self, X, y):
        raise NotImplementedError("Subclass must implement abstract method")

    def fit(self, X, y):
        print(datetime.datetime.now())
        t_start = time.time()
        if self.postclip:
            res = self._raw_fit(self._filter_cols(X),y)
        else:
            res = self._raw_fit(self._filter_cols(X),self._clip(y))
        print("elapsed training time: %.2f" % (time.time() - t_start))
        return res

    def _raw_fit_early(self, Xt, yt, Xv, yv):
        raise NotImplementedError("Subclass must implement abstract method")

    def fit_early(self, Xt, yt, Xv, yv):
        '''
        Early stopping fit
        '''
        print(datetime.datetime.now())
        t_start = time.time()
        if self.postclip:
            res = self._raw_fit_early(self._filter_cols(Xt),yt,self._filter_cols(Xv),self._clip(yv))
        else:
            res = self._raw_fit_early(self._filter_cols(Xt),self._clip(yt),self._filter_cols(Xv),self._clip(yv))
        print("elapsed training time: %.2f" % (time.time() - t_start))
        return res
    
    def _raw_predict(self, X):
        raise NotImplementedError("Subclass must implement abstract method")

    def predict(self, X):
        assert self.trained == True
        t_start = time.time()
        res = self._clip(self._raw_predict(self._filter_cols(X)))
        print("elapsed infer time: %.2f" % (time.time() - t_start))
        return res
        
    def scoring(self, X_val, y_val, X_train=None, y_train=None):
        assert self.trained == True
    
        if X_train is not None:
            assert y_train is not None
            preds = self.predict(X_train)
            #print('Train R-squared for '+self.name+' is %f' % (r2_score(self._clip(y_train), preds)))
            print('Train RMSE for '+self.name+' is %f' % (rmse_score(self._clip(y_train), preds)))

        preds = self.predict(X_val)
        #print('Dev R-squared for '+self.name+' is %f' % (r2_score(self._clip(y_val), preds)))
        print('Dev RMSE for '+self.name+' is %f' % (rmse_score(self._clip(y_val), preds)))           

    def delete(self):
        if self.model is not None:
            del(self.model)
            gc.collect()
            self.model = None
            self.trained = False

####################
# NNRegressor
####################
    
class modelNNR(base_model):
    '''
    NNRegressor model with best parameters
    '''
    
    def __init__(self, target_range):
        self.name = 'NNR'
        self.postclip = False # Clip labels for fit
        self.target_range = target_range
        self.fit_cols = None # Better to use all features
        self.trained = False

        num_feat = 32
        n_hidden = 200
        nn_params = {
                'num_feat': num_feat, 
                'n_hidden1': n_hidden, 
                'n_hidden2': int(n_hidden/2), 
                'dropout': 0.2, 
                'optim_type': 'adam',
                'learning_rate': 1e-4, 
                'batch_size': 1024
            }
        self.model = NNRegressor(num_feat)
        self.model.set_params(**nn_params)
        print(self.model.get_params())
        
    def _raw_fit(self, X, y):
        # Train the model with fixed epochs (the best) and NO EARLY STOP
        max_epochs = 36
        e_losses, v_losses = self.model.train( 
                                           X.values,
                                           np.reshape(y,(y.shape[0],1)),
                                           None,
                                           None,
                                           max_epochs=max_epochs, 
                                           mid_validations=0, # faster train
                                           verbosity=0 )
        self.trained = True

    def _raw_fit_early(self, Xt, yt, Xv, yv):
        # Train the model with early stop
        max_epochs = 100
        e_losses, v_losses = self.model.train( 
                                           Xt.values,
                                           np.reshape(yt,(yt.shape[0],1)),
                                           Xv.values,
                                           np.reshape(yv,(yv.shape[0],1)),
                                           max_epochs=max_epochs, 
                                           mid_validations=0, # faster train
                                           verbosity=0 )
        self.trained = True
    
    def _raw_predict(self, X):
        return np.squeeze(self.model.predict(X.values))

    
####################
# Linear Regressor
####################
            
class modelLR(base_model):
    '''
    Basic Linear Regressor
    '''
                  
    def __init__(self, target_range):
        self.name = 'LR'
        self.postclip = True
        self.target_range = target_range
        self.fit_cols = None # Better to use all features
        self.trained = False

        self.model = LinearRegression()

    def _raw_fit(self, X, y):
        self.model = self.model.fit(X.values, y)
        self.trained = True

    def _raw_predict(self, X):
        return self.model.predict(X.values)

####################
# ElasticNet Regressor
####################
    
class modelEN(base_model):
    '''
    ElasticNet Regressor model with best parameters
    '''
                  
    def __init__(self, target_range):
        self.name = 'ElasticNet'
        self.postclip = True
        self.target_range = target_range
        self.fit_cols = None # Better to use all features
        self.trained = False

        #self.alpha, self.l1_ratio = 0.404, 1.0 #CV search, but worsens RMSE...
        self.alpha, self.l1_ratio = 1.0, 0.5

        self.max_iter = 1000
        self.en_params = {
                 'max_iter': self.max_iter,
                 'random_state': 0, # Changing this could generate ensembling options
    
                 'alpha': self.alpha,
                 'l1_ratio': self.l1_ratio,

                 'tol': 0.0001,
                 'fit_intercept': True,
                 'normalize': False,
                 'positive': False,
                 'precompute': False,
                 'selection': 'cyclic',
                 'copy_X': True,
                 'warm_start': False
            }

        self.model = ElasticNet()
        self.model = self.model.set_params(**self.en_params)
                  
    def _raw_fit(self, X, y):
        self.model = self.model.fit(X.values, y)
        self.trained = True

    def _raw_predict(self, X):
        return self.model.predict(X.values)

####################
# DecissionTree Regressor
####################
    
class modelDTR(base_model):
    '''
    Decission Tree Regressor model with best parameters
    '''
                  
    def __init__(self, target_range):
        self.name = 'DecissionTree'
        self.postclip = False
        self.target_range = target_range
        self.fit_cols = None # Better to use all features
        self.trained = False

        self.dtr_params = {
                'random_state': 0, # Changing this could generate ensembling options

                'max_depth': None,
                'max_features': None,
                'max_leaf_nodes': None,
                'min_impurity_decrease': 0.0,
                'min_impurity_split': None,
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'min_weight_fraction_leaf': 0.0,

                'presort': False,
                'splitter': 'best',
                'criterion': 'mse'
        }

        self.model = DecisionTreeRegressor()
        self.model = self.model.set_params(**self.dtr_params)
                  
    def _raw_fit(self, X, y):
        self.model = self.model.fit(X.values, y)
        self.trained = True

    def _raw_predict(self, X):
        return self.model.predict(X.values)
    
####################
# LightGBM
####################
    
class modelLGB(base_model):
    '''
    LightGBM model with best parameters
    '''
    
    def __init__(self, target_range):
        self.name = 'LightGBM'
        self.postclip = False # Clip labels for fit
        self.target_range = target_range
        self.fit_cols = None # Better to use all features
        self.trained = False

        self.lgb_lr = 0.03
        self.lgb_iterations = 110
        self.lgb_params = {
               'feature_fraction': 0.75,
               'metric': 'rmse',
               'nthread':1, 
               'min_data_in_leaf': 2**7, 
               'bagging_fraction': 0.75, 
               'learning_rate': self.lgb_lr, 
               'objective': 'mse', 
               'bagging_seed': 2**7, 
               'num_leaves': 2**7,
               'bagging_freq':1,
               'verbose':0 
              }
        self.model = None # later creation
        
    def _raw_fit(self, X, y):
        self.model = lgb.train(self.lgb_params, 
                               lgb.Dataset(X, label=y), 
                               num_boost_round=self.lgb_iterations)
        self.trained = True

    def _raw_predict(self, X):
        return self.model.predict(X)


####################
# Helpers
####################

def rmse_score(y_true,y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

# Function to score to test and validation
def scoring(model_name, scoring_model, target_range,
            X_train, y_train, X_val, y_val, values=True, cols=None, preds_transform=None):
    '''
    Score to test and validation data
    '''

    if cols is not None:
        S_train = X_train[cols]
        S_val = X_val[cols]
    else:
        S_train = X_train
        S_val = X_val

    if values:
        S_train = S_train.values
        S_val = S_val.values

        
    preds = scoring_model.predict(S_train).clip(*target_range)
    if preds_transform is not None:
        preds = preds_transform(preds)
    #print('Train R-squared for '+model_name+' is %f' % (r2_score(y_train.clip(*target_range), preds)))
    print('Train RMSE for '+model_name+' is %f' % (rmse_score(y_train.clip(*target_range), preds)))

    preds = scoring_model.predict(S_val).clip(*target_range)
    if preds_transform is not None:
        preds = preds_transform(preds)
    #print('Dev R-squared for '+model_name+' is %f' % (r2_score(y_val.clip(*target_range), preds)))
    print('Dev RMSE for '+model_name+' is %f' % (rmse_score(y_val.clip(*target_range), preds))) 

def grid_search_tuning(estimator, param_grid, X_train, y_train, X_val, y_val):
    '''
    Model cross validation tuning with grid search
    '''
    
        # Prepare cross validation data
    X = X_train.append(X_val)
    Y = np.concatenate([y_train, y_val])
    train_ind=np.zeros(X.shape[0])
    for i in range(0, len(X_train)):
        train_ind[i] = -1
    ps = PredefinedSplit(test_fold=(train_ind))

    # Do grid search
    gs = GridSearchCV(cv = ps, 
                      estimator = estimator, 
                      param_grid = param_grid, 
                      scoring='neg_mean_squared_error')
    gs.fit(X, Y)

    # Get best estimator parameters
    best_model = gs.best_estimator_
    best_params = best_model.get_params()
    print(best_params)
    
    return best_params, best_model               

def compare_models(m1, m2, X_train, y_train, X_val, y_val, m1_early=False, m2_early=False):
    '''
    Plot scatter for two models prediction, so check if are uncorrelated
    '''

    # Fit
    print("\nFit "+m1.name)
    if m1_early:
        m1.fit_early(X_train,y_train,X_val,y_val)
    else:
        m1.fit(X_train,y_train)
    m1.scoring(X_val,y_val)
    print("\nFit "+m2.name)
    if m2_early:
        m2.fit_early(X_train,y_train)
    else:
        m2.fit(X_train,y_train)
    m2.scoring(X_val,y_val)
    
    # Prepare plot
    fig, (ax1, ax2) = plt.subplots(1,2)
    
    # Show predictions correlation for train
    print('\nPredicting for train...')
    preds1 = m1.predict(X_train)
    preds2 = m2.predict(X_train)
    p = ax1.scatter(preds1,preds2)
    p = ax1.set_title('TRAIN correlation')
    p = ax1.set_xlabel(m1.name+' preds')
    p = ax1.set_ylabel(m2.name+' preds')

    # Show predictions correlation for dev
    print('\nPredicting for dev...')
    preds1 = m1.predict(X_val)
    preds2 = m2.predict(X_val)
    p = ax2.scatter(preds1,preds2,color='red')
    p = ax2.set_title('DEV correlation')
    p = ax2.set_xlabel(m1.name+' preds')
    p = ax2.set_ylabel(m2.name+' preds')

    fig.set_size_inches(16.5, 8.5)
    plt.show()

####################
# Naive two layer NN model to optimize
####################

class NNRegressor(object):
    '''
    Two layer NN Adam MES with early stopping, to optimize
    '''
    
    def __init__(self, num_feat, n_hidden1=100, n_hidden2=50, dropout=0.2, 
                 optim_type='adam', learning_rate=0.001, batch_size=1024):
        '''
        num_feat = X.shape[1]
        n_hidden1, n_hidden2 = network size, the size depends of the data to learn from, 
                                big size implies more learning capacity
                                but the bigger the more processing power and more to overfit prone
        dropout = fraction of layers to dropout, regularization parameter
        optim_type = the optimizer type
        learning_rate = the optimizer learning rate
        batch_size = number of samples in a batch
        '''
        self.name = 'NN'
        self.trained = False
        self.trained_epochs = 0
        self.epoch_losses, self_val_losses = [], []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self._do_graph(num_feat, n_hidden1, n_hidden2, dropout, optim_type, learning_rate, batch_size)

    def _do_graph(self, num_feat, n_hidden1, n_hidden2, dropout, optim_type, learning_rate, batch_size):
        assert optim_type in ['adam', 'sgd', 'asgd']
        
        # define the network, two hidden layers

        self.num_feat, self.n_hidden1, self.n_hidden2, self.dropout = num_feat, n_hidden1, n_hidden2, dropout
        self.net = torch.nn.Sequential(
                    torch.nn.Linear(num_feat, n_hidden1),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Linear(n_hidden1, n_hidden2),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Linear(n_hidden2, 1),
                )
        if self.device == 'cuda':
            print("model in GPU")
            self.net = self.net.cuda()

        # fit elements
        
        self.optim_type, self.learning_rate, self.batch_size = optim_type, learning_rate, batch_size
        if optim_type == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        if optim_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)
        if optim_type == 'asgd':
            self.optimizer = torch.optim.ASGD(self.net.parameters(), lr=learning_rate)
        self.loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss        
        
    def get_params(self):
        return {
               'num_feat': self.num_feat,
               'n_hidden1': self.n_hidden1,
               'n_hidden2': self.n_hidden2, 
               'dropout': self.dropout, 
               'optim_type': self.optim_type,
               'learning_rate': self.learning_rate,
               'batch_size': self.batch_size,
              }
        if self.trained:
            print("trained with",self.trained_epochs,"epochs")

    def set_params(self, num_feat=None, n_hidden1=100, n_hidden2=50, dropout=0.2, 
                   optim_type='adam', learning_rate=0.001, batch_size=1024):
        assert not self.trained
        if num_feat is None:
            num_feat = self.num_feat
        self._do_graph(num_feat, n_hidden1, n_hidden2, dropout, optim_type, learning_rate, batch_size)
    
    def _prepare_feed(self, X_train, y_train, X_val, y_val, mid_validations, num_workers=4):
        
        assert isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray)
        assert len(X_train.shape) == 2 and len(y_train.shape) == 2
        assert X_train.shape[1] == self.num_feat and y_train.shape[1] == 1
        assert X_train.shape[0] == y_train.shape[0]

        num_train_samples = X_train.shape[0]
        num_train_steps_in_epoch = int(num_train_samples/self.batch_size)
        assert num_train_steps_in_epoch > mid_validations
        
        if X_val is not None:
            assert isinstance(X_val, np.ndarray) and isinstance(y_val, np.ndarray)
            assert len(X_val.shape) == 2 and len(y_val.shape) == 2
            assert X_val.shape[1] == self.num_feat and y_val.shape[1] == 1
            assert X_val.shape[0] == y_val.shape[0]

            num_val_samples = X_val.shape[0]
            assert num_train_samples > num_val_samples
        
        # train data in batches 
        
        trainset = thdata.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        trainloader = thdata.DataLoader(trainset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        # validation data usually is not too big, allocate in GPU if available and no batching
        
        if X_val is None or y_val is None:
            X_v = None
            y_v = None
        else:
            X_v = torch.from_numpy(X_val).float().to(self.device)
            y_v = torch.from_numpy(y_val).float().to(self.device)    
        
        return trainloader, X_v, y_v
        
    def _autoclone(self):
        self.net.cpu()
        new_net = copy.deepcopy(self.net)
        self.net.to(self.device)
        return new_net
        
    def train(self, X_train, y_train, X_val, y_val,
                 max_epochs=100, mid_validations=0, verbosity=0):
        '''
        X_train = numpy array with size (num_samples, num_feat)
        y_train = numpy array with size (num_samples, 1)
        X_val, y_val are the same but with num_validation_samples
        
        max_epochs = maximum number of epochs, if early stopping is not trigered train to that
        mid_validations = number of validations in the mid of every epoch (informative)
        verbosity = 0 less, 2 more, if verbosity < 2 mid_validatios is ignored
        '''
        
        trainloader, X_v, y_v = self._prepare_feed(X_train, y_train, X_val, y_val, mid_validations)
        
        self.trained = True
        print('\n%s ---- START training' % (now()))
        if verbosity < 1:
            print(".",end=" ")
            sys.stdout.flush()
            
        val_step = -1
        if mid_validations > 0:
            val_step = int(len(trainloader)/mid_validations) 
        epoch_losses = []
        val_losses = []
        
        early_min_epochs = 6
        early_compare_to = 3
        last_nets = []
        self.early_stop = False

        for epoch in range(max_epochs):

            # train batches
            self.net.train()
            step = -1
            running_losses = []
            for inputs, labels in trainloader:
                step +=1
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                prediction = self.net(inputs)
                loss = self.loss_func(prediction, labels)

                self.optimizer.zero_grad()   # clear gradients for next train
                loss.backward()              # backpropagation, compute gradients
                self.optimizer.step()        # apply gradients

                t_loss = loss.cpu().data.numpy()
                running_losses.append(t_loss)

                if step % val_step == 0 and verbosity > 1 and val_step > 0 and X_v is not None:
                    with torch.no_grad():
                        self.net.eval()
                        # informative mid-epoch validation
                        predval = self.net(X_v)
                        lossval = self.loss_func(predval, y_v)
                        v_loss = lossval.cpu().data.numpy()
                        # show learning process
                        print('%s Epoch = %d/%d Step = %d/%d Loss = %.4f ValLoss = %.4f' % (now(),
                                                                            epoch, max_epochs-1, step, len(trainloader)-1, t_loss, v_loss))

            # end of epoch
            e_loss = sum(running_losses)/len(trainloader)
            epoch_losses.append(e_loss)
            if X_v is None:
                if verbosity > 0:
                    print('%s ---- END Epoch = %d/%d AvgLoss = %.4f' % (now(), epoch, max_epochs-1, e_loss))
                else:
                    print(".",end=" ")
                    sys.stdout.flush()
            else:
                with torch.no_grad():
                    self.net.eval()
                    # validation
                    predval = self.net(X_v)
                    lossval = self.loss_func(predval, y_v)
                    v_loss = lossval.cpu().data.numpy()
                    val_losses.append(v_loss)
                    if verbosity > 0:
                        print('%s ---- END Epoch = %d/%d AvgLoss = %.4f ValLoss = %.4f' % (now(), epoch, max_epochs-1, e_loss, v_loss))
                    else:
                        print(".",end=" ")
                    # very basic early stopping
                    last_nets.append(self._autoclone())
                    if len(val_losses) > early_min_epochs:
                        if v_loss > val_losses[-early_compare_to]:
                            if verbosity < 1:
                                print("")
                            print("%s *** Early stop at epoch %d/%d, best_ValLoss = %.4f" % (now(), epoch, max_epochs-1, val_losses[-3]))
                            # retrieve the network 2 steps before, the asssumed best
                            self.net = last_nets[-early_compare_to].to(self.device)
                            self.early_stop = True
                            break
                    if len(last_nets) > early_compare_to:
                        last_nets.pop(0)

        self.trained_epochs = epoch + 1
        self.epoch_losses, self.val_losses = epoch_losses, val_losses
        if verbosity < 1 and not self.early_stop:
            print("")
        if X_v is None:
            print('%s --- DONE Epoch = %d/%d AvgLoss = %.4f\n' % (now(), epoch, max_epochs-1, e_loss))
        else:
            print('%s --- DONE Epoch = %d/%d AvgLoss = %.4f ValLoss = %.4f\n' % (now(), epoch, max_epochs-1, e_loss, v_loss))
        
        return epoch_losses, val_losses

    def predict(self, X):
        '''
        Predict for a set of samples
        return a y_pred with shape = (num_samples, 1)
        '''
        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2
        assert X.shape[1] == self.num_feat
        
        if len(X) < 500000 and self.device == 'cuda':
            net = self.net.to(self.device)
            net.eval()
            X = torch.from_numpy(X).float().to(self.device)
            y_pred = net(X).cpu().data.numpy()
        else:
                # too big for GPU, use CPU
            net = self.net.to('cpu')
            net.eval()
            X = torch.from_numpy(X).float()
            y_pred = net(X).data.numpy()
            
        return y_pred

    def plot_training_evolution(self, loss_limit=None):
        assert self.trained
        p = plt.plot(self.epoch_losses, label='Training loss')
        p = plt.plot(self.val_losses, label='Validation loss')
        p = plt.title('Evolution of training losses')
        p = plt.legend(frameon=False)
        
            # limit to the interesting part
        if loss_limit is not None:
            idxs = np.where(np.array(self.epoch_losses) < loss_limit)
            if len(idxs[0]) > 0:
                idx = idxs[0][1]
                p = plt.gca().set_xlim(left=idx)
                p = plt.gca().set_ylim(bottom=0,top=loss_limit)
        
        plt.show() 
        return p
    