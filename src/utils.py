import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset import ChestXrayDataSet
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from focal_loss.focal_loss import FocalLoss

# Set the seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Build data loaders
def load_data(data,data_path,crop_size=224,shuffle=False, batch_size=32):
  
  transform = transforms.Compose( [ transforms.RandomResizedCrop(crop_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])
                                  ])

  dataset = ChestXrayDataSet(data,data_path,transform)
  loader = torch.utils.data.DataLoader(dataset = dataset, shuffle=shuffle, batch_size = batch_size)
  return loader

# Load the model with previously trained weights
def load_model(model,model_path,file=None,device='cpu'):
  if file is not None:
    model.load_state_dict(torch.load(model_path + file,map_location=device))
  return model

# Get Class Level ROC-AUC Metric
def compute_class_auc_roc_score(y_true,pred_probs):
  CLASS_AUC = []
  for i in range(y_true.shape[1]):
    auc = roc_auc_score(y_true[:,i],pred_probs[:,i])
    CLASS_AUC.append(auc)

  return CLASS_AUC

# Evaluate Model
def evaluate_model(model,val_loader,criterion,device='cpu'):
  model.eval()
  
  val_loss = 0
  y_true = torch.LongTensor()
  y_pred_prob = torch.FloatTensor()

  for i, data in enumerate(val_loader):
    data, labels = data
    data, labels = data.to(device), labels.to(device)

    # Get the model predictions and collect them for each batch
    pred_prob = model(data)
    y_pred_prob = torch.cat((y_pred_prob, pred_prob.detach().to('cpu')), dim=0)
    y_true = torch.cat((y_true, labels.detach().to('cpu')), dim=0)
    # Compute the loss
    loss = criterion(pred_prob,labels)

    # Accumulate batch loss
    val_loss += loss.item()
  
  val_loss = val_loss / len(val_loader)
  CLASS_AUC = compute_class_auc_roc_score(y_true,y_pred_prob)

  return val_loss,CLASS_AUC

# Train Model
def train_model(  model, 
                  train_loader,
                  criterion,optimizer,
                  parameters,
                  device='cpu',
                  val_loader=None
                ):
  model.train()

  TRAIN_LOSS = []
  VAL_LOSS = []
  VAL_CLASS_AUC = []

  for epoch in range(parameters['n_epochs']):
    print(' EPOC==>',epoch+1)
    train_loss = 0
    #y_true = torch.LongTensor()
    y_pred_prob = torch.FloatTensor()

    for i, data in enumerate(train_loader):
      data, labels = data
      data, labels = data.to(device), labels.to(device)

      # Initialize the optimizer
      optimizer.zero_grad()

      # Get the model predictions and collect them for each batch
      pred_prob = model(data)
      y_pred_prob = torch.cat((y_pred_prob, pred_prob.detach().to('cpu')), dim=0)
      #y_true = torch.cat((y_true, labels.detach().to('cpu')), dim=0)

      # Compute the loss
      loss = criterion(pred_prob,labels)
      # Perform back propagation
      loss.backward()
      # Update weights
      optimizer.step()
      
      # Accumulate batch loss
      train_loss += loss.item()
    
    # Average the training loss across entire training dataset
    train_loss = train_loss / len(train_loader)
    TRAIN_LOSS.append(train_loss)
    print('   Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))

    # Save Model
    if parameters['save_model_per_epoch']=="True":
      model_file = parameters['model_path'] + str(epoch+1) + '-' + parameters['model_descriptor'] + '.pth' 
      torch.save(model.state_dict(),model_file)
      print('   ===Model Saved====')
    
    # Perform validation
    if val_loader is not None:
      val_loss, auc = evaluate_model(model,val_loader,criterion,device)
      VAL_LOSS.append(val_loss)
      VAL_CLASS_AUC.append(auc)
      print('   Epoch: {} \t Validation Loss: {:.6f}'.format(epoch+1,val_loss))

    # Save Metrics
    if parameters['save_metrics'] == "True":
      
      filename = parameters['metrics_path'] + parameters['model_descriptor'] + '-TRAIN-LOSS.csv' 
      pd.DataFrame(TRAIN_LOSS).to_csv(filename)

      if len(VAL_LOSS) > 0:
        filename = parameters['metrics_path'] + parameters['model_descriptor'] + '-VAL-LOSS.csv' 
        pd.DataFrame(VAL_LOSS).to_csv(filename)

      if len(VAL_CLASS_AUC) > 0:
        filename = parameters['metrics_path'] + parameters['model_descriptor'] + '-VAL-CLASS-AUC.csv' 
        pd.DataFrame(VAL_CLASS_AUC).to_csv(filename)

      print('   ===Metrics Saved====')

  return TRAIN_LOSS,VAL_LOSS,VAL_CLASS_AUC

# Train the model
def perform_training(model,parameters):
    # Read Training Data
    train_meta_file = parameters['meta_data_path']+'train_list.csv'
    val_meta_file = parameters['meta_data_path']+'val_list.csv'

    train_df = pd.read_csv(train_meta_file)
    val_df = pd.read_csv(val_meta_file)
    
    # Build the train loader
    print('-----Building Train Loader Started-----\n')
    train_loader = load_data( train_df,
                              parameters['train_data_path'],
                              crop_size=224,
                              shuffle=True, 
                              batch_size=parameters['train_batch_size']
                            )
    
    print('-----Building Train Loader Complete-----\n')
    # Build the val loader
    val_loader = None
    if parameters['perform_validation'] == "True":

      print('-----Building Val Loader Started-----\n')
      val_loader = load_data( val_df,
                              parameters['val_data_path'],
                              crop_size=224,
                              shuffle=False, 
                              batch_size=parameters['val_batch_size']
                            )

      print('-----Building Val Loader Complete-----\n')

    # Get the device on which model is going to run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Specify the loss function and optimizer
    if parameters['use_focal_loss'] == 'True':
      criterion = FocalLoss(alpha=parameters[alpha], gamma=parameters[gamma])
    else:
      criterion=nn.BCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'])

    # Start Training
    print('=====Model Training Started============\n')
    # Load the model to device
    model = model.to(device)
    
    # Train Model
    TRAIN_LOSS,VAL_LOSS,VAL_CLASS_AUC = train_model(  model,
                                                      train_loader,
                                                      criterion,optimizer,
                                                      parameters,
                                                      device,
                                                      val_loader)
    
    print('=====Model Training Complete============\n')

    return model, TRAIN_LOSS,VAL_LOSS,VAL_CLASS_AUC


# Testing the model
def perform_testing(model,parameters):
    # Read Testing Data
    test_meta_file = parameters['meta_data_path']+'test_list.csv'
    test_df = pd.read_csv(test_meta_file)
  
    # Build the test loader
    print('-----Building Test Loader Started-----\n')
    test_loader = load_data( test_df,
                              parameters['test_data_path'],
                              crop_size=224,
                              shuffle=True, 
                              batch_size=parameters['test_batch_size']
                            )
    
    print('-----Building Test Loader Complete-----\n')

    # Get the device on which model is going to run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    print('-----Loading Trained Model Started-----\n')
    model.load_state_dict(torch.load(parameters['trained_model_path'] + parameters['model_file'],map_location=device))
    print('-----Loading Trained Model Complete-----\n')

    # Specify the loss function
    if parameters['use_focal_loss'] == 'True':
      criterion = FocalLoss(alpha=parameters[alpha], gamma=parameters[gamma])
    else:
      criterion=nn.BCELoss()

    # Start Testing
    print('=====Model Testing Started============\n')
    # Load the model to device
    model = model.to(device)
    
    # Test Model
    test_loss,auc = evaluate_model( model,
                                            test_loader,
                                            criterion,
                                            device)
    
    TEST_LOSS = []
    TEST_CLASS_AUC = []
    
    TEST_LOSS.append(test_loss)
    TEST_CLASS_AUC.append(auc)
    
    filename = parameters['metrics_path'] + parameters['model_descriptor'] + '-TEST-LOSS.csv' 
    pd.DataFrame(TEST_LOSS).to_csv(filename)

    filename = parameters['metrics_path'] + parameters['model_descriptor'] + '-TEST-CLASS-AUC.csv' 
    pd.DataFrame(TEST_CLASS_AUC).to_csv(filename)

    print('   ===Metrics Saved====')
    print('=====Model Testing Complete============\n')

    return TEST_LOSS,TEST_CLASS_AUC