import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

# Attention Module
class AttentionModule(nn.Module):
    def __init__(self,query_dim,num_classes):
        super(AttentionModule,self).__init__()

        self.query_dim = query_dim
        self.num_classes = num_classes
        self.alpha = None

        # context vectors for computing attention
        self.U = nn.Linear(query_dim, self.num_classes)
        torch.rand(())
        xavier_uniform_(self.U.weight)

    def forward_calc_atten(self, keys,query):
        """        
        INPUT:
            keys: (batch size, num_local_feature_vectors, num_filter_maps)
                the local feature vectors
            query: (batch sixe, num_class,num_filter_maps)
                vector transformations for each label
        
        OUTPUT:
            alpha: (batch size, num_class, num_local_feature_vectors)
                the attention weights
        """    
        x = keys.transpose(1,2)
        alpha = torch.matmul(self.U.weight,x)
        alpha = alpha / np.sqrt(x.shape[1])
        self.alpha = F.softmax(alpha,dim=2)
        #print('alpha-->',alpha.shape)
        return self.alpha

    def forward_aply_atten(self, alpha, keys):
        """
        INPUT: 
            keys: (batch size, num_local_feature_vectors, num_filter_maps)
                the local feature vectors
            alpha: (batch size, num_class, num_local_feature_vectors)
                the attention weights
            
        OUTPUT:
            v: (batch size, num_class, num_filter_maps), vector representations for each label
        """
        v = torch.matmul(alpha,keys)
        #print('V-->',v.shape)
        return v

    def forward(self,keys):
        """
        INPUT:
            keys: (batch size, num_local_feature_vectors, num_filter_maps)
        OUTPUT:
            attention_vectors : (batch size, num_class, num_filter_maps), vector representations for each label
        """
        alpha = self.forward_calc_atten(keys=keys, query=self.U)
        return self.forward_aply_atten(alpha, keys)


##################
# Baseline with Attention Model
class VGG16Attention(nn.Module):
  def __init__(self,num_classes=14,
                    num_filter_maps_block1 = 256,
                    num_filter_maps_block2 = 512,
                    num_filter_maps_block3 = 512):
    super(VGG16Attention,self).__init__()

    self.num_classes = num_classes
   
    # Load the pretrained VGG16 model
    vgg16 = torchvision.models.vgg16_bn(pretrained=True)

    self.vgg16_block1 = vgg16.features[0:24]
    self.vgg16_block2 = vgg16.features[24:34]
    self.vgg16_block3 = vgg16.features[34:]
    self.avgpool = vgg16.avgpool

    # dimension of local vectors at end of each block
    self.num_filter_maps_block1 = num_filter_maps_block1
    self.num_filter_maps_block2 = num_filter_maps_block2
    self.num_filter_maps_block3 = num_filter_maps_block3
        
    # attention modules
    self.attn1 = AttentionModule(self.num_filter_maps_block1, num_classes)
    self.attn2 = AttentionModule(self.num_filter_maps_block2, num_classes)
    self.attn3 = AttentionModule(self.num_filter_maps_block3, num_classes)

    # final layer
    final_input_dim = self.num_filter_maps_block1 + \
                      self.num_filter_maps_block2 + \
                      self.num_filter_maps_block3

    self.final = nn.Linear(final_input_dim, num_classes)
    xavier_uniform_(self.final.weight)

    # Freeze the weights of the model
    for param in self.vgg16_block1.named_parameters():
      param[1].requires_grad = False
  
  def forward_linear(self, v):
    """        
        INPUT: 
            v: (batch size, num_class, num_filter_maps), vector representations for each label
            
        OUTPUT:
            y_hat: (batch size, num_class), label probability
    """
    y_hat = torch.mul(self.final.weight, v)
    y_hat = torch.sum(y_hat,dim=2)
    y_hat = y_hat + self.final.bias
    y_hat = torch.sigmoid(y_hat)
    return y_hat

  def forward(self, image):

    """ 1. apply vgg16 convolution """
    x_block1 = self.vgg16_block1(image)
    x_block2 = self.vgg16_block2(x_block1)
    x_block3 = self.vgg16_block3(x_block2)
    x_block3 = self.avgpool(x_block3)
    #print('blocks',x_block1.shape,x_block2.shape,x_block3.shape)
    
    """ 2. change the shape of the input """
    key1 = x_block1.permute(0,2,3,1)
    key1 = key1.view(key1.shape[0],-1,key1.shape[-1])

    key2 = x_block2.permute(0,2,3,1)
    key2 = key2.view(key2.shape[0],-1,key2.shape[-1])

    key3 = x_block3.permute(0,2,3,1)
    key3 = key3.view(key3.shape[0],-1,key3.shape[-1])
    #print('keys',key1.shape,key2.shape,key3.shape)    

    """ 3. calculate and apply attention """
    v1 = self.attn1(key1)
    v2 = self.attn2(key2)
    v3 = self.attn3(key3)
    #print('v1,v2,v3', v1.shape,v2.shape,v3.shape)
    
    """ 4. concatenate the attention vectors from each block """
    v = torch.cat((v1,v2,v3),dim=2)
    #print('v', v.shape)
           
    """ 5. final layer classification """
    y_hat = self.forward_linear(v)
    return y_hat

#######################

# Baseline Model
class VGG16Base(nn.Module):
  def __init__(self,num_classes=14):
    super(VGG16Base,self).__init__()

    self.num_classes = num_classes
    # Load the pretrained VGG16 model
    self.vgg16 = torchvision.models.vgg16_bn(pretrained=True)

    # Adjust the last layer according to number of classes and make it smaller
    self.vgg16.classifier = nn.Linear(in_features=25088, out_features=num_classes, bias=True)

    # Freeze the weights of the model, except for the last 6 CNN & linear layers
    for param in self.vgg16.named_parameters():
      if param[0] in {'classifier.weight','classifier.bias',
                      'features.40.weight','features.41.weight','features.40.bias','features.41.bias',
                      'features.37.weight','features.38.weight','features.37.bias','features.38.bias',
                      'features.34.weight','features.35.weight','features.34.bias','features.35.bias',
                      'features.30.weight','features.31.weight','features.30.bias','features.31.bias',
                      'features.27.weight','features.28.weight','features.27.bias','features.28.bias',
                      'features.24.weight','features.25.weight','features.24.bias','features.25.bias'}: continue
      param[1].requires_grad = False
  
  def forward(self,x):
    x = self.vgg16(x)
    return torch.sigmoid(x)

#######################