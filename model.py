import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPEnc(nn.Module):
    def __init__(self, hidden_dim=256, dropout_rate=0.3):
      super(MLPEnc, self).__init__()

      # Input layer
      self.l1 = nn.LazyLinear(hidden_dim)
      self.bn1 = nn.BatchNorm1d(hidden_dim)

      # Hidden layer 1
      self.l2 = nn.LazyLinear(hidden_dim)
      self.bn2 = nn.BatchNorm1d(hidden_dim)

      # Hidden layer 2
      self.l3 = nn.LazyLinear(hidden_dim)
      self.bn3 = nn.BatchNorm1d(hidden_dim)

      # Dropout layer
      self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
      x = torch.flatten(x, start_dim=1)
      
      x = self.l1(x)
      x = self.bn1(x)
      x = F.relu(x)
      x = self.dropout(x)

      x = self.l2(x)
      x = self.bn2(x)
      x = F.relu(x)
      x = self.dropout(x)

      return x

class ConvEnc(nn.Module):
  def __init__(self, conv_depth, kernel_size, dropout_rate):
    super(ConvEnc, self).__init__()
    self.conv = nn.LazyConv1d(conv_depth, kernel_size)
    self.batch_norm = nn.BatchNorm1d(conv_depth)
    self.dropout = nn.Dropout(dropout_rate)

    self.flatten = nn.Flatten()

  def forward(self, x):
    x = self.conv(x)
    x = F.relu(x)
    x = self.batch_norm(x)
    x = self.dropout(x)
    x = self.flatten(x)

    return x


class FC_Classifier(torch.nn.Module):
    def __init__(self, num_classes, num_hidden_layers=1, hidden_dim=256, drop_prob=0.2):
        super(FC_Classifier, self).__init__()
        # Hidden layers (if any)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.LazyLinear(hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=drop_prob)
                )
            )
        self.clf  = nn.Linear(hidden_dim, num_classes)

    def forward(self, emb):
        for layer in self.hidden_layers:
            emb = layer(emb)
        return self.clf(emb)

class ProjH(torch.nn.Module):
  def __init__(self, projDim):
     super(ProjH, self).__init__()
     self.proj = nn.LazyLinear(projDim)
  
  def forward(self, x):
     return self.proj(x)

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha=1.):
        ctx.alpha = alpha
        return x.view_as(x)
        #print(alpha)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * -ctx.alpha
        return output, None

def grad_reverse(x,alpha=1.):
    return GradReverse.apply(x,alpha)

class XModelv5(nn.Module):
  def __init__(self, encoder_type = 'MLP', hidden_dim=256, dropout_rate=0.3, nb_classes=3, n_doms=5):
    super(XModelv5, self).__init__()
    if encoder_type == 'MLP':
        self.enc_inv = MLPEnc(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    elif:
        self.enc_inv = ConvEnc(conv_depth=50, kernel_size=3, dropout_rate=0.2)

    self.cl = nn.LazyLinear(nb_classes)
    self.cl_dom_adv = FC_Classifier(n_doms, num_hidden_layers=2)
    self.proj = ProjH(128)
    self.softm = nn.Softmax(dim=1)

  def forward(self, x):
    emb_inv = self.enc_inv(x)
    
    x_cl = self.cl(emb_inv)

    #CDAN MODULE    
    softmax_output = self.softm( x_cl.detach() )
    cdan_features = torch.bmm(softmax_output.unsqueeze(2), emb_inv.unsqueeze(1))
    cdan_features = cdan_features.view(-1, softmax_output.size(1) * emb_inv.size(1))
    x_dom_adv = self.cl_dom_adv(grad_reverse(cdan_features))

    return x_cl, x_dom_adv, self.proj(emb_inv)
