import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

default_new_tensor = torch.FloatTensor

class conv1d_local(nn.Module):
    def __init__(self, L, nc, nf, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(conv1d_local, self).__init__()
        
        self.padding = padding
        
        kernel_locations = get_kernel_locations(L, kernel_size, stride, padding, dilation)
        self.kernel_locations = nn.ParameterList(map(lambda x: nn.Parameter(x, requires_grad=False), kernel_locations))
        # self.kernel_locations = kernel_locations
        
        L_out = len(self.kernel_locations)
        self.kernel_weights = nn.ParameterList([])
        for _ in range(L_out):
            t = default_new_tensor(nf, nc, kernel_size)
            nn.init.normal_(t)
            p = nn.Parameter(t)
            self.kernel_weights.append(p)
        
        if bias:
            self.bias_flag = True
            t = default_new_tensor(nf, L_out) # (nf, L_out)
            nn.init.normal_(t)
            self.bias = nn.Parameter(t)
        #
    def forward(self, x):
        # x in shape ((nb,) nc, L)
        x = nn.ConstantPad1d(self.padding, 0)(x)
        conv_results = []
        for n_location, kernel_location in enumerate(self.kernel_locations):
            kernel_weight = self.kernel_weights[n_location] # (nf, nc, kernel_size)
            # kernel_location = torch.LongTensor(kernel_location) # (kernel_size, )
            conv_block = torch.index_select(x, 2, kernel_location) # ((nb,) nc, kernel_size)
            conv_block = torch.unsqueeze(conv_block, 1) # ((nb,) 1, nc, kernel_size)
            conv_result = torch.sum(torch.sum(torch.mul(kernel_weight, conv_block), 3), 2) # ((nb,) nf)
            conv_results.append(conv_result)
        output = torch.stack(conv_results, 2) # ((nb,) nf, L_out)
        if self.bias_flag: output = torch.add(output, self.bias)
        
        return output
            
            
            


#
        
def get_kernel_locations(L, kernel_size, stride, padding, dilation):
    kernel_locations = []
    kernel_location = 0
    n_kernel = int(np.floor((L+2*padding-dilation*(kernel_size-1)-1)/stride+1))
    for _ in range(n_kernel):
        conv_locations = get_conv_locations(kernel_location, kernel_size, dilation)
        kernel_locations.append(conv_locations)
        kernel_location = kernel_location + stride
    
    return kernel_locations


def get_conv_locations(kernel_location, kernel_size, dilation):
    conv_locations = []
    p = kernel_location
    for _ in range(kernel_size):
        conv_locations.append(p)
        p = p + dilation
    conv_locations = torch.LongTensor(conv_locations)
    
    return conv_locations