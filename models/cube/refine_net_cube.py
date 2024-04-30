import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cube.midas_cube import CubeMidasNet
from models.cube.cube_conv import dense_to_sparse, sparse_to_dense

class RefineNetMono(nn.Module):
    def __init__(self, model='resnet34',pretrained=True, scale_factor=1, pred_residual=True):
        super().__init__()
      
        self.network = CubeMidasNet(input_ch=4,model=model, pretrained=pretrained, use_new_output=True)
        
        self.scale_factor = scale_factor
        self.pred_residual = pred_residual
        
    
    def forward(self, image, initial, ref_idx_to_process):
        #(b,6,3,w,w) (b,6,3,w,w) (b,6,1,w,w)
        with torch.no_grad():
            batch_size = image.size(0)
            device = image.device
            input = torch.cat([initial,image],dim=2)
           
            idx_flat = ref_idx_to_process.flatten(0)
            to_process = torch.arange(idx_flat.size(0),device=device)[idx_flat] #(k)
            
            sparse_input = dense_to_sparse(input, to_process) #(k,c,w,w)

        residual = self.network(sparse_input, to_process, batch_size)
        if self.pred_residual:
            refined = self.scale_factor*residual + sparse_input[:,0:1,...]
        else:
            refined = residual

        refined_dense = sparse_to_dense(refined, to_process, batch_size)
        
        out = {'refined': refined_dense}
        return out
    