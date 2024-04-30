import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from data_processing.write_ply import write_ply_standard
import imageio
import numpy as np
import json

class TestMetricProcessor(nn.Module):
    def __init__(self, gt_clamp_range=(0.2,10), save_freq = 10):
        super().__init__()
        self.gt_clamp_range = gt_clamp_range
        self.save_freq = save_freq

    def get_stats(self, gt, pred):
        stats_dict = {}
        
        thresh = torch.max((gt / pred), (pred / gt))
        stats_dict['delta1'] = torch.nanmean( torch.where(torch.isfinite(thresh), (thresh < 1.25).float(), thresh),dim=1)
        stats_dict['delta2'] = torch.nanmean( torch.where(torch.isfinite(thresh), (thresh < 1.25**2).float(), thresh),dim=1)
        stats_dict['delta3'] = torch.nanmean( torch.where(torch.isfinite(thresh), (thresh < 1.25**3).float(), thresh),dim=1)

        
        stats_dict['abs_rel'] = torch.nanmean( torch.abs(gt-pred)/gt , dim=1)
        stats_dict['sq_rel'] = torch.nanmean( (gt-pred)**2/gt, dim=1)

        stats_dict['rmse'] = torch.sqrt(  torch.nanmean( (gt-pred)**2 , dim=1) )
        stats_dict['log_rmse'] = torch.sqrt(  torch.nanmean( (torch.log(gt)-torch.log(pred))**2 , dim=1) )
        return stats_dict
 
    def step(self, net_out, sample, save_dir):
       
        if 'pred_distance_original' in net_out:

            gt = sample['distance'][:,0].flatten(1).clamp(*self.gt_clamp_range)
            pred = net_out['pred_distance_original'].flatten(1).clamp(*self.gt_clamp_range)
            pred_refined = net_out['pred_refined_distance_original'].flatten(1).clamp(*self.gt_clamp_range)
            sd = self.get_stats(gt, pred)
            sd_refined = self.get_stats(gt, pred_refined)
        

            for k,v in sd_refined.items():
                sd[k + '_refined'] = v
        else:
            sd = {}
        
        if sample['idx'][0] % self.save_freq == 0:
            curr_save_dir = os.path.join(save_dir, 'visuals', '%06d' % sample['idx'][0].item())
            os.makedirs(curr_save_dir, exist_ok=True)

            pred_refined_distance = net_out['pred_refined_distance'][0].squeeze(0) #(h,w)
            cam = sample['canon_camera'][0,0] #()
            im = sample['canon_image'][0,0] #(3,h,w)
            im = im.permute(1,2,0)
            pc, _ = cam.unproject_depth(pred_refined_distance, depth_is_along_ray=True) #(h,w,3)
            mask = (pred_refined_distance > 0.01) & (torch.mean(im,dim=-1) > 0.01)
            pc = pc[mask, :]
            pc_color = im[mask,:]
            write_ply_standard(os.path.join(curr_save_dir, 'pred_pc.ply'), pc.reshape(-1,3), colors=pc_color.reshape(-1,3))
            for i in range(sample['image'].size(1)):
                im = sample['image'][0,i].permute(1,2,0)
                imageio.imwrite(os.path.join(curr_save_dir, 'image_%d.jpg' % i), (im.cpu().numpy()*255).astype(np.uint8))
            
            if 'pred_refined_distance_original' in net_out:
                pred_depth = net_out['pred_refined_distance_original'][0].squeeze()
                imageio.imwrite(os.path.join(curr_save_dir, 'log_pred_depth.png'), (torch.log(pred_depth).cpu().numpy()*50).astype(np.uint8))
        return sd
	
    def final_compute(self, agg_out_dicts, save_dir):
        results_path = os.path.join(save_dir, 'stats.json')
        results = {}
        for k,v in agg_out_dicts.items():
            if not k.startswith('idx'):
               results[k] = torch.mean(v.float()).item()
        
        json_str = json.dumps(results, indent=4, sort_keys=True)
        print('results: ')
        print(json_str)
        print('saving results to: ', results_path)
        with open(results_path, 'w') as f:
            f.write(json_str)
