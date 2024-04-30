import torch
from nvtorchcam.utils import flatten_cubemap_for_visual

def scale_between(im, min_max):
    im = (im-min_max[0])/(min_max[1]-min_max[0])
    im = im.clamp(min=0,max=1)
    return im

class ImageLogger:
    def __init__(self, log_freq=100, 
                       val_log_freq=100,
                       distance_min_max=(0,6),
                       cube_mode=False):
        
        self.log_freq = log_freq
        self.val_log_freq = val_log_freq
        self.distance_min_max = distance_min_max
        self.cube_mode = cube_mode
        
        
    def __call__(self, my_model, sample, batch_idx, net_out, prefix):
        write_to_log = False
        if (prefix[:3] == 'val') and (batch_idx % self.val_log_freq == 0):
            write_to_log = True
            image_num = 10000*my_model.current_epoch + batch_idx

        if (prefix[:5] == 'train') and (my_model.global_step % self.log_freq == 0):
            write_to_log = True
            image_num = my_model.global_step
            
        if write_to_log:

            gt_image = sample['canon_image'][0] #(M,c,h,w) 
            if self.cube_mode:
                gt_image = flatten_cubemap_for_visual(gt_image, mode=1)
            gt_image = gt_image.transpose(0,1).flatten(1,2)
            my_model.logger.experiment.add_image(prefix + '_image', gt_image, image_num)

            canon_distance = sample['canon_distance'][0,0]
            if self.cube_mode:
                canon_distance = flatten_cubemap_for_visual(canon_distance, mode=1)
            my_model.logger.experiment.add_image(prefix+'_gt_distance', scale_between(canon_distance, self.distance_min_max), image_num)

            pred_distance = net_out['pred_distance'][0]
            if self.cube_mode:
                pred_distance = flatten_cubemap_for_visual(pred_distance, mode=1)
            my_model.logger.experiment.add_image(prefix+'_pred_distance', scale_between(pred_distance, self.distance_min_max), image_num)

            pred_refined_distance = net_out['pred_refined_distance'][0]
            if self.cube_mode:
                pred_refined_distance = flatten_cubemap_for_visual(pred_refined_distance, mode=1)
            my_model.logger.experiment.add_image(prefix+'_pred_refined_distance', scale_between(pred_refined_distance, self.distance_min_max), image_num)


                