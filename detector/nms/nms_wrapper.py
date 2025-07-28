import numpy as np
import torch

NMS_CPU_AVAILABLE = False
print("Warning: nms_cpu extension disabled to enforce soft_nms_cpu")
NMS_CUDA_AVAILABLE = False
print("Warning: nms_cuda extension not available")

from .soft_nms_cpu import soft_nms_cpu
import torchvision.ops

def nms(dets, iou_thr, device_id=None):
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets  # Keep on original device for torchvision
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        dets_th = torch.from_numpy(dets)
    else:
        raise TypeError('dets must be a Tensor or numpy array, got {}'.format(type(dets)))

    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.device.type in ['mps', 'cpu']:
            # Try torchvision.nms for MPS/CPU
            boxes = dets_th[:, :4]
            scores = dets_th[:, 4]
            inds = torchvision.ops.nms(boxes, scores, iou_thr)
        else:
            raise RuntimeError("Unsupported device type for NMS")
    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds

def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_np = dets.detach().cpu().numpy()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_np = dets
    else:
        raise TypeError('dets must be a Tensor or numpy array, got {}'.format(type(dets)))

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError('Invalid method for SoftNMS: {}'.format(method))
    new_dets, inds = soft_nms_cpu(dets_np, iou_thr, method=method_codes[method], sigma=sigma, min_score=min_score)
    if is_tensor:
        return dets.new_tensor(new_dets), dets.new_tensor(inds, dtype=torch.long)
    else:
        return new_dets.astype(np.float32), inds.astype(np.int64)