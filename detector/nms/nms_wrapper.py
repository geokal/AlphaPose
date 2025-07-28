import numpy as np
import torch

# Import nms_cpu only if available (disabled due to PyTorch 2.0.1 compatibility issues)
try:
    from . import nms_cpu
    NMS_CPU_AVAILABLE = True
except ImportError:
    NMS_CPU_AVAILABLE = False
    print("Warning: nms_cpu extension not available due to PyTorch compatibility issues")

try:
    from . import nms_cuda
    NMS_CUDA_AVAILABLE = True
    import torchvision.ops
except ImportError:
    NMS_CUDA_AVAILABLE = False
    print("Warning: nms_cuda extension not available")

from .soft_nms_cpu import soft_nms_cpu


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets.to('cpu')
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.device.type == 'mps' or dets_th.device.type == 'cpu':
            # Force soft-NMS on Apple Silicon (MPS) and CPU
            return soft_nms(dets, iou_thr)
        elif dets_th.is_cuda or dets_th.device.type == 'mps':
            if NMS_CUDA_AVAILABLE and dets_th.is_cuda:
                inds = nms_cuda.nms(dets_th, iou_thr)
            else:
                # Fallback to torchvision.ops.nms for MPS/cpu
                boxes_cpu = dets_th.cpu()
                scores = boxes_cpu[:, 4]
                keep = torchvision.ops.nms(boxes_cpu[:, :4], scores, iou_thr)
                inds = keep.to(dets_th.device)
                raise RuntimeError("CUDA NMS not available")
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
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError('Invalid method for SoftNMS: {}'.format(method))
    new_dets, inds = soft_nms_cpu(
        dets_np,
        iou_thr,
        method=method_codes[method],
        sigma=sigma,
        min_score=min_score)

    if is_tensor:
        return dets.new_tensor(new_dets), dets.new_tensor(
            inds, dtype=torch.long)
    else:
        return new_dets.astype(np.float32), inds.astype(np.int64)
