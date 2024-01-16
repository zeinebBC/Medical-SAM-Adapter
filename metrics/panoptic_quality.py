"""
David Gastager - Zeiss
Mean Intersection over Union Metric
Allows the user to automatically match an arbitrary amount of predicted masks to a ground truth mask and calculate their Panoptix Quality.
# https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Benchmarking_Self-Supervised_Learning_on_Diverse_Pathology_Datasets_CVPR_2023_paper.pdf
# https://iq.opengenus.org/pq-sq-rq/
"""
from typing import Literal, Optional, Union
import torch
from torchmetrics import Metric
import numpy as np
from scipy.optimize import linear_sum_assignment

MATCHING_TYPE = Literal['multi', 'greedy', 'optimal']


class PanopticQuality(Metric):
    """ 
    Panoptic Quality Metrics.
    First matches predicted instances masks with ground truth/target instance masks according to MATCHING_TYPE.
    Then calculates micro (per instance) and macro (per image) panoptic quality metric.
    """
    is_differentiable=False
    higher_is_better=True
    full_state_update=False

    def __init__(
            self,
            return_per_image: bool = False,
            iou_thresh: float = 0.5
        ):
        """
        :param return_per_instance: Return additional parameter in dictionary with ious of all matched instances
        :param matching_strategy: Type of assignment: Any-to-any: 'multi'; One-to-one: 'optimal', 'greedy'
        :param mean_type: 'macro': per image mean, 'micro': per mask mean, None: No mean. returns all ious
        """
        super().__init__()
        self.return_per_image = return_per_image
        self.iou_thresh = iou_thresh

        self.add_state('n_images', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('n_instances', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('pq', default=[], dist_reduce_fx='cat')
        self.add_state('seg_q', default=[], dist_reduce_fx='cat')
        self.add_state('rec_q', default=[], dist_reduce_fx='cat')

    def update(self, preds: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray]):
        """ 
        Update step of mean IoU calculation metric
        :param preds: predicted segmentation masks with shape (m_instance, width, height)
        :param target: target segmentation masks with shape (n_instances, width, height)
        Note that n_instances = m_instances doesn't have to hold, as they'll be matched according to matching_strategy
        All calculations are based on this: https://iq.opengenus.org/pq-sq-rq/
        """
        pq, seg_q, rec_q, n_images = self._calc_pq(preds, target)
        self.pq += pq
        self.seg_q += seg_q
        self.rec_q += rec_q
        self.n_images += n_images

    def __call__(self, preds: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray], iou_thresh: Optional[float] = None):
        pq, seg_q, rec_q, n_images = self._calc_pq(preds, target, iou_thresh=iou_thresh)
        pq = torch.Tensor(pq).flatten().mean()
        rec_q = torch.Tensor(rec_q).flatten().mean()
        seg_q = torch.Tensor(seg_q).flatten().mean()

        return {
            'panoptic_quality': pq,
            'recognition_quality': rec_q,
            'segmentation_quality': seg_q,
        }

    def _calc_pq(self, preds: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray], iou_thresh: Optional[float] = None):
        """ 
        Calculation of Panoptic Quality, Segmentation Quality and Recogition Quality
        :param preds: predicted segmentation masks with shape (m_instance, width, height)
        :param target: target segmentation masks with shape (n_instances, width, height)
        Note that n_instances = m_instances doesn't have to hold, as they'll be matched according to matching_strategy
        All calculations are based on this: https://iq.opengenus.org/pq-sq-rq/
        """
        # Convert Numpy to torch
        preds = torch.from_numpy(preds) if isinstance(preds, np.ndarray) else preds
        target = torch.from_numpy(target) if isinstance(target, np.ndarray) else target
        # Ensure batch dimension is present
        preds = preds.unsqueeze(0) if len(preds.shape) == 3 else preds
        target = target.unsqueeze(0) if len(target.shape) == 3 else target

        seg_qs = []
        rec_qs = []
        pqs = []
        # iterate over images in batch
        for pred, label in zip(preds, target):
            iou_matrix = self._calc_iou_matrix(pred, label)
            iou_assigned, idcs_assigned = self._optimal_matching(iou_matrix)
            t = iou_assigned > self.iou_thresh if iou_thresh is None else iou_assigned > iou_thresh
            TP = t.sum()
            FP = (~t).sum()
            FN = min(label.shape[0] - iou_assigned.shape[0], 0)
            seg_q = iou_assigned.sum()/TP if TP != 0 else 0
            rec_q = TP / (TP + 0.5*(FP + FN))
            seg_qs.append(seg_q)
            rec_qs.append(rec_q)
            pqs.append(seg_q * rec_q)
        n_images = preds.shape[0]
        return pqs, seg_qs, rec_qs, n_images

    def compute(self):
        """ 
        Calculate recognition quality, segmentation quality and panoptic quality and return resulting dictionary.
        """
        pq = torch.Tensor(self.pq).flatten().mean()
        rec_q = torch.Tensor(self.rec_q).flatten().mean()
        seg_q = torch.Tensor(self.seg_q).flatten().mean()

        out_dict = {
            'panoptic_quality': pq,
            'recognition_quality': rec_q,
            'segmentation_quality': seg_q,
        }

        if self.return_per_image:
            out_dict['image_pq'] = self.pq
            out_dict['image_rq'] = self.rec_q
            out_dict['image_sq'] = self.seg_q
        return out_dict

    @staticmethod
    def _optimal_matching(iou_matrix: torch.Tensor):
        """
        :param iou_matrix: Exhaustive Intersection over Union matri with (rows: labels x cols: predictions)
        Matches prediction to labels based on their exhaustive IoU matrix using optimal assignment:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        """
        iou_matrix = iou_matrix.numpy() #if type(iou_matrix) == torch.Tensor else iou_matrix
        row_idcs, col_idcs = linear_sum_assignment(iou_matrix, maximize=True)
        iou_assigned = np.zeros(iou_matrix.shape[0])
        iou_assigned[row_idcs] = iou_matrix[row_idcs, col_idcs]

        idcs_assigned = np.zeros(iou_matrix.shape[0], dtype=int)
        idcs_assigned[row_idcs] = col_idcs
        return torch.from_numpy(iou_assigned), torch.from_numpy(idcs_assigned)

    @staticmethod
    def _calc_iou_matrix(preds: torch.Tensor, labels: torch.Tensor):
        """ 
        Generates iou matrix between all label/ground truth masks and all predicted masks for one image
        :param preds: predicted masks with shape (n_masks, width, height)
        :param labels: ground truth masks with shape (m_masks, width, height)
        width, height should be the same, but n_masks and m_masks can differ.
        """
        iou_matrix = torch.zeros((labels.shape[0], preds.shape[0]))
        for i, label in enumerate(labels):
            for j, pred in enumerate(preds):
                intersection = ((label>0) * (pred>0)).sum()
                union = ((label>0) + (pred>0)).sum()
                iou_matrix[i,j] = intersection.sum() / union.sum() if union>0 else 0.0
        return iou_matrix
