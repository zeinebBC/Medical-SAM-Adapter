"""
David Gastager - Zeiss
Mean Intersection over Union Metric
Allows the user to automatically match an arbitrary amount of predicted masks to a ground truth mask and calculate their IoU.
"""
import torch
from torchmetrics import Metric
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Literal, Optional, Union

MATCHING_TYPE = Literal['multi', 'greedy', 'optimal']

class MeanIoU(Metric):
    """ 
    Mean Intersection over Union Metric.
    First matches predicted instances masks with ground truth/target instance masks according to MATCHING_TYPE.
    Then calculates micro (per instance) and macro (per image) intersection over union metric.
    """
    is_differentiable=False
    higher_is_better=True
    full_state_update=False

    def __init__(
            self,
            matching_strategy: MATCHING_TYPE = 'optimal',  
            return_iou_per_image: bool = False,
            return_iou_per_instance: bool = False,
        ):
        """
        :param matching_strategy: Type of assignment: Any-to-any: 'multi'; One-to-one: 'optimal', 'greedy'
        :param return_iou_per_image: Return additional parameter in dictionary with ious of all images
        :param return_iou_per_instance: Return additional parameter in dictionary with ious of all matched instances
        """
        super().__init__()
        self.matching_strategy = matching_strategy# If true one prediction can be assigned to multiple ground truths
        self.return_iou_per_instance = return_iou_per_instance
        self.return_iou_per_image = return_iou_per_image

        self.add_state('n_images', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('n_instances', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('ious_micro', default=[], dist_reduce_fx='cat')
        self.add_state('ious_macro', default=[], dist_reduce_fx='cat')

    def update(self, preds: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray]):
        """ 
        Update step of mean IoU calculation metric
        :param preds: predicted segmentation masks with shape (m_instance, width, height)
        :param target: target segmentation masks with shape (n_instances, width, height)
        Note that n_instances = m_instances doesn't have to hold, as they'll be matched according to matching_strategy
        """
        # Convert Numpy to torch
        preds = torch.from_numpy(preds) if type(preds) == np.ndarray else preds
        target = torch.from_numpy(target) if type(target) == np.ndarray else target
        # Ensure batch dimension is present
        preds = preds.unsqueeze(0) if len(preds.shape) == 3 else preds
        target = target.unsqueeze(0) if len(target.shape) == 3 else target

        # iterate over images in batch
        for pred, label in zip(preds, target):
            iou_matrix = self._calc_iou_matrix(pred, label)
            if self.matching_strategy == 'multi':
                # Allow assignment of one prediction to multiple labels
                iou_assigned, _ = iou_matrix.max(axis=1)
            elif self.matching_strategy == 'greedy':
                # Use greedy assignment of predictions to labels, based on object size
                iou_assigned, _ = self._greedy_matching(label, iou_matrix)
            else: #if self.matching_strategy == 'optimal':
                # Optimal one-to-one matching
                iou_assigned, _ = self._optimal_matching(iou_matrix)

            self.n_instances += iou_assigned.shape[0]
            macro = iou_assigned if iou_assigned is not None else 0.0
            self.ious_macro.append(macro.mean())
            self.ious_micro.append(iou_assigned)

        self.n_images += preds.shape[0]

    def compute(self):
        """ 
        Calculate micro and macro mean IoUs and return resulting dictionary.
        """
        # Calculate micro and macro means
        miou_micro = torch.cat(self.ious_micro).flatten().mean()
        miou_macro = torch.Tensor(self.ious_macro).flatten().mean()

        miou_dict = {
            'mIoU_micro': miou_micro,
            'mIoU_macro': miou_macro,
            'n_instances': self.n_instances,
            'n_images': self.n_images
        }

        if self.return_iou_per_image:
            miou_dict['image_ious'] = self.ious_macro
        if self.return_iou_per_instance:
            miou_dict['instance_ious'] = self.ious_micro
        return miou_dict

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
    def _greedy_matching(label, iou_matrix: torch.Tensor):
        """
        Matches predictions to label based on their exhaustive IoU matrix using a greedy algorithm with object size in mind
        :param iou_matrix: Exhaustive Intersection over Union matri with (rows: labels x cols: predictions)
        """
        label = label.numpy() #if type(label) == torch.Tensor else label
        iou_matrix = iou_matrix.numpy() #if type(iou_matrix) == torch.Tensor else iou_matrix
        n_objects = label.shape[0]
        # Calculate object sizes
        sizes = (label>0).sum(axis=(1,2))
        # rearrange iou_matrix by descending size ob objects
        idcs = np.argsort(sizes)[::-1]
        iou_matrix = iou_matrix[idcs]

        ious = []
        idcs = []
        for i in range(n_objects):
            if iou_matrix.size == 0:
                ious.append(0.0)
                idcs.append(-1)
                continue
            idx = iou_matrix[i].argmax()
            idcs.append(idx)
            ious.append(iou_matrix[i, idx])
            iou_matrix = np.delete(iou_matrix, idx, 1) # Delete columns of assigned object 
        return torch.from_numpy(np.array(ious)), torch.from_numpy(np.array(idcs))

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
