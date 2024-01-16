"""
David Gastager - Zeiss
Mean Intersection over Union Metric
Allows the user to automatically match an arbitrary amount of predicted masks to a ground truth mask and calculate their IoU.
"""
import numpy as np
from typing import Literal, Optional, Union
from scipy.optimize import linear_sum_assignment

ASSIGNMENT_TYPE = Literal['multi', 'greedy', 'optimal']
MEAN_TYPE = Literal['micro', 'macro', None]

class MeanIoUNumpy:
    def __init__(
            self, 
            assignment_strategy: Optional[ASSIGNMENT_TYPE] = 'optimal', 
            mean_type: Optional[MEAN_TYPE] = 'macro'
        ):
        """
        :param assignment_strategy: Type of assignment: Any-to-any: 'multi'; One-to-one: 'optimal', 'greedy'
        :param mean_type: 'macro': per image mean, 'micro': per mask mean, None: No mean. returns all ious
        """
        self.assignment_strategy = assignment_strategy# If true one prediction can be assigned to multiple ground truths
        self.mean_type = mean_type

    def forward(self, preds, labels):
        ious = []
        for pred, label in zip(preds, labels):
            iou_matrix = self._calc_iou_matrix(pred, label)
            if self.assignment_strategy == 'multi':
                # Allow assignment of one prediction to multiple labels
                iou_assigned = iou_matrix.max(axis=1) 
            elif self.assignment_strategy == 'greedy':
                # Use greedy assignment of predictions to labels, based on object size
                iou_assigned, _ = self._greedy_matching(label, iou_matrix)
            else: #if self.assignment_strategy == 'optimal':
                # Optimal one-to-one matching 
                iou_assigned, _ = self._optimal_matching(iou_matrix)
            
            if self.mean_type == 'macro':
                iou_assigned = iou_assigned if iou_assigned is not None else 0.0
                ious.append(iou_assigned.mean())
            else:
                ious.append(iou_assigned)

        if self.mean_type is None:
            return np.array(ious)
        else: # self.mean_type == 'micro' or self.mean_type == 'macro':
            return np.array(ious).flatten().mean()

    def _optimal_matching(self, iou_matrix):
        row_idcs, col_idcs = linear_sum_assignment(iou_matrix, maximize=True)
        iou_assigned = np.zeros(iou_matrix.shape[0])
        iou_assigned[row_idcs] = iou_matrix[row_idcs, col_idcs]
        
        idcs_assigned = np.zeros(iou_matrix.shape[0], dtype=int)
        idcs_assigned[row_idcs] = col_idcs
        return iou_assigned, idcs_assigned

    def _greedy_matching(self, label, iou_matrix):
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
                idcs.append(None)
                continue
            idx = iou_matrix[i].argmax()
            idcs.append(idx)
            ious.append(iou_matrix[i, idx])
            iou_matrix = np.delete(iou_matrix, idx, 1) # Delete columns of assigned object 
        return np.array(ious), np.array(idcs)

    @staticmethod
    def _calc_iou_matrix(preds, labels):
        """ 
        Generates iou matrix between all label/ground truth masks and all predicted masks for one image
        :param preds: predicted masks with shape (n_masks, width, height)
        :param labels: ground truth masks with shape (m_masks, width, height)
        width, height should be the same, but n_masks and m_masks can differ.
        """
        iou_matrix = np.zeros((labels.shape[0], preds.shape[0]))
        for i, g in enumerate(labels):
            for j, p in enumerate(preds):
                intersection = ((g>0) * (p>0)).sum()
                union = ((g>0) + (p>0)).sum()
                iou_matrix[i,j] = intersection.sum() / union.sum() if union>0 else 0.0
        return iou_matrix

    # def _hungarian_matching(self, iou_matrix): 
    #     """ 
    #     Find optimal prediction for each label mask
    #     Following
    #     https://www.hungarianalgorithm.com/ 
    #     """
    #     # 1. Negate and subtract, bc we want maximization of IoU
    #     iou_matrix = - iou_matrix + iou_matrix.max()
    #     n = max(iou_matrix.shape)
    #     # If more rows than cols
    #     subtract_cols = True if iou_matrix.shape[0] > iou_matrix.shape[1] else False
    #     # 2. Pad matrix with zeros if uneven number of label and pred masks
    #     iou_matrix = np.pad(iou_matrix, ((0,n-iou_matrix.shape[0]), (0, n - iou_matrix.shape[1])))

    #     # 3. Subtract row/col minima
    #     print(np.around(iou_matrix,3))
    #     iou_matrix -= iou_matrix.min(axis=1)[:,None,...] if not subtract_cols else iou_matrix.min(axis=0)
    #     print(np.around(iou_matrix,3))