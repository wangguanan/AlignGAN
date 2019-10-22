import torch
import torch.nn as nn
import numpy as np

from .metric import cosine


class RankingLoss:

    def __init__(self, soft_bh):
        self.soft_bh = soft_bh

    def __call__(self, *args, **kwargs):
        pass

    def _label2similarity(sekf, label1, label2):
        '''
        compute similarity matrix of label1 and label2
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [n]
        :return: torch.Tensor, [m, n], {0, 1}
        '''
        m, n = len(label1), len(label2)
        l1 = label1.view(m, 1).expand([m, n])
        l2 = label2.view(n, 1).expand([n, m]).t()
        similarity = l1 == l2
        return similarity

    def _batch_hard(self, mat_distance, mat_similarity, more_similar):

        if more_similar is 'smaller':
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
            hard_p = sorted_mat_distance[:, self.soft_bh[0]]
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
            hard_n = sorted_mat_distance[:, self.soft_bh[1]]
            return hard_p, hard_n

        elif more_similar is 'larger':
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1, descending=False)
            hard_p = sorted_mat_distance[:, self.soft_bh[0]]
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
            hard_n = sorted_mat_distance[:, self.soft_bh[1]]
            return hard_p, hard_n



class TripletLoss(RankingLoss):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    '''

    def __init__(self, margin, soft_bh, metric):
        '''

        :param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
        :param bh: batch hard
        :param metric: l2 distance or cosine distance
        '''

        self.margin = margin
        self.margin_loss = nn.MarginRankingLoss(margin=margin)
        self.soft_bh = soft_bh
        self.metric = metric

    def __call__(self, emb1, emb2, emb3, label1, label2, label3):
        '''

        :param emb1: torch.Tensor, [m, dim]
        :param emb2: torch.Tensor, [n, dim]
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [b]
        :return:
        '''

        if self.metric is 'cosine':
            mat_dist = cosine(emb1, emb2)
            mat_sim = self._label2similarity(label1, label2)
            hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            mat_dist = cosine(emb1, emb3)
            mat_sim = self._label2similarity(label1, label3)
            _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            margin_label = -torch.ones_like(hard_p)

        return self.margin_loss(hard_n, hard_p, margin_label)



class TripletLossWithMat(RankingLoss):

    '''
    the inputs (distances not embeddings, similarity not labels) are martix
    '''

    def __init__(self, margin, soft_bh, more_similar):

        self.margin = margin
        self.margin_loss = nn.MarginRankingLoss(margin=margin)
        self.soft_bh = soft_bh
        self.more_similar = more_similar

    def __call__(self, distance4positive, similarity4positive, distance4negative, similarity4negative):

        hard_p, _ = self._batch_hard(distance4positive, similarity4positive, more_similar=self.more_similar)
        _, hard_n = self._batch_hard(distance4negative, similarity4negative, more_similar=self.more_similar)
        margin_label = -torch.ones_like(hard_p)
        return self.margin_loss(hard_n, hard_p, margin_label)








