import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()


def R_set(x):
    '''Create an indicator matrix of risk sets, where T_j >= T_i.
    Note that the input data have been sorted in descending order.
    Input:
        x: a PyTorch tensor that the number of rows is equal to the number of samples.
    Output:
        indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
    '''
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)

    return (indicator_matrix)

class NegativeLogLikelihood(nn.Module):
    def __init(self, reduction = 'mean'):
        super(NegativeLogLikelihood, self).__init__()
        self.reduction = reduction

    def forward(self, pred, label):
        '''Calculate the average Cox negative partial log-likelihood.
                Input:
                    pred: linear predictors from trained model.
                    ytime: true survival time from load_data().
                    yevent: true censoring status from load_data().
                Output:
                    cost: the cost that is to be minimized.
        '''
        ytime, yevent = label[:, 0], label[:, 1]
        n_observed = yevent.sum(0)
        if use_cuda:
            ytime_indicator = R_set(ytime).cuda()
        else:
            ytime_indicator = R_set(ytime)
        risk_set_sum = ytime_indicator.mm(torch.exp(pred))
        diff = pred - torch.log(risk_set_sum)
        yevent = yevent.view(len(pred), -1).to(dtype=torch.float)
        sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
        cost = (- (sum_diff_in_observed / n_observed)).view(1, )

        return (cost)

