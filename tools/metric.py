import torch


def cosine(inputs_1, inputs_2):
    '''

    :param inputs_1: torch.tensor, 2d
    :param inputs_2: torch.tensor, 2d
    :param labels: 2d,
    :return:
    '''

    bs1 = inputs_1.size()[0]
    bs2 = inputs_2.size()[0]

    frac_up = torch.matmul(inputs_1, inputs_2.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(inputs_1, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(inputs_2, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down

    return cosine