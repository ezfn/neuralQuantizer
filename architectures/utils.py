from torch import nn


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        '''
        TODO: the first dimension is the data batch_size
        so we need to decide how the input shape should be like
        '''
        return input.view((input.shape[0], -1))