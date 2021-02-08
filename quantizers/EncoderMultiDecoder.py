from torch import nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize

from quantizers import EncoderDecoder


class EncoderMultiDecoder( EncoderDecoder.EncoderDecoder ):
    def __init__(self, encoder, decoder, primary_loss, n_embed=1024, decay=0.8,
                 # the exponential moving average decay, lower means the dictionary will change faster
                 commitment=1., eps=1e-5):
        super().__init__(encoder, decoder, primary_loss, n_embed, decay, commitment, eps)
        self.decoder = nn.ModuleList(self.decoder)

    def decode(self, z):
        predictions = []
        for decoder in self.decoder:
            predictions.append( decoder( z ) )
        return predictions

    def calculate_prime_loss(self, y_hat_list, y):
        loss = 0
        for y_hat in y_hat_list:
            loss += self.primary_loss(y_hat, y)
        return loss / len(y_hat_list)


