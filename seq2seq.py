
import torch, helper
import torch.nn as nn
from torch.autograd import Variable
from nn_layer import EmbeddingLayer, Encoder, Decoder


class Seq2Seq(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""
        super(Seq2Seq, self).__init__()
        self.config = args
        self.num_directions = 2 if self.config.bidirection else 1
        self.dictionary = dictionary

        self.embedding = EmbeddingLayer(len(self.dictionary), self.config)
        self.embedding.init_embedding_weights(self.dictionary, embedding_index, self.config.emsize)

        self.encoder = Encoder(self.config.input_size, self.config.nhid_enc, self.config.bidirection, self.config)
        self.decoder = Decoder(self.config.emsize, self.config.nhid_enc * self.num_directions, len(self.dictionary),
                               self.config)

    @staticmethod
    def compute_decoding_loss(logits, target, seq_idx, length):
        losses = -torch.gather(logits, dim=1, index=target.unsqueeze(1)).squeeze()
        mask = helper.mask(length, seq_idx)  # mask: batch x 1
        losses = losses * mask.float()
        num_non_zero_elem = torch.nonzero(mask.data).size()
        if not num_non_zero_elem:
            return losses.sum(), 0
        else:
            return losses.sum(), num_non_zero_elem[0]

    def forward(self, videos, video_len, decoder_input, target_length):
        # encode the video features
        encoded_videos = self.encoder(videos, video_len)

        if self.config.pool_type == 'max':
            hidden_states = torch.max(encoded_videos, 1)[0].squeeze()
        elif self.config.pool_type == 'mean':
            hidden_states = torch.sum(encoded_videos, 1).squeeze() / encoded_videos.size(1)
        elif self.config.pool_type == 'last':
            if self.num_directions == 2:
                hidden_states = torch.cat(
                    (encoded_videos[:, -1, :self.config.nhid_enc], encoded_videos[:, -1, self.config.nhid_enc:]), 1)
            else:
                hidden_states = encoded_videos[:, -1, :]

        # Initialize hidden states of decoder with the last hidden states of the encoder
        if self.config.model is 'LSTM':
            cell_states = Variable(torch.zeros(*hidden_states.size()))
            if self.config.cuda:
                cell_states = cell_states.cuda()
            decoder_hidden = (hidden_states.unsqueeze(0).contiguous(), cell_states.unsqueeze(0).contiguous())
        else:
            decoder_hidden = hidden_states.unsqueeze(0).contiguous()

        decoding_loss = 0
        total_local_decoding_loss_element = 0
        for idx in range(decoder_input.size(1) - 1):
            input_variable = decoder_input[:, idx]
            embedded_decoder_input = self.embedding(input_variable).unsqueeze(1)
            decoder_output, decoder_hidden = self.decoder(embedded_decoder_input, decoder_hidden)
            target_variable = decoder_input[:, idx + 1]

            local_loss, num_local_loss = self.compute_decoding_loss(decoder_output, target_variable, idx, target_length)
            decoding_loss += local_loss
            total_local_decoding_loss_element += num_local_loss

        if total_local_decoding_loss_element > 0:
            decoding_loss = decoding_loss / total_local_decoding_loss_element

        return decoding_loss
