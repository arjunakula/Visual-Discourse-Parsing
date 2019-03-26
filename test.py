###############################################################################
# Author: Wasi Ahmad
# Project: Transform video to natural language
# Date Created: 12/04/2017
#
# File Description: This script visualizes query and document representations.
###############################################################################

import torch, helper, util, os, numpy, data, multi_bleu
from seq2seq import Seq2Seq
from torch.autograd import Variable

args = util.get_args()
# Set the random seed manually for reproducibility.
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)


def generate_video_description(model, videos, descriptions, des_len):
    # encode the video features
    encoded_videos = model.encoder(videos, video_len)

    if model.config.pool_type == 'max':
        hidden_states = torch.max(encoded_videos, 1)[0].squeeze()
    elif model.config.pool_type == 'mean':
        hidden_states = torch.sum(encoded_videos, 1).squeeze() / encoded_videos.size(1)
    elif model.config.pool_type == 'last':
        if model.num_directions == 2:
            hidden_states = torch.cat(
                (encoded_videos[:, -1, :model.config.nhid_enc], encoded_videos[:, -1, model.config.nhid_enc:]), 1)
        else:
            hidden_states = encoded_videos[:, -1, :]

    # Initialize hidden states of decoder with the last hidden states of the encoder
    if model.config.model is 'LSTM':
        cell_states = Variable(torch.zeros(*hidden_states.size()))
        if model.config.cuda:
            cell_states = cell_states.cuda()
        decoder_hidden = (hidden_states.unsqueeze(0).contiguous(), cell_states.unsqueeze(0).contiguous())
    else:
        decoder_hidden = hidden_states.unsqueeze(0).contiguous()

    sos_token_index = model.dictionary.word2idx['<s>']
    eos_token_index = model.dictionary.word2idx['</s>']

    # First input of the decoder is the sentence start token
    decoder_input = Variable(torch.LongTensor([sos_token_index]))
    decoded_words = []
    for di in range(50):
        if model.config.cuda:
            decoder_input = decoder_input.cuda()
        embedded_decoder_input = model.embedding(decoder_input).unsqueeze(1)
        decoder_output, decoder_hidden = model.decoder(embedded_decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == eos_token_index:
            decoded_words.append('</s>')
            break
        else:
            decoded_words.append(model.dictionary.idx2word[ni])
        decoder_input = Variable(torch.LongTensor([ni]))

    return " ".join(decoded_words[:-1])


if __name__ == "__main__":
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file,
                                                   dictionary.word2idx)
    model = Seq2Seq(dictionary, embeddings_index, args)
    print(model)
    if args.cuda:
        model = model.cuda()
    helper.load_model_states_from_checkpoint(model, os.path.join(args.save_path, 'model_best.pth.tar'), 'state_dict',
                                             args.cuda)
    print('model, embedding index and dictionary loaded.')
    model.eval()

    # load the test dataset
    test_corpus = data.Corpus(args.tokenize)
    test_corpus.parse(args.data + 'data.test')
    print('test set size = ', len(test_corpus.data))

    targets = []
    candidates = []

    fw = open(args.save_path + 'predictions.txt', 'w')
    for video in test_corpus.data:
        videos, video_len, descriptions, des_len = helper.videos_to_tensor([video], dictionary)
        if args.cuda:
            videos = videos.cuda()  # batch_size x max_images_per_video x num_image_features
            descriptions = descriptions.cuda()  # batch_size x max_description_length
            des_len = des_len.cuda()  # batch_size

        target = generate_video_description(model, videos, descriptions, des_len)
        candidate = " ".join(video.description[1:-1])
        targets.append(target)
        candidates.append(candidate)
        fw.write(candidate + '\t' + target + '\n')
    fw.close()

    print("target size = ", len(targets))
    print("candidate size = ", len(candidates))
    multi_bleu.print_multi_bleu(targets, candidates)
