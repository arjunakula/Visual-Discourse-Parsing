
import os, helper
from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.pad_token = '<pad>'
        self.idx2word.append(self.pad_token)
        self.word2idx[self.pad_token] = len(self.idx2word) - 1

    def build_dict(self, videos, max_words=50000):
        word_count = Counter()
        for video in videos.data:
            word_count.update(video.description)

        most_common = word_count.most_common(max_words)
        for (index, w) in enumerate(most_common):
            self.idx2word.append(w[0])
            self.word2idx[w[0]] = len(self.idx2word) - 1

    def contains(self, word):
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)


class Video(object):
    def __init__(self):
        self.img_features = []
        self.description = []

    def add_image_features(self, feats):
        self.img_features.append(list(map(float, feats.split())))

    def add_description(self, text, tokenize):
        if self.description:
            print('already description is assigned!')
        else:
            self.description = ['<s>'] + helper.tokenize(text, tokenize) + ['</s>']


class Corpus(object):
    def __init__(self, _tokenize):
        self.tokenize = _tokenize
        self.data = []

    def parse(self, in_file, max_example=None):
        """Parses the content of a file."""
        assert os.path.exists(in_file)

        with open(in_file, 'r') as f:
            video = Video()
            lastline = None
            for line in f:
                line = line.strip()
                if line:
                    if lastline:
                        video.add_image_features(lastline)
                    lastline = line
                else:
                    video.add_description(lastline, self.tokenize)
                    self.data.append(video)
                    video = Video()
                    lastline = None
                    if len(self.data) == max_example:
                        break
