
import time, helper
from torch.nn.utils import clip_grad_norm


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, optimizer, dictionary, config, best_loss):
        self.model = model
        self.dictionary = dictionary
        self.config = config

        self.optimizer = optimizer
        self.best_dev_loss = best_loss
        self.times_no_improvement = 0
        self.stop = False
        self.train_losses = []
        self.dev_losses = []

    def train_epochs(self, train_corpus, dev_corpus, start_epoch, n_epochs):
        """Trains model for n_epochs epochs"""
        for epoch in range(start_epoch, start_epoch + n_epochs):
            if not self.stop:
                print('\nTRAINING : Epoch ' + str((epoch + 1)))
                self.train(train_corpus)
                # training epoch completes, now do validation
                print('\nVALIDATING : Epoch ' + str((epoch + 1)))
                dev_loss = self.validate(dev_corpus)
                self.dev_losses.append(dev_loss)
                print('validation loss = %.4f' % dev_loss)
                # save model if dev loss goes down
                if self.best_dev_loss == -1 or self.best_dev_loss > dev_loss:
                    self.best_dev_loss = dev_loss
                    helper.save_checkpoint({
                        'epoch': (epoch + 1),
                        'state_dict': self.model.state_dict(),
                        'best_loss': self.best_dev_loss,
                        'optimizer': self.optimizer.state_dict(),
                    }, self.config.save_path + 'model_best.pth.tar')
                    self.times_no_improvement = 0
                else:
                    self.times_no_improvement += 1
                    # no improvement in validation loss for last n iterations, so stop training
                    if self.times_no_improvement == self.config.early_stop:
                        self.stop = True
                # save the train and development loss plot
                helper.save_plot(self.train_losses, self.config.save_path, 'training_loss_plot_', epoch + 1)
                helper.save_plot(self.dev_losses, self.config.save_path, 'dev_loss_plot_', epoch + 1)
            else:
                break

    def train(self, train_corpus):
        # Turn on training mode which enables dropout.
        self.model.train()

        # splitting the data in batches
        train_batches = helper.batchify(train_corpus.data, self.config.batch_size)
        print('number of train batches = ', len(train_batches))

        start = time.time()
        print_loss_total = 0
        plot_loss_total = 0

        num_batches = len(train_batches)
        for batch_no in range(1, num_batches + 1):
            # Clearing out all previous gradient computations.
            self.optimizer.zero_grad()
            videos, video_len, descriptions, des_len = helper.videos_to_tensor(train_batches[batch_no - 1],
                                                                               self.dictionary)
            if self.config.cuda:
                videos = videos.cuda()  # batch_size x max_images_per_video x num_image_features
                descriptions = descriptions.cuda()  # batch_size x max_description_length
                des_len = des_len.cuda()    # batch_size

            loss = self.model(videos, video_len, descriptions, des_len)
            # Important if we are using nn.DataParallel()
            if loss.size(0) > 1:
                loss = loss.mean()
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            clip_grad_norm(filter(lambda p: p.requires_grad, self.model.parameters()), self.config.max_norm)
            self.optimizer.step()

            print_loss_total += loss.data[0]
            plot_loss_total += loss.data[0]

            if batch_no % self.config.print_every == 0:
                print_loss_avg = print_loss_total / self.config.print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (
                    helper.show_progress(start, batch_no / num_batches), batch_no,
                    batch_no / num_batches * 100, print_loss_avg))

            if batch_no % self.config.plot_every == 0:
                plot_loss_avg = plot_loss_total / self.config.plot_every
                self.train_losses.append(plot_loss_avg)
                plot_loss_total = 0

    def validate(self, dev_corpus):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        dev_batches = helper.batchify(dev_corpus.data, self.config.batch_size)
        print('number of dev batches = ', len(dev_batches))

        dev_loss = 0
        num_batches = len(dev_batches)
        for batch_no in range(1, num_batches + 1):
            videos, video_len, descriptions, des_len = helper.videos_to_tensor(dev_batches[batch_no - 1],
                                                                               self.dictionary)
            if self.config.cuda:
                videos = videos.cuda()  # batch_size x max_images_per_video x num_image_features
                descriptions = descriptions.cuda()  # batch_size x max_description_length
                des_len = des_len.cuda()    # batch_size

            loss = self.model(videos, video_len, descriptions, des_len)
            if loss.size(0) > 1:
                loss = loss.mean()
            dev_loss += loss.data[0]

        return dev_loss / num_batches
