from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter

class ClusterContrastTrainer(object):
    def __init__(self, args, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.args = args
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, logger=None):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)

            # forward
            f_out, f_out_gap = self._forward(inputs)

            loss = 0
            loss_cc = self.memory(f_out, labels)
            loss += loss_cc * self.args.loss_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Lr {:.10f} \t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              optimizer.param_groups[-1]['lr'],
                              losses.val, losses.avg,
                              ))
            
            if logger is not None:
                logger.add_scalar('loss', loss.item(), i + epoch * self.args.iters)

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

