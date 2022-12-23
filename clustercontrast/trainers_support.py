from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch
import torch.nn.functional as F
import numpy as np
import math
from .losses import CriterionLP

class ClusterContrastTrainerSupport(object):
    def __init__(self, args, encoder, memory=None):
        super(ClusterContrastTrainerSupport, self).__init__()
        self.args = args
        self.encoder = encoder
        self.memory = memory
        self.criterion_lp = CriterionLP(args)

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, logger=None):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_cc = AverageMeter()
        losses_lp = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            ### define the degree direction of PLI
            cur_lambda = np.log((math.e - 1) * (epoch * train_iters + i) / (train_iters * self.args.epochs) + 1) * self.args.support_base_lambda * 0.5

            # input data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)

            # forward
            f_out, _ = self._forward(inputs)

            ###### generate support samples
            f_out_support, _ = self.generate_hard_samples(f_out, labels, cur_lambda)
            support_labels = labels.repeat_interleave(self.args.topk)
            
            loss = 0
            if self.args.use_support:
                loss_cc = self.memory(torch.cat((f_out, f_out_support), dim=0), torch.cat((labels, support_labels), dim=0))
            else:
                loss_cc = self.memory(f_out, labels)
            loss += loss_cc * self.args.loss_weight

            # calculate label_preserving (LP) loss
            loss_lp = self.criterion_lp(f_out, f_out_support, labels, support_labels)
            loss += loss_lp * self.args.lp_loss_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            losses_cc.update(loss_cc.item())
            losses_lp.update(loss_lp.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Lr {:.10f} \t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss_cc {:.3f} ({:.3f})\t'
                      'Loss_lp {:.3f} ({:.3f})\t'
                      'CurrentLambda {:.10f} \t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              optimizer.param_groups[-1]['lr'],
                              losses.val, losses.avg,
                              losses_cc.val, losses_cc.avg,
                              losses_lp.val, losses_lp.avg,
                              cur_lambda))
            
            if logger is not None:
                logger.add_scalar('loss', loss.item(), i + epoch * self.args.iters)
                logger.add_scalar('loss_cc', loss_cc.item(), i + epoch * self.args.iters)
                logger.add_scalar('loss_lp', loss_lp.item(), i + epoch * self.args.iters)

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

    def generate_hard_samples(self, feats, labels, cur_lambda):
        B = feats.size(0)
        C = feats.size(-1)
        nearest_index, nearest_sim = self.find_nearest_center(feats, labels)
        nearest_center = self.memory.features[nearest_index]
        target_center = self.memory.features[labels]
        target_center = target_center.view(B, 1, -1)

        cur_lambda = cur_lambda * np.ones(self.args.topk)
        cur_lambda = torch.tensor(cur_lambda).cuda().float().view(1, self.args.topk, 1)

        supports = feats.view(B, 1, -1) + (nearest_center - target_center) * cur_lambda * 0.5
        supports = supports.reshape(B * self.args.topk, C)
        supports = F.normalize(supports, dim=1).cuda()
        return supports, nearest_sim
    
    def find_nearest_center(self, feats, labels):
        B = feats.size(0)
        sim = feats.detach().mm(self.memory.features.t().detach())
        sim_sort, indexes = torch.sort(sim, dim=1, descending=True)
        false_labels = indexes != labels.view(-1, 1)
        indexes = indexes[false_labels].view(B, -1)
        sim_sort = sim_sort[false_labels].view(B, -1)
        indexes = indexes[:, :self.args.topk]
        sim_sort = sim_sort[:, :self.args.topk]
        sim_sort /= sim_sort[:, 0].view(-1, 1)
        return indexes, sim_sort
