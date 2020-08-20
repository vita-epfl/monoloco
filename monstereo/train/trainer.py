# pylint: disable=too-many-statements

"""
Training and evaluation of a neural network which predicts 3D localization and confidence intervals
given 2d joints
"""

import copy
import os
import datetime
import logging
from collections import defaultdict
import sys
import time
import warnings
from itertools import chain

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from .datasets import KeypointsDataset
from .losses import CompositeLoss, MultiTaskLoss, AutoTuneMultiTaskLoss
from ..network import extract_outputs, extract_labels
from ..network.architectures import SimpleModel
from ..utils import set_logger


class Trainer:
    # Constants
    VAL_BS = 10000

    tasks = ('d', 'x', 'y', 'h', 'w', 'l', 'ori', 'aux')
    val_task = 'd'
    lambdas = (1, 1, 1, 1, 1, 1, 1, 1)

    def __init__(self, joints, epochs=100, bs=256, dropout=0.2, lr=0.002,
                 sched_step=20, sched_gamma=1, hidden_size=256, n_stage=3, r_seed=1, n_samples=100,
                 monocular=False, save=False, print_loss=True):
        """
        Initialize directories, load the data and parameters for the training
        """

        # Initialize directories and parameters
        dir_out = os.path.join('data', 'models')
        if not os.path.exists(dir_out):
            warnings.warn("Warning: output directory not found, the model will not be saved")
        dir_logs = os.path.join('data', 'logs')
        if not os.path.exists(dir_logs):
            warnings.warn("Warning: default logs directory not found")
        assert os.path.exists(joints), "Input file not found"

        self.joints = joints
        self.num_epochs = epochs
        self.save = save
        self.print_loss = print_loss
        self.monocular = monocular
        self.lr = lr
        self.sched_step = sched_step
        self.sched_gamma = sched_gamma
        self.clusters = ['10', '20', '30', '50', '>50']
        self.hidden_size = hidden_size
        self.n_stage = n_stage
        self.dir_out = dir_out
        self.n_samples = n_samples
        self.r_seed = r_seed
        self.auto_tune_mtl = False

        # Select the device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print('Device: ', self.device)
        torch.manual_seed(r_seed)
        if use_cuda:
            torch.cuda.manual_seed(r_seed)

        # Remove auxiliary task if monocular
        if self.monocular and self.tasks[-1] == 'aux':
            self.tasks = self.tasks[:-1]
            self.lambdas = self.lambdas[:-1]

        losses_tr, losses_val = CompositeLoss(self.tasks)()

        if self.auto_tune_mtl:
            self.mt_loss = AutoTuneMultiTaskLoss(losses_tr, losses_val, self.lambdas, self.tasks)
        else:
            self.mt_loss = MultiTaskLoss(losses_tr, losses_val, self.lambdas, self.tasks)
        self.mt_loss.to(self.device)

        if not self.monocular:
            input_size = 68
            output_size = 10
        else:
            input_size = 34
            output_size = 9

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        name_out = 'ms-' + now_time
        if self.save:
            self.path_model = os.path.join(dir_out, name_out + '.pkl')
            self.logger = set_logger(os.path.join(dir_logs, name_out))
            self.logger.info("Training arguments: \nepochs: {} \nbatch_size: {} \ndropout: {}"
                             "\nmonocular: {} \nlearning rate: {} \nscheduler step: {} \nscheduler gamma: {}  "
                             "\ninput_size: {} \noutput_size: {}\nhidden_size: {} \nn_stages: {} "
                             "\nr_seed: {} \nlambdas: {} \ninput_file: {}"
                             .format(epochs, bs, dropout, self.monocular, lr, sched_step, sched_gamma, input_size,
                                     output_size, hidden_size, n_stage, r_seed, self.lambdas, self.joints))
        else:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

        # Dataloader
        self.dataloaders = {phase: DataLoader(KeypointsDataset(self.joints, phase=phase),
                                              batch_size=bs, shuffle=True) for phase in ['train', 'val']}

        self.dataset_sizes = {phase: len(KeypointsDataset(self.joints, phase=phase))
                              for phase in ['train', 'val']}

        # Define the model
        self.logger.info('Sizes of the dataset: {}'.format(self.dataset_sizes))
        print(">>> creating model")

        self.model = SimpleModel(input_size=input_size, output_size=output_size, linear_size=hidden_size,
                                 p_dropout=dropout, num_stage=self.n_stage, device=self.device)
        self.model.to(self.device)
        print(">>> model params: {:.3f}M".format(sum(p.numel() for p in self.model.parameters()) / 1000000.0))
        print(">>> loss params: {}".format(sum(p.numel() for p in self.mt_loss.parameters())))

        # Optimizer and scheduler
        all_params = chain(self.model.parameters(), self.mt_loss.parameters())
        self.optimizer = torch.optim.Adam(params=all_params, lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.sched_step, gamma=self.sched_gamma)

    def train(self):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 1e6
        best_training_acc = 1e6
        best_epoch = 0
        epoch_losses = defaultdict(lambda: defaultdict(list))
        for epoch in range(self.num_epochs):
            running_loss = defaultdict(lambda: defaultdict(int))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                for inputs, labels, _, _ in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            outputs = self.model(inputs)
                            loss, loss_values = self.mt_loss(outputs, labels, phase=phase)
                            self.optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                            self.optimizer.step()
                            self.scheduler.step()

                        else:
                            outputs = self.model(inputs)
                        with torch.no_grad():
                            loss_eval, loss_values_eval = self.mt_loss(outputs, labels, phase='val')
                            self.epoch_logs(phase, loss_eval, loss_values_eval, inputs, running_loss)

            self.cout_values(epoch, epoch_losses, running_loss)

            # deep copy the model

            if epoch_losses['val'][self.val_task][-1] < best_acc:
                best_acc = epoch_losses['val'][self.val_task][-1]
                best_training_acc = epoch_losses['train']['all'][-1]
                best_epoch = epoch
                best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        print('\n\n' + '-' * 120)
        self.logger.info('Training:\nTraining complete in {:.0f}m {:.0f}s'
                         .format(time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Best training Accuracy: {:.3f}'.format(best_training_acc))
        self.logger.info('Best validation Accuracy for {}: {:.3f}'.format(self.val_task, best_acc))
        self.logger.info('Saved weights of the model at epoch: {}'.format(best_epoch))

        if self.print_loss:
            print_losses(epoch_losses)

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return best_epoch

    def epoch_logs(self, phase, loss, loss_values, inputs, running_loss):

        running_loss[phase]['all'] += loss.item() * inputs.size(0)
        for i, task in enumerate(self.tasks):
            running_loss[phase][task] += loss_values[i].item() * inputs.size(0)

    def evaluate(self, load=False, model=None, debug=False):

        # To load a model instead of using the trained one
        if load:
            self.model.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

        # Average distance on training and test set after unnormalizing
        self.model.eval()
        dic_err = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))  # initialized to zero
        dic_err['val']['sigmas'] = [0.] * len(self.tasks)
        dataset = KeypointsDataset(self.joints, phase='val')
        size_eval = len(dataset)
        start = 0
        with torch.no_grad():
            for end in range(self.VAL_BS, size_eval + self.VAL_BS, self.VAL_BS):
                end = end if end < size_eval else size_eval
                inputs, labels, _, _ = dataset[start:end]
                start = end
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Debug plot for input-output distributions
                if debug:
                    debug_plots(inputs, labels)
                    sys.exit()

                # Forward pass
                outputs = self.model(inputs)
                self.compute_stats(outputs, labels, dic_err['val'], size_eval, clst='all')

            self.cout_stats(dic_err['val'], size_eval, clst='all')
            # Evaluate performances on different clusters and save statistics
            for clst in self.clusters:
                inputs, labels, size_eval = dataset.get_cluster_annotations(clst)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass on each cluster
                outputs = self.model(inputs)
                self.compute_stats(outputs, labels, dic_err['val'], size_eval, clst=clst)
                self.cout_stats(dic_err['val'], size_eval, clst=clst)

        # Save the model and the results
        if self.save and not load:
            torch.save(self.model.state_dict(), self.path_model)
            print('-' * 120)
            self.logger.info("\nmodel saved: {} \n".format(self.path_model))
        else:
            self.logger.info("\nmodel not saved\n")

        return dic_err, self.model

    def compute_stats(self, outputs, labels, dic_err, size_eval, clst):
        """Compute mean, bi and max of torch tensors"""

        loss, loss_values = self.mt_loss(outputs, labels, phase='val')
        rel_frac = outputs.size(0) / size_eval

        tasks = self.tasks[:-1] if self.tasks[-1] == 'aux' else self.tasks  # Exclude auxiliary

        for idx, task in enumerate(tasks):
            dic_err[clst][task] += float(loss_values[idx].item()) * (outputs.size(0) / size_eval)

        # Distance
        errs = torch.abs(extract_outputs(outputs)['d'] - extract_labels(labels)['d'])

        assert rel_frac > 0.99, "Variance of errors not supported with partial evaluation"

        # Uncertainty
        bis = extract_outputs(outputs)['bi'].cpu()
        bi = float(torch.mean(bis).item())
        bi_perc = float(torch.sum(errs <= bis)) / errs.shape[0]
        dic_err[clst]['bi'] += bi * rel_frac
        dic_err[clst]['bi%'] += bi_perc * rel_frac
        dic_err[clst]['std'] = errs.std()

        # (Don't) Save auxiliary task results
        if self.monocular:
            dic_err[clst]['aux'] = 0
            dic_err['sigmas'].append(0)
        else:
            acc_aux = get_accuracy(extract_outputs(outputs)['aux'], extract_labels(labels)['aux'])
            dic_err[clst]['aux'] += acc_aux * rel_frac

        if self.auto_tune_mtl:
            assert len(loss_values) == 2 * len(self.tasks)
            for i, _ in enumerate(self.tasks):
                dic_err['sigmas'][i] += float(loss_values[len(tasks) + i + 1].item()) * rel_frac

    def cout_stats(self, dic_err, size_eval, clst):
        if clst == 'all':
            print('-' * 120)
            self.logger.info("Evaluation, val set: \nAv. dist D: {:.2f} m with bi {:.2f} ({:.1f}%), \n"
                             "X: {:.1f} cm,  Y: {:.1f} cm \nOri: {:.1f}  "
                             "\n H: {:.1f} cm, W: {:.1f} cm, L: {:.1f} cm"
                             "\nAuxiliary Task: {:.1f} %, "
                             .format(dic_err[clst]['d'], dic_err[clst]['bi'], dic_err[clst]['bi%'] * 100,
                                     dic_err[clst]['x'] * 100, dic_err[clst]['y'] * 100,
                                     dic_err[clst]['ori'], dic_err[clst]['h'] * 100, dic_err[clst]['w'] * 100,
                                     dic_err[clst]['l'] * 100, dic_err[clst]['aux'] * 100))
            if self.auto_tune_mtl:
                self.logger.info("Sigmas: Z: {:.2f}, X: {:.2f}, Y:{:.2f}, H: {:.2f}, W: {:.2f}, L: {:.2f}, ORI: {:.2f}"
                                 " AUX:{:.2f}\n"
                                 .format(*dic_err['sigmas']))
        else:
            self.logger.info("Val err clust {} --> D:{:.2f}m,  bi:{:.2f} ({:.1f}%), STD:{:.1f}m   X:{:.1f} Y:{:.1f}  "
                             "Ori:{:.1f}d,   H: {:.0f} W: {:.0f} L:{:.0f}  for {} pp. "
                             .format(clst, dic_err[clst]['d'], dic_err[clst]['bi'], dic_err[clst]['bi%'] * 100,
                                     dic_err[clst]['std'], dic_err[clst]['x'] * 100, dic_err[clst]['y'] * 100,
                                     dic_err[clst]['ori'], dic_err[clst]['h'] * 100, dic_err[clst]['w'] * 100,
                                     dic_err[clst]['l'] * 100, size_eval))

    def cout_values(self, epoch, epoch_losses, running_loss):

        string = '\r' + '{:.0f} '
        format_list = [epoch]
        for phase in running_loss:
            string = string + phase[0:1].upper() + ':'
            for el in running_loss['train']:
                loss = running_loss[phase][el] / self.dataset_sizes[phase]
                epoch_losses[phase][el].append(loss)
                if el == 'all':
                    string = string + ':{:.1f}  '
                    format_list.append(loss)
                elif el in ('ori', 'aux'):
                    string = string + el + ':{:.1f}  '
                    format_list.append(loss)
                else:
                    string = string + el + ':{:.0f}  '
                    format_list.append(loss * 100)

        if epoch % 10 == 0:
            print(string.format(*format_list))


def debug_plots(inputs, labels):
    inputs_shoulder = inputs.cpu().numpy()[:, 5]
    inputs_hip = inputs.cpu().numpy()[:, 11]
    labels = labels.cpu().numpy()
    heights = inputs_hip - inputs_shoulder
    plt.figure(1)
    plt.hist(heights, bins='auto')
    plt.show()
    plt.figure(2)
    plt.hist(labels, bins='auto')
    plt.show()


def print_losses(epoch_losses):
    for idx, phase in enumerate(epoch_losses):
        for idx_2, el in enumerate(epoch_losses['train']):
            plt.figure(idx + idx_2)
            plt.plot(epoch_losses[phase][el][10:], label='{} Loss: {}'.format(phase, el))
            plt.savefig('figures/{}_loss_{}.png'.format(phase, el))
            plt.close()


def get_accuracy(outputs, labels):
    """From Binary cross entropy outputs to accuracy"""

    mask = outputs >= 0.5
    accuracy = 1. - torch.mean(torch.abs(mask.float() - labels)).item()
    return accuracy
