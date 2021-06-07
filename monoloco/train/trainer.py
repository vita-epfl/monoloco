# pylint: disable=too-many-statements

"""
Training and evaluation of a neural network that, given 2D joints, estimates:
- 3D localization and confidence intervals
- Orientation
- Bounding box dimensions
"""

import copy
import os
import datetime
import logging
from collections import defaultdict
import sys
import time
from itertools import chain

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['figure.dpi'] = 300
except ImportError:
    plt = None

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from .. import __version__
from .datasets import KeypointsDataset
from .losses import CompositeLoss, MultiTaskLoss, ConsistencyLoss
from ..network import extract_labels, extract_outputs
from ..network.architectures import MonStereoModel, MonoLocoPPModel, TwoBlocks
from ..utils import set_logger


class Trainer:
    # Constants
    VAL_BS = 100000
    tasks = []
    lambdas = []
    tasks_1 = ('d', 'x', 'y')
    tasks_2 = ('h', 'w', 'l')
    tasks_h = ('h',)
    val_task_1 = 'd'
    val_task_h = 'h'
    lambdas_1 = (1, 1, 1)
    lambdas_2 = (1, 1, 1)
    lambdas_h = (1, )
    # lambdas = (0, 0, 0, 0, 0, 0, 1, 0)
    clusters = ['10', '20', '30', '40']
    input_size = dict(mono=34, stereo=68)
    output_size_1 = len(tasks_1)
    output_size_2 = len(tasks_2)
    if 'd' in tasks_1:
        output_size_1 += 1
    if 'ori' in tasks_1:
        output_size_1 += 1
    if 'd' in tasks_2:
        output_size_2 += 1
    if 'ori' in tasks_2:
        output_size_2 += 1
    output_size_1 = dict(mono=output_size_1, stereo=output_size_1 + 1)
    output_size_2 = dict(mono=output_size_2, stereo=output_size_2 + 1)
    output_size = output_size_2  # for logger
    dir_figures = os.path.join('figures', 'losses')

    def __init__(self, args):
        """
        Initialize directories, load the data and parameters for the training
        """

        assert os.path.exists(args.joints), "Input file not found"
        self.mode = args.mode
        self.joints = args.joints
        self.num_epochs = args.epochs
        self.no_save = args.no_save
        self.print_loss = args.print_loss
        self.lr = args.lr
        self.sched_step = args.sched_step
        self.sched_gamma = args.sched_gamma
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.n_stage = args.n_stage
        self.r_seed = args.r_seed
        self.auto_tune_mtl = args.auto_tune_mtl

        # Select path out
        if args.out:
            self.path_out = args.out  # full path without extension
            dir_out, _ = os.path.split(self.path_out)
        else:
            dir_out = os.path.join('data', 'outputs')
            name = 'monoloco_pp' if self.mode == 'mono' else 'monstereo'
            now = datetime.datetime.now()
            now_time = now.strftime("%Y%m%d-%H%M")[2:]
            name_out = name + '-' + now_time + '.pkl'
            self.path_out = os.path.join(dir_out, name_out)
        assert os.path.exists(dir_out), "Directory to save the model not found"
        print(self.path_out)
        # Select the device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print('Device: ', self.device)
        torch.manual_seed(self.r_seed)
        if use_cuda:
            torch.cuda.manual_seed(self.r_seed)

        # Remove auxiliary task if monocular
        if self.mode == 'stereo':
            self.tasks.append('aux')
            self.lambdas.append(1)

        # Dataloader
        self.dataloaders = {phase: DataLoader(KeypointsDataset(self.joints, phase=phase),
                                              batch_size=args.bs, shuffle=True) for phase in ['train', 'val']}

        self.dataset_sizes = {phase: len(KeypointsDataset(self.joints, phase=phase))
                              for phase in ['train', 'val']}
        self.dataset_version = KeypointsDataset(self.joints, phase='train').get_version()

        self._set_logger(args)

        # Define the model
        self.logger.info('Sizes of the dataset: {}'.format(self.dataset_sizes))
        self.scheduler = None

    def train(self):
        # 1) PRE-TRAIN CONSISTENCY
        since = time.time()

        best_acc_h = 1e6
        best_training_acc_h = 1e6
        best_epoch_h = 0
        epoch_losses_h = defaultdict(lambda: defaultdict(list))

        self.model_h = TwoBlocks(input_size=3, output_size=1)
        self.model_h.to(self.device)
        best_model_wts_h = copy.deepcopy(self.model_h.state_dict())
        losses_tr_h, losses_val_h = CompositeLoss(self.tasks_h)()
        self.loss_h = ConsistencyLoss(losses_tr_h, losses_val_h, self.lambdas_h, self.tasks_h)
        self.loss_h.to(self.device)
        print(">>> model params: {:.3f}M".format(sum(p.numel() for p in self.model_h.parameters()) / 1000000.0))

        all_params = chain(
            self.model_h.parameters(),
            self.loss_h.parameters(),
        )
        self.optimizer = torch.optim.Adam(params=all_params, lr=self.lr*2)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.sched_step, gamma=self.sched_gamma)

        for epoch in range(50):
            running_loss_h = defaultdict(lambda: defaultdict(int))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model_h.train()
                else:
                    self.model_h.eval()

                for inputs, labels, _, _ in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    inputs_y = inputs[:, 1::2]
                    h_max, _ = torch.max(inputs_y, dim=1)
                    h_min, _ = torch.min(inputs_y, dim=1)
                    h = h_max - h_min
                    h = h.reshape(-1, 1)
                    h_min = h_min.reshape(-1, 1)
                    inputs_h = torch.cat((labels[:, 3:4], h, h_min), dim=1)
                    labels_h = labels[:, 4:5]

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            self.optimizer.zero_grad()
                            outputs_h = self.model_h(inputs_h)
                            loss_h, _ = self.loss_h(outputs_h, labels_h)
                            loss_h.backward()
                            torch.nn.utils.clip_grad_norm_(self.model_h.parameters(), 3)
                            self.optimizer.step()
                            self.scheduler.step()

                        else:
                            outputs_h = self.model_h(inputs_h)
                        with torch.no_grad():
                            loss_eval_h, loss_values_eval_h = self.loss_h(outputs_h, labels_h, phase='val')
                            epoch_logs(running_loss_h, self.tasks_h, phase, loss_eval_h, loss_values_eval_h, inputs_h)
            cout_values(epoch, epoch_losses_h, running_loss_h, self.dataset_sizes[phase])

            # deep copy the model
            if epoch_losses_h['val'][self.val_task_h][-1] < best_acc_h:
                best_acc_h = epoch_losses_h['val'][self.val_task_h][-1]
                best_training_acc_h = epoch_losses_h['train']['all'][-1]
                best_epoch_h = epoch
                best_model_wts_h = copy.deepcopy(self.model_h.state_dict())

        time_elapsed = time.time() - since
        print('\n\n' + '-' * 120)
        self.logger.info('Training:\nTraining complete in {:.0f}m {:.0f}s'
                         .format(time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Best training Accuracy: {:.3f}'.format(best_training_acc_h))
        self.logger.info('Best validation Accuracy for {}: {:.2f} cm'.format(self.val_task_h, 100*best_acc_h))
        self.logger.info('Saved weights of the model at epoch: {}'.format(best_epoch_h))


        # load best model weights
        self.model_h.load_state_dict(best_model_wts_h)
        path_model_h = './model_h.pkl'
        torch.save(self.model_h.state_dict(), path_model_h)
        # ------------------------------------------------------------------------------------------------------------

        # 2) PRE-TRAIN MAIN NETWORK
        since = time.time()
        best_acc = 1e6
        best_training_acc = 1e6
        best_epoch = 0
        epoch_losses_1 = defaultdict(lambda: defaultdict(list))

        print(">>> creating model")
        model = MonStereoModel if self.mode == 'stereo' else MonoLocoPPModel

        self.model_1 = model(
            input_size=self.input_size[self.mode],
            output_size=self.output_size_1[self.mode],
            linear_size=self.hidden_size,
            p_dropout=self.dropout,
            num_stage=self.n_stage,
        )
        self.model_1.to(self.device)
        losses_tr_1, losses_val_1 = CompositeLoss(self.tasks_1)()
        self.loss_1 = MultiTaskLoss(losses_tr_1, losses_val_1, self.lambdas_1, self.tasks_1)
        self.loss_1.to(self.device)

        print(">>> model params: {:.3f}M".format(sum(p.numel() for p in self.model_1.parameters()) / 1000000.0))
        print(">>> loss params: {}".format(sum(p.numel() for p in self.loss_1.parameters())))
        best_model_wts_1 = copy.deepcopy(self.model_1.state_dict())

        # Optimizer and scheduler
        all_params = chain(
            self.model_1.parameters(),
            self.loss_1.parameters(),
        )
        self.optimizer = torch.optim.Adam(params=all_params, lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.sched_step, gamma=self.sched_gamma)

        for epoch in range(self.num_epochs):
            running_loss_1 = defaultdict(lambda: defaultdict(int))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model_1.train()
                else:
                    self.model_1.eval()

                for inputs, labels, _, _ in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            self.optimizer.zero_grad()

                            outputs_1 = self.model_1(inputs)
                            loss_1, _ = self.loss_1(outputs_1, labels, phase=phase)
                            loss_1.backward()
                            torch.nn.utils.clip_grad_norm_(self.model_1.parameters(), 3)
                            self.optimizer.step()
                            self.scheduler.step()
                        else:
                            outputs_1 = self.model_1(inputs)
                        with torch.no_grad():
                            loss_eval_1, loss_values_eval_1 = self.loss_1(outputs_1, labels, phase='val')
                            epoch_logs(running_loss_1, self.tasks_1, phase, loss_eval_1, loss_values_eval_1, inputs)

            cout_values(epoch, epoch_losses_1, running_loss_1, self.dataset_sizes[phase])

            # deep copy the model
            if epoch_losses_1['val'][self.val_task_1][-1] < best_acc:
                best_acc = epoch_losses_1['val'][self.val_task_1][-1]
                best_training_acc = epoch_losses_1['train']['all'][-1]
                best_epoch = epoch
                best_model_wts_1 = copy.deepcopy(self.model_1.state_dict())

        time_elapsed = time.time() - since
        print('\n\n' + '-' * 120)
        self.logger.info('Training:\nTraining complete in {:.0f}m {:.0f}s'
                         .format(time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Best training Accuracy: {:.3f}'.format(best_training_acc))
        self.logger.info('Best validation Accuracy for {}: {:.3f}'.format(self.val_task_1, best_acc))
        self.logger.info('Saved weights of the model at epoch: {}'.format(best_epoch))

        # save best model weights
        self.model_1.load_state_dict(best_model_wts_1)
        path_model_1 = './model_1.pkl'
        torch.save(self.model_1.state_dict(), path_model_1)

        # ------------------------------------------------------------------------------------------------------------
        # 3) TRAINING PERCEPTUAL LOSS
        since = time.time()
        print('-' * 100)
        print('-' * 100)
        print(f"Reloading model 1 from epoch {best_epoch} with accuracy {best_acc:.1f} m")
        self.model_1 = model(
            input_size=self.input_size[self.mode],
            output_size=self.output_size_1[self.mode],
            linear_size=self.hidden_size,
            p_dropout=self.dropout,
            num_stage=self.n_stage,
        )
        self.model_1.load_state_dict(torch.load(path_model_1, map_location=lambda storage, loc: storage))
        print(f"Reloading model h from epoch {best_epoch_h} with accuracy {100*best_acc_h:.2f} cm")
        print('-' * 100)
        best_acc = 1e6
        self.model_h.load_state_dict(torch.load(path_model_h, map_location=lambda storage, loc: storage))
        self.model_h.eval()
        losses_tr_1, losses_val_1 = CompositeLoss(self.tasks_1)()
        self.loss_1 = MultiTaskLoss(losses_tr_1, losses_val_1, self.lambdas_1, self.tasks_1)
        self.model_1.to(self.device)
        self.model_h.to(self.device)
        self.loss_1.to(self.device)

        # Optimizer and scheduler
        all_params = chain(
            self.model_1.parameters(),
            self.loss_1.parameters(),
        )

        self.optimizer = torch.optim.Adam(params=all_params, lr=self.lr/2)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.sched_step/2, gamma=self.sched_gamma)

        for epoch in range(self.num_epochs, int(2 * self.num_epochs)):
            running_loss_1 = defaultdict(lambda: defaultdict(int))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model_1.train()
                else:
                    self.model_1.eval()

                for inputs, labels, _, _ in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    inputs_y = inputs[:, 1::2]
                    h_max, _ = torch.max(inputs_y, dim=1)
                    h_min, _ = torch.min(inputs_y, dim=1)
                    h = h_max - h_min
                    h = h.reshape(-1, 1)
                    h_min = h_min.reshape(-1, 1)

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            self.optimizer.zero_grad()

                            outputs_1 = self.model_1(inputs)
                            loss_1, _ = self.loss_1(outputs_1, labels, phase=phase)

                            inputs_h_f = torch.cat((outputs_1[:, 0:1], h, h_min), dim=1)
                            inputs_h = torch.cat((labels[:, 3:4], h, h_min), dim=1)
                            with torch.no_grad():
                                outputs_h_f = self.model_h(inputs_h_f)
                                outputs_h = self.model_h(inputs_h)
                            loss_h, _ = self.loss_h(outputs_h_f, outputs_h)
                            loss = loss_1 + loss_h
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model_1.parameters(), 3)
                            self.optimizer.step()
                            self.scheduler.step()

                        else:
                            outputs_1 = self.model_1(inputs)
                        with torch.no_grad():
                            loss_eval_1, loss_values_eval_1 = self.loss_1(outputs_1, labels, phase='val')
                            epoch_logs(running_loss_1, self.tasks_1, phase, loss_eval_1, loss_values_eval_1, inputs)

            cout_values(epoch, epoch_losses_1, running_loss_1, self.dataset_sizes[phase])

            # deep copy the model
            if epoch_losses_1['val'][self.val_task_1][-1] < best_acc:
                best_acc = epoch_losses_1['val'][self.val_task_1][-1]
                best_training_acc = epoch_losses_1['train']['all'][-1]
                best_epoch = epoch
                best_model_wts_1 = copy.deepcopy(self.model_1.state_dict())

        time_elapsed = time.time() - since
        print('\n\n' + '-' * 120)
        self.logger.info('Training:\nTraining complete in {:.0f}m {:.0f}s'
                         .format(time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Best training Accuracy: {:.3f}'.format(best_training_acc))
        self.logger.info('Best validation Accuracy for {}: {:.3f}'.format(self.val_task_1, best_acc))
        self.logger.info('Saved weights of the model at epoch: {}'.format(best_epoch))

        if self.print_loss:
            print_losses(epoch_losses_1, self.dir_figures)

        # load best model weights
        self.model_1.load_state_dict(best_model_wts_1)
        return best_epoch

    def evaluate(self, load=False, model=None, debug=False):

        # To load a model instead of using the trained one
        if load:
            self.model_1.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

        # Average distance on training and test set after unnormalizing
        self.model_1.eval()
        dic_err = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))  # initialized to zero
        dic_err['val']['sigmas'] = [0.] * len(self.tasks_1)
        dataset = KeypointsDataset(self.joints, phase='val')
        with torch.no_grad():

            # Evaluate performances on different clusters and save statistics
            for clst in self.clusters:
                inputs, labels, size_eval = dataset.get_cluster_annotations(clst)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass on each cluster
                outputs = self.model_1(inputs)
                compute_stats(self.tasks_1, self.loss_1, outputs, labels, dic_err['val'], size_eval, clst=clst)
                cout_stats(self.logger, dic_err['val'], size_eval, clst=clst)

            # Evaluate on all the instances
            # Debug plot for input-output distributions

            start = 0
            size_eval = len(dataset)
            for end in range(self.VAL_BS, size_eval + self.VAL_BS, self.VAL_BS):
                end = end if end < size_eval else size_eval
                inputs, labels, _, _ = dataset[start:end]
                start = end
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model_1(inputs)
                compute_stats(self.tasks_1, self.loss_1, outputs, labels, dic_err['val'], size_eval, clst='all')
            cout_stats(self.logger, dic_err['val'], size_eval, clst='all')
            if debug:
                debug_plots(self.model_1(inputs), labels)
                sys.exit()

        # Save the model and the results
        if not (self.no_save or load):
            torch.save(self.model_1.state_dict(), self.path_model)
            print('-' * 120)
            self.logger.info("\nmodel saved: {} \n".format(self.path_model))
        else:
            self.logger.info("\nmodel not saved\n")

        return dic_err, self.model_1

    def _set_logger(self, args):
        if self.no_save:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.path_model = self.path_out
            print(self.path_model)
            self.logger = set_logger(os.path.splitext(self.path_out)[0])  # remove .pkl
            self.logger.info(  # pylint: disable=logging-fstring-interpolation
                f'\nVERSION: {__version__}\n'
                f'\nINPUT_FILE: {args.joints}'
                f'\nInput file version: {self.dataset_version}'
                f'\nTorch version: {torch.__version__}\n'
                f'\nTraining arguments:'
                f'\nmode: {self.mode} \nlearning rate: {args.lr} \nbatch_size: {args.bs}'
                f'\nepochs: {args.epochs} \ndropout: {args.dropout} '
                f'\nscheduler step: {args.sched_step} \nscheduler gamma: {args.sched_gamma} '
                f'\ninput_size: {self.input_size[self.mode]} \noutput_size: {self.output_size[self.mode]} '
                f'\nhidden_size: {args.hidden_size}'
                f' \nn_stages: {args.n_stage} \n r_seed: {args.r_seed} \nlambdas: {self.lambdas}'
            )


def epoch_logs(running_loss, tasks, phase, loss, loss_values, inputs):

    running_loss[phase]['all'] += loss.item() * inputs.size(0)
    for i, task in enumerate(tasks):
        running_loss[phase][task] += loss_values[i].item() * inputs.size(0)


def compute_stats(tasks, loss, outputs, labels, dic_err, size_eval, clst):
    """Compute mean, bi and max of torch tensors"""

    _, loss_values = loss(outputs, labels, phase='val')
    rel_frac = outputs.size(0) / size_eval

    for idx, task in enumerate(tasks):
        dic_err[clst][task] += float(loss_values[idx].item()) * (outputs.size(0) / size_eval)

    # Distance + Uncertainty
    if 'd' in tasks:
        errs = torch.abs(extract_outputs(outputs, tasks)['d'] - extract_labels(labels)['d'])
        assert rel_frac > 0.99, "Variance of errors not supported with partial evaluation"
        bis = extract_outputs(outputs, tasks)['bi'].cpu()
        bi = float(torch.mean(bis).item())
        bi_perc = float(torch.sum(errs <= bis)) / errs.shape[0]
        dic_err[clst]['bi'] += bi * rel_frac
        dic_err[clst]['bi%'] += bi_perc * rel_frac
        dic_err[clst]['std'] = errs.std()

    # Auxiliary task for stereo
    if 'aux' in task:
        acc_aux = get_accuracy(extract_outputs(outputs, tasks)['aux'], extract_labels(labels)['aux'])
        dic_err[clst]['aux'] += acc_aux * rel_frac


def cout_stats(logger, dic_err, size_eval, clst):

    logger.info('-' * 80)
    logger.info(f'Validation set for the cluster {clst} with {size_eval} people')
    for task in dic_err[clst]:
        if task == 'bi%':
            unit = '%'
            dic_err[clst][task] *= 100
        elif task == 'ori':
            unit = 'degrees'
        else:
            unit = 'meters'
        logger.info(f'{task.upper()}: {dic_err[clst][task]:.3f} {unit}')
    logger.info('-' * 100 + '\n')


def cout_values(epoch, epoch_losses, running_loss, data_size):
    string = '\r' + '{:.0f} '
    format_list = [epoch]
    for phase in running_loss:
        string = string + phase[0:1].upper() + ':'
        for el in running_loss['train']:
            loss = running_loss[phase][el] / data_size
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


def print_losses(epoch_losses, dir_figures):
    os.makedirs(dir_figures, exist_ok=True)

    if plt is None:
        raise Exception('please install matplotlib')

    for idx, phase in enumerate(epoch_losses):
        for idx_2, el in enumerate(epoch_losses['train']):
            plt.figure(idx + idx_2)
            plt.title(phase + '_' + el)
            plt.xlabel('epochs')
            plt.plot(epoch_losses[phase][el][10:], label='{} Loss: {}'.format(phase, el))
            plt.savefig(os.path.join(dir_figures, '{}_loss_{}.png'.format(phase, el)))
            plt.close()


def debug_plots(outputs, labels):
    outputs = outputs.cpu().numpy()[:, 0]
    labels = labels.cpu().numpy()[:, 4:5]
    plt.figure(1)
    plt.hist(labels, bins='auto')
    plt.hist(outputs, bins='auto')
    plt.title("Bounding box height analysis")
    plt.ylabel("Number of instances")
    plt.xlabel("Box Height [m]")
    plt.ylabel("Number of instances")
    plt.xlabel("Box Height [m]")
    plt.legend(("Ground-truth", "Network predictions"))
    plt.show()


def get_accuracy(outputs, labels):
    """From Binary cross entropy outputs to accuracy"""

    mask = outputs >= 0.5
    accuracy = 1. - torch.mean(torch.abs(mask.float() - labels)).item()
    return accuracy
