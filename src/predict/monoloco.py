
"""
Monoloco predictor. It receives pifpaf joints and outputs distances
"""

import logging
import time

import torch

from models.architectures import LinearModel
from utils.misc import laplace_sampling
from utils.normalize import unnormalize_bi
from utils.pifpaf import get_network_inputs


class MonoLoco:

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    OUTPUT_SIZE = 2
    INPUT_SIZE = 17 * 2
    LINEAR_SIZE = 256
    IOU_MIN = 0.25
    N_SAMPLES = 100

    def __init__(self, model, device, n_dropout=0):

        self.device = device
        self.n_dropout = n_dropout
        if self.n_dropout > 0:
            self.epistemic = True
        else:
            self.epistemic = False

        # load the model parameters
        self.model = LinearModel(input_size=self.INPUT_SIZE, output_size=self.OUTPUT_SIZE, linear_size=self.LINEAR_SIZE)
        self.model.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
        self.model.eval()  # Default is train
        self.model.to(self.device)

    def forward(self, keypoints, kk):
        """forward pass of monoloco network"""

        if not keypoints:
            return None

        with torch.no_grad():
            kps_torch = torch.tensor(keypoints).to(self.device)
            kk = torch.tensor(kk).to(self.device)
            inputs = get_network_inputs(kps_torch, kk)
            start = time.time()
            if self.n_dropout > 0:
                self.model.dropout.training = True  # Manually reactivate dropout in eval
                total_outputs = torch.empty((0, inputs.size()[0])).to(self.device)

                for _ in range(self.n_dropout):
                    outputs = self.model(inputs)
                    outputs = unnormalize_bi(outputs)
                    samples = laplace_sampling(outputs, self.N_SAMPLES)
                    total_outputs = torch.cat((total_outputs, samples), 0)
                varss = total_outputs.std(0)
                self.model.dropout.training = False
            else:
                varss = [0] * inputs.size()[0]

            # # Don't use dropout for the mean prediction
            start_single = time.time()
            outputs = self.model(inputs)
            outputs = unnormalize_bi(outputs)
        end = time.time()
        print("Total Forward pass time with {} forward passes = {:.2f} ms"
              .format(self.n_dropout, (end-start) * 1000))
        print("Single forward pass time = {:.2f} ms".format((end - start_single) * 1000))
        return outputs, varss
