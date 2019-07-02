
import json
import numpy as np
import torch

from torch.utils.data import Dataset


class NuScenesDataset(Dataset):
    """
    Get mask joints or ground truth joints and transform into tensors
    """

    def __init__(self, joints, phase):
        """
        Load inputs and outputs from the pickles files from gt joints, mask joints or both
        """
        assert(phase in ['train', 'val', 'test'])

        with open(joints, 'r') as f:
            dic_jo = json.load(f)

        # Define input and output for normal training and inference
        self.inputs = np.array(dic_jo[phase]['X'])
        self.outputs = np.array(dic_jo[phase]['Y']).reshape(-1, 1)
        self.names = dic_jo[phase]['names']
        self.kps = np.array(dic_jo[phase]['kps'])

        # Extract annotations divided in clusters
        self.dic_clst = dic_jo[phase]['clst']

    def __len__(self):
        """
        :return: number of samples (m)
        """
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        """
        Reading the tensors when required. E.g. Retrieving one element or one batch at a time
        :param idx: corresponding to m
        """
        inputs = torch.from_numpy(self.inputs[idx, :]).float()
        outputs = torch.from_numpy(np.array(self.outputs[idx])).float()
        names = self.names[idx]
        kps = self.kps[idx, :]

        return inputs, outputs, names, kps

    def get_cluster_annotations(self, clst):
        """Return normalized annotations corresponding to a certain cluster
        """
        inputs = torch.from_numpy(np.array(self.dic_clst[clst]['X'])).float()
        outputs = torch.from_numpy(np.array(self.dic_clst[clst]['Y'])).float()
        count = len(self.dic_clst[clst]['Y'])

        return inputs, outputs, count







