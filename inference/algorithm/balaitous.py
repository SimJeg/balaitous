import torch
from torchvision import transforms

from algorithm.transforms import *


class Balaitous():
    """
    Balaitous is an updated version of the AI-severity algorithm described in  Lassau et al. [1]
    and implemented in the scancovia repository [2]. Given an input CT scan, the model outputs 
    a probability for COVID disease and for severe outcome (intubation or death within one month).
    [1] https://www.nature.com/articles/s41467-020-20657-4
    [2] https://github.com/owkin/scancovia

    Parameters
    ----------
    device: str
        'cuda' to run on GPU, 'cpu' to run on cpu
    """

    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.transforms = transforms.Compose([
            ParseAgeSex(),
            SitkResample(output_spacing=(1.5, 1.5, 5)),
            CreateLungMask(device),
            SitkToNumpy(),
            CleanLungMask(hu_threshold=-100, percentile=5),
            CropPad(output_size=224, constant_values=-1024),
            CropLungs(percentile=10),
            iBOTExtractor(key='full', checkpoint='vitl_ct', n_blocks=24, vmin=-1024, vmax=600, device=device),
            ApplyLungMask(constant_values=-1024),
            iBOTExtractor(key='lung', checkpoint='vitl_i22k', n_blocks=16, vmin=-1024, vmax=600, device=device),
            ApplySTOICLogisticRegressions()
        ])

    def __call__(self, input_image):
        """
        Parameters
        ----------
        input_image: sitk image
    

        Returns
        -------
        dict
            sample. Main keys : sample['prediction']['severe'] or sample['prediction']['covid']

        """
        sample = self.transforms({'image': input_image})
        return sample
