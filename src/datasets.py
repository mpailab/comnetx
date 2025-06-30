import torch

class Dataset:
    """Dataset treatment"""

    # TODO (to konovalov) add batch supportion
    # TODO (to uporova) add loader for npy and npz formats
    # TODO (to drobyshev) add loader for magi format

    def __init__(self, dataset_name : str):
        self.name = dataset_name
        self.adj = None
        self.features = None
        self.label = None

    def load(tensor_type : str = "coo") -> torch.Tensor:
        """
        Load dataset

        Parameters
        ----------
        tensor_type : str
            Type of output tensor: coo, csr, csc, dense
            Default: coo
        """
        pass