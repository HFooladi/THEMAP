from themap.utils.distance_utils import get_configure
from third_party.otdd.otdd.pytorch.distance import DatasetDistance

MOLECULE_DISTANCE_METHODS = ["otdd", "euclidean", "cosine"]

class AbstractDatasetDistance:
    def __init__(self, D1=None, D2=None, method="euclidean"):
        self.source = D1
        self.target = D2
        self.method = method

    def get_distance(self):
        raise NotImplementedError

class MoleculeDatasetDistance:
    def __init__(self, D1=None, D2=None, method="euclidean", **kwargs):
        """
        Calculate the distance between two molecule datasets
        Args:
            D1 (DataLoader): Torch dataloader for the first dataset
            D2 (DataLoader): Torch dataloader The second dataset
            method (str): The distance method to use
        """
        super().__init__(D1, D2, method)
        self.source = D1
        self.target = D2
        self.method = method
    
    def get_hopts(self):
        return get_configure(self.method)
    
    def otdd_distance(self):
        hopts = self.get_hopts()
        dist = DatasetDistance(self.source, self.target, hopts)
        d = dist.distance(maxsamples = 1000)
        print(f'OTDD(src,tgt)={d}')
        return d

    def get_distance(self):
        if self.method == "otdd":
            return self.otdd_distance()
        else:
            return self.euclidean_distance()
    
    def __repr__(self):
        return f"MoleculeDatasetDistance(D1={self.source}, D2={self.target}, method={self.method})"




class ProteinDatasetDistance(AbstractDatasetDistance):
    """
    Calculate the distance between two protein datasets
    Args:
        D1 (ProteinDataset): The first protein dataset
        D2 (ProteinDataset): The second protein dataset
        method (str): The distance method to use
    """
    def __init__(self, D1=None, D2=None, method="euclidean"):
        super().__init__(D1, D2, method)

    def euclidean_distance(self):
        # Calculate the Euclidean distance between two protein datasets
        return get_configure(self.source, self.target)