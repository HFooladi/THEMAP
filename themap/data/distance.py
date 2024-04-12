class DatasetDistance:
    def __init__(self, D1=None, D2=None, method="euclidean"):
        self.source = D1
        self.target = D2
        self.method = method
        self.prob_network = None

    def get_distance(self):
        if self.method == "euclidean":
            return self.euclidean_distance()
        elif self.method == "cosine":
            return self.cosine_distance()
        else:
            raise ValueError("Invalid distance method")
