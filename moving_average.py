class MovingAverage:
    """This class might be a bad idea.
    Might delete.
    """

    def __init__(self, maxlen):
        self.alpha = 2 / (maxlen + 1)
        self.average = 0

    def __add__(self, value):
        self.average = self.alpha * value + (1 - self.alpha) * self.average
        return self
