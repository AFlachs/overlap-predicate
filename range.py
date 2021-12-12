import random

random.seed(10)


class Range:

    def __init__(self, max_val: int):
        n1 = random.randint(0, max_val)
        n2 = random.randint(0, max_val)
        self.start = min(n1, n2)
        self.end = max(n1, n2)

    def len(self):
        return self.end - self.start

    def __str__(self):
        return "[" + str(self.start) + ", " + str(self.end) + "]"

    def __gt__(self, other):
        return self.end > other.end

    def __ge__(self, other):
        return self.end >= other.end

    def __lt__(self, other):
        return self.start < other.start

    def __le__(self, other):
        return self.start <= other.start