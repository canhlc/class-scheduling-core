from deap import base

class FitnessCustom(base.Fitness):
    score = 0
    def setValues(self, values):
        super().setValues(values)
        # self.score = sum(x/y for x, y in zip(self.wvalues, self.weights))/len(self.weights)

    @property
    def score(self):
        return sum(self.wvalues)/len(self.weights)
    
    def getValues(self):
        return super().getValues()
    
    def delValues(self):
        return super().delValues()
    
    def dominates(self, other, obj=slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.

        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False

        if self.score > other.score:
            return True
        for self_wvalue, other_wvalue in zip(self.wvalues[obj], other.wvalues[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False

        return not_equal

    values = property(getValues, setValues, delValues,
                    ("Fitness values. Use directly ``individual.fitness.values = values`` "
                    "in order to set the fitness and ``del individual.fitness.values`` "
                    "in order to clear (invalidate) the fitness. The (unweighted) fitness "
                    "can be directly accessed via ``individual.fitness.values``."))

    def __str__(self):
        """Return the values of the Fitness object."""
        values = self.values if self.valid else tuple()
        return ' '.join(format(f, '.3f') for f in values)