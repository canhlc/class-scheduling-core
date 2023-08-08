class ObjectiveAbs:
    def __init__(self, n_obj, individuals, problem=None) -> None:
        self._individuals = individuals
        self._problem = problem
        self._n_individual = len(individuals)
        self.__n_obj = n_obj
        self._criterias = [0.0] * self._n_individual * n_obj

    @property
    def score(self):
        # return sum(self._criterias) / len(self._criterias)
        return sum(self._criterias) / self.__n_obj
    
    @property
    def checking_result(self):
        return self._criterias