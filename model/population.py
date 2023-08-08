from functools import partial


class Population:
    def __init__(self, pop_size, func, *args, **kwargs):
        pfunc = partial(func, *args, **kwargs)
        self.individuals = [pfunc() for _ in range(pop_size)]

