from abc import ABC
from collections import defaultdict, namedtuple
import datetime
import random
from deap import tools
import numpy

class EMO(ABC):
    def __init__(self, toolbox, pop_size=100, number_generation=5000, cross_probability=0.8,
                 mutation_probability=0.3) -> None:
        super().__init__()
        self.toolbox = toolbox
        self.pop_size = pop_size
        self.n_gen = number_generation
        self.cxpb = cross_probability
        self.mutpb = mutation_probability

    def varAnd(self, pop, toolbox, cxpb, mutbp):
        offspring = []
        k = self.pop_size
        for _ in range(0, k, 2):
            success_choice = False
            while not success_choice:
                ran_pos1 = random.randrange(k)
                ran_pos2 = random.randrange(k)
                success_choice = ran_pos1 != ran_pos2

            child1 = toolbox.clone(pop[ran_pos1])
            child2 = toolbox.clone(pop[ran_pos2])

            if random.random() < cxpb:
                child1, child2 = toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
            offspring.extend([child1, child2])

        for i in range(len(offspring)):
            if random.random() < mutbp:
                offspring[i] = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        return offspring

    def run(self):
        pass

    def result(self):
        pass


def server_side_event(app, sse, data):
    """ Function to publish server side event """
    with app.app_context():
        sse.publish(data, type='dataUpdate')
        print("Event Scheduled at ", datetime.datetime.now())


class APNSGA3(EMO):
    def __init__(self, toolbox, pop_size, number_generation, cxpb, mutbp) -> None:
        self.worst = None
        self.best = None
        self.T = 15
        self.pop_size_max = int(1.5 * pop_size)
        # exacerbation value (ex value)
        self.__ex_max = 0.5
        super().__init__(toolbox, pop_size, number_generation, cxpb, mutbp)

    def result(self):
        return self.best

    def __calc_ex(self, individual):
        '''
        The "exacerbation value" factor is used to determine which individual should be eliminated.
        If ex(x_i) is large <=> f(x_i) is worse and close to f(x_worst) 
            or the solution x_i has not changed for a long time.
        '''
        numerator, denominator = 0.0, 0.0
        for f, obj in enumerate(individual.fitness.values):
            numerator += obj - self.best.fitness.values[f]
            denominator += self.worst.fitness.values[f] - \
                           self.best.fitness.values[f]

        # addition of the constant "1.0" is to avoid the divisor of 0
        return (numerator + 1) / (denominator + 1)

    def __population_dec(self, population):
        N = len(population)
        N_min = self.pop_size
        if N <= N_min:
            return
        i = 0
        while i < N:
            ex = self.__calc_ex(population[i])
            if ex > self.__ex_max and i > int(0.3 * N_min):
                # delete x_i from population
                del population[i]
                N -= 1
                if N <= N_min:
                    break
            i += 1

    def __population_inc(self, m, population):
        '''
        Describe: increase the size of population and introduce new individuals to enhance diversity and escape the local optimal.
        '''
        if self.worst.fitness.dominates(m.fitness):
            population.append(m)
            self.worst = m
        else:
            population.insert(-1, m)

    def dual_ctrl_strategy(self, population, best_not_enhance):
        pop_size = len(population)
        pop_size_tmp = pop_size

        for i in range(pop_size_tmp):
            x_i = population[i]

            # generate new chromosome by the genetic operator
            m_i = self.toolbox.mutate(self.toolbox.clone(x_i))
            m_i.fitness.values = self.toolbox.evaluate(m_i)

            if m_i.fitness.dominates(x_i.fitness):
                population[i] = m_i
                if m_i.fitness.dominates(self.best.fitness):
                    self.best = m_i

            else:
                if best_not_enhance >= self.T and len(population) < self.pop_size_max:
                    self.__population_inc(m_i, population)

            self.worst = population[-1]
        self.__population_dec(population)

    def my_dual_ctrl_strategy(self, population, best_not_enhance):
        pass

    def run(self,
            population,
            max_repeat=100,
            min_fitness=0.999,
            halloffame=None,
            stats=None,
            verbose=__debug__
            ):

        pops = [population, None]
        best_not_enhance = 0

        cur, next = 0, 1
        cur_iter = 0
        # Compile statistics about the population
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        # Evalutate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0]
            ind.check = fit[1]

        last_best_fitness_score = population[0].fitness.score
        # sched = BackgroundScheduler(daemon=True)
        # sched.add_job(server_side_event(app, sse, {'userId': '123'}),'interval',seconds=random.randrange(5,20))
        # sched.start()
        while cur_iter < self.n_gen:
            if cur_iter > 0:
                notification = F"Fitness: {self.best.fitness}\t\t Score: {self.best.fitness.score}\t\t Generation: {cur_iter} \n"
                print(notification)
                # if self.best.fitness.score > 0.99999999:
                    # break

                different = abs(self.best.fitness.score - last_best_fitness_score)
                if last_best_fitness_score >= self.best.fitness.score:
                    best_not_enhance += 1
                else:

                    last_best_fitness_score = self.best.fitness.score
                    best_not_enhance = 0

                # Compile statistics about the new population

                # print(logbook.stream)
                # if best_not_enhance > 15:
                #     print("Not enhance")
                #     print(F"Fitness: {last_best_fitness_score} , {self.best.fitness.values} \t\t", "Generation:", cur_iter, end="...\r")
                # else:
                # print(self.best.check)
                
                # print(self.best.check)
                # yield {
                #         "event": "new_message",
                #         "id": "message_id",
                #         "retry": MESSAGE_STREAM_RETRY_TIMEOUT,
                #         "data": f"Fitness value {fitnesses}",
                #     }
                # logger.debug(notification)
                # await asyncio.sleep(MESSAGE_STREAM_DELAY)

            # Vary the population
            offspring = self.varAnd(
                pops[cur], self.toolbox, self.cxpb, self.mutpb)

            # Evalutate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit[0]
                ind.check = fit[1]

            offspring.sort(key=lambda ind: ind.fitness.score, reverse=True)

            # if halloffame:
            #     halloffame.update(offspring)

            pops[cur].extend(offspring)

            pops[next] = self.toolbox.select(pops[cur], self.pop_size)
            if pops[next][0].fitness.dominates(pops[cur][0].fitness):
                self.best = pops[next][0]
                record = stats.compile(pops[next]) if stats is not None else {}
            else:
                record = stats.compile(pops[cur]) if stats is not None else {}
                self.best = pops[cur][0]

            logbook.record(gen=cur_iter, evals=len(invalid_ind), **record)

            # self.dual_ctrl_strategy(pops[next], best_not_enhance)

            cur, next = next, cur
            cur_iter += 1

        # record = stats.compile(population) if stats is not None else {}
        # logbook.record(gen=0, evals=len(invalid_ind), **record)

        # if halloffame:
        #     halloffame.update(population)

        # if verbose:
        #     print(logbook.stream)

        # # Begin the generational process:
        # for gen in range(1, self.n_gen):


class selNSGA3WithMemory(object):
    """Class version of NSGA-III selection including memory for best, worst and
    extreme points. Registering this operator in a toolbox is a bit different
    than classical operators, it requires to instantiate the class instead
    of just registering the function::

        >>> from deap import base
        >>> ref_points = uniform_reference_points(nobj=3, p=12)
        >>> toolbox = base.Toolbox()
        >>> toolbox.register("select", selNSGA3WithMemory(ref_points))

    """

    def __init__(self, ref_points, nd="log"):
        self.ref_points = ref_points
        self.nd = nd
        self.best_point = numpy.full((1, ref_points.shape[1]), numpy.inf)
        self.worst_point = numpy.full((1, ref_points.shape[1]), -numpy.inf)
        self.extreme_points = None

    def __call__(self, individuals, k):
        chosen, memory = selNSGA3(individuals, k, self.ref_points, self.nd,
                                  self.best_point, self.worst_point,
                                  self.extreme_points, True)
        self.best_point = memory.best_point.reshape((1, -1))
        self.worst_point = memory.worst_point.reshape((1, -1))
        self.extreme_points = memory.extreme_points
        return chosen


NSGA3Memory = namedtuple("NSGA3Memory", ["best_point", "worst_point", "extreme_points"])

def sortNondominated(individuals, k, first_front_only=False):
    """Sort the first *k* *individuals* into different nondomination levels
    using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
    see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
    where :math:`M` is the number of objectives and :math:`N` the number of
    individuals.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param first_front_only: If :obj:`True` sort only the first front and
                             exit.
    :returns: A list of Pareto fronts (lists), the first list includes
              nondominated individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """
    if k == 0:
        return []

    map_fit_ind = defaultdict(list)
    for ind in individuals:
        map_fit_ind[ind.fitness].append(ind)
    fits = list(map_fit_ind.keys())

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i+1:]:
            if fit_i.dominates(fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif fit_j.dominates(fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all individuals are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts

def selNSGA3(individuals, k, ref_points, nd="standard", best_point=None,
             worst_point=None, extreme_points=None, return_memory=False):
    if nd == "standard":
        pareto_fronts = tools.sortNondominated(individuals, k)
    elif nd == "log":
        pareto_fronts = tools.sortLogNondominated(individuals, k)
    else:
        raise Exception("selNSGA3: The choice of non-dominated sorting "
                        "method '{0}' is invalid.".format(nd))

    # Extract fitnesses as a numpy array in the nd-sort order
    # Use wvalues * -1 to tackle always as a minimization problem
    fitnesses = numpy.array([ind.fitness.wvalues for f in pareto_fronts for ind in f])
    fitnesses *= -1

    # Get best and worst point of population, contrary to pymoo
    # we don't use memory
    if best_point is not None and worst_point is not None:
        best_point = numpy.min(numpy.concatenate((fitnesses, best_point), axis=0), axis=0)
        worst_point = numpy.max(numpy.concatenate((fitnesses, worst_point), axis=0), axis=0)
    else:
        best_point = numpy.min(fitnesses, axis=0)
        worst_point = numpy.max(fitnesses, axis=0)

    extreme_points = tools.emo.find_extreme_points(fitnesses, best_point, extreme_points)
    front_worst = numpy.max(fitnesses[:sum(len(f) for f in pareto_fronts), :], axis=0)
    intercepts = tools.emo.find_intercepts(extreme_points, best_point, worst_point, front_worst)
    niches, dist = tools.emo.associate_to_niche(fitnesses, ref_points, best_point, intercepts)

    # Get counts per niche for individuals in all front but the last
    niche_counts = numpy.zeros(len(ref_points), dtype=numpy.int64)
    index, counts = numpy.unique(niches[:-len(pareto_fronts[-1])], return_counts=True)
    niche_counts[index] = counts

    # Choose individuals from all fronts but the last
    chosen = list(tools.emo.chain(*pareto_fronts[:-1]))

    # Use niching to select the remaining individuals
    sel_count = len(chosen)
    n = k - sel_count
    selected = tools.emo.niching(pareto_fronts[-1], n, niches[sel_count:], dist[sel_count:], niche_counts)
    chosen.extend(selected)

    if return_memory:
        return chosen, NSGA3Memory(best_point, worst_point, extreme_points)
    return chosen
