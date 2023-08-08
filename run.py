import array
import random

from deap import algorithms, base, creator, tools
import multiprocessing

from deap.tools import selNSGA3, uniform_reference_points

import algorithm.nsga3 as nsga3
from utils.utils import Utils


class ClassSchedulingProblem:
    def __init__(self,
                 pop_size=100):
        self.topic_list = None
        self.class_list = None
        self.teacher_list = None
        self.heatmap = None

    def run(self):
        demand_path = 'dataset/demand.csv'
        teacher_path = 'dataset/available.csv'


        self.topic_list, self.class_list = Utils.read_input("demand", path=demand_path, ttype='TEX')
        self.teacher_list = Utils.read_input("teacher", path=teacher_path, ttype='TEX')

        for topic in self.topic_list:
            for teacher in self.teacher_list:
                if Utils.can_teach(topic, teacher):
                    topic.add_possible_teacher(teacher)


def generate_individual(problem):
    individual = [

    ]
    for tc in problem.class_list:
        topic = tc.topic
        class_duration = topic.duration
        possible_teachers = topic.possible_teachers

        if not possible_teachers:
            individual.append(None)
        else:
            tid = int(random.random() * len(possible_teachers))
            teacher = problem.teacher_list[tid]
            timeslot = teacher.available_slots[random.randrange(len(teacher.available_slots) - class_duration)]
            individual.append([tc, teacher, timeslot])
    return individual


def evaluate_individual(individual, problem):
    # Initialize the fitness values
    # Initialize
    class_time_slots = {}
    remaining_slots = {}
    teacher_conflict = 0
    timetable_conflict = 0

    for teacher in problem.teacher_list:
        remaining_slots[teacher.id] = teacher.n_remaining_ts

    # evaluate each class in the individual
    for assignment in individual:
        topic_class = assignment[0]
        teacher = assignment[1]
        first_timeslot = assignment[2]
        class_duration = topic_class.topic.duration
        remaining_slots[teacher.id] -= class_duration

        # Check if the teacher is already assigned to another class in the same time slot
        for i in range(class_duration):
            timeslot = first_timeslot + i
            if timeslot in class_time_slots:
                if teacher.id in class_time_slots[timeslot]:
                    # If the teacher is already assigned to another class in the same time slot, penalize the individual
                    teacher_conflict += 1
            else:
                class_time_slots[timeslot] = []

            class_time_slots[timeslot].append(teacher.id)

    for ts, assignment in class_time_slots.items():
        count_assignment = len(assignment)
        if problem.heatmap[ts] - count_assignment < 0:
            timetable_conflict += 1

    remaining_score = sum(remaining_slots.values())
    fitness = (teacher_conflict, timetable_conflict)

    return fitness


def crossover(individual1, individual2):
    size = min(len(individual1), len(individual2))
    cxpoint1 = random.randint(1, size - 2)
    cxpoint2 = random.randint(cxpoint1, size - 1)
    individual1[cxpoint1:cxpoint2], individual2[cxpoint1:cxpoint2] = individual2[cxpoint1:cxpoint2], individual1[
                                                                                                     cxpoint1:cxpoint2]

    return individual1, individual2


# Define the genetic operators (continued)
def mutation(individual, problem):
    index = random.randint(0, len(individual) - 1)
    topic_class = individual[index][0]
    teacher = individual[index][1]
    time_slot = individual[index][2]
    topic = topic_class.topic
    class_duration = topic.duration
    if random.random() < 0.5:  # change the teacher
        possible_teachers = topic.possible_teachers
        if possible_teachers:
            new_teacher = random.choice(possible_teachers)
            individual[index][1] = new_teacher
            individual[index][2] = new_teacher.available_slots[
                random.randrange(len(new_teacher.available_slots) - class_duration)]
    else:  # change the time slot
        new_time_slot = random.choice([t for t in teacher.available_slots if t != time_slot])
        individual[index][2] = new_time_slot
    return individual,


# a three-objectives
creator.create("FitnessMin3", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual3", list,
               fitness=creator.FitnessMin3,
               objectives=[-1] * 2,
               converted_objectives=[-1] * 2)


def prepare_toolbox(
        problem_instance,
        n_var,
        pool
):
    toolbox = base.Toolbox()
    toolbox.register("map", pool.map)

    toolbox.register("evaluate", evaluate_individual, problem=problem_instance)
    toolbox.register("individual", lambda x: creator.Individual3(generate_individual(x)), x=problem_instance)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutation, problem=problem_instance)
    toolbox.register("select", selNSGA3, ref_points=uniform_reference_points(nobj=2))

    toolbox.pop_size = 100  # population size
    toolbox.max_gen = 5000  # max number of iteration
    toolbox.mut_prob = 0.03
    toolbox.cross_prob = 0.8

    return toolbox


def nsga_iii(toolbox, stats=None, verbose=True):
    # initialize the population
    population = toolbox.population(n=toolbox.pop_size)

    return algorithms.eaMuPlusLambda(population
                                     , toolbox
                                     , mu=toolbox.pop_size
                                     , lambda_=toolbox.pop_size
                                     , cxpb=toolbox.cross_prob
                                     , mutpb=toolbox.mut_prob
                                     , ngen=toolbox.max_gen
                                     , stats=stats
                                     , verbose=verbose)


def NSGA3(toolbox, stats=None, verbose=True):
    # initialize the population
    population = toolbox.population(n=toolbox.pop_size)
    return nsga3.run(population
                     , toolbox
                     , mu=toolbox.pop_size
                     , lambda_=toolbox.pop_size
                     , cxpb=toolbox.cross_prob
                     , mutpb=toolbox.mut_prob
                     , ngen=toolbox.max_gen
                     , stats=stats
                     , verbose=verbose)


def main():
    class_scheduling_problem = ClassSchedulingProblem()
    class_scheduling_problem.run()

    # Define the statistics to compute
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", min)
    pool = multiprocessing.Pool(processes=8)
    toolbox = prepare_toolbox(problem_instance=class_scheduling_problem, n_var=2, pool=pool)
    res, logbook = nsga_iii(toolbox, stats=stats)
    # pool.close()
    # for gen in range(5000):

    pareto_front = tools.ParetoFront()
    pareto_front.update(res)
    print(pareto_front[0])


if __name__ == "__main__":
    main()
