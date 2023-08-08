import random
from math import factorial


import numpy
import numpy as np
from deap import base, creator, tools
from multiprocessing import Pool
import pandas as pd
from algorithm.algorithm import APNSGA3, selNSGA3WithMemory
from model.fitness import FitnessCustom
from model.objectives import ClassObjectives, TeacherObjectives, TeacherPreferenceObjective
from model.problem import ClassSchedulingProblem, crossover
from utils.parse_file import generate_time_dict


def check_teacher_time_pref(teacher_obj, class_day, class_time):
    if not class_day in teacher_obj.available_ts_by_day:
        return False
    if class_time >= max(0, teacher_obj.start_ts_of_day(class_day) - 4) and min(31, class_time <= teacher_obj.last_ts_of_day(class_day) + 4):
        return True
    return False

def random_timetable(problem):
    arr = []
    tcs = problem.class_list
    for tc in tcs:
        # Choose a teacher who can teach this topic randomly
        # teachers = [t for t in tc.topic.teachers if (tc.timeslot[0] in t.available_ts_by_day and tc.timeslot[1] in t.available_ts_by_day[tc.timeslot[0]])]
        teachers = [t for t in tc.topic.teachers if check_teacher_time_pref(t, tc.timeslot[0], tc.timeslot[1])]
        # print(tc.topic.code, tc.timeslot[0] ,tc.timeslot[1], teachers)
        random_idx = int(random.random() * len(teachers))
        timeslot = [tc.timeslot[0], tc.timeslot[1]]
        reservation = [teachers[random_idx].id, timeslot]
        arr.append(reservation)

    
    return arr


def mutate(individual, problem, mut_size=5):
    for i in range(mut_size, 0, -1):
        random_id = random.randint(0, len(individual) - 1)
        topic_class = problem.class_list[random_id]
        # teachers = [t for t in topic_class.topic.teachers if (topic_class.timeslot[0] in t.available_ts_by_day and topic_class.timeslot[1] in t.available_ts_by_day[topic_class.timeslot[0]])]
        teachers = [t for t in topic_class.topic.teachers if check_teacher_time_pref(t, topic_class.timeslot[0], topic_class.timeslot[1])]
        other_teachers = [t.id for t in teachers if t.id != individual[random_id][0]]
        if len(other_teachers) == 0:
            return individual
        new_teacher_id = random.choice(other_teachers)
        individual[random_id][0] = new_teacher_id
    return individual

iter = 0

def crossover(schedule0, schedule1, n_cross_point):
    # print("crossover")
    size = min(len(schedule0), len(schedule1))
    cxpoint1 = random.randint(1, size - 2)
    cxpoint2 = random.randint(cxpoint1, size - 1)
    schedule0[cxpoint1:cxpoint2], schedule1[cxpoint1:cxpoint2] = schedule1[cxpoint1:cxpoint2], schedule0[cxpoint1:cxpoint2]
    return schedule0, schedule1


def next_nve_class(individuals):
    i=0
    while i < len(individuals):
        yield i, individuals[i]
        i+=1

def eval(individuals, problem):
    # print("evel ", iter)
    obj_class_demand = ClassObjectives(1, individuals, problem)
    obj_teacher = TeacherObjectives(1, individuals, problem)
    obj_teacher_pref = TeacherPreferenceObjective(1, individuals, problem)

    obj_class_demand.qualify()
    obj_teacher.qualify()
    obj_teacher_pref.qualify()
    
    return ( obj_class_demand.score , obj_teacher.score, obj_teacher_pref.score), (obj_class_demand.checking_result, obj_teacher_pref.checking_result)



problem = ClassSchedulingProblem()
problem.run()

creator.create("FitnessMin", FitnessCustom, weights=(-1.7, -1.0, -0.3))
# We create a class Individual, which has base type of array, it also uses our
# just created creator.FitnessMin() class.
creator.create("Individual", list, fitness=creator.FitnessMin)

def export_result(result, problem):
    arr = []
    # print(result.check)
    _, reversed = generate_time_dict(8, 24)
    for position, topic_class in enumerate(problem.class_list):
        assignment_info = result[position]
        teacher = problem.teacher_list[assignment_info[0]]
        timeslot = assignment_info[1]
        obj = {
            'topic_code': topic_class.topic.level,
            'level_code': topic_class.topic.code,
            'topic_tag': topic_class.topic.class_type,
            'teacher_email': teacher.email,
            'teacher_package': teacher.package,
            'day': timeslot[0],
            'time': reversed[timeslot[1]], 
            'duration': topic_class.topic.conversion_teaching_hour,
            'overlap': result.check[0][position ],
            'distance': result.check[1][position],
            
        }
        arr.append(obj)
    
    df = pd.DataFrame(arr).sort_values(by=["teacher_email", "day", "time"])
    df.to_csv('result.csv', encoding='utf-8', index=False, sep=';')


def run():
    pool = Pool(processes=8)

    # We create DEAP's toolbox. Which will contain our mutation functions, etc.
    toolbox = base.Toolbox()
    toolbox.register("map", pool.map)
    # We create a function named 'random_timetable', which
    toolbox.register('random_timetable', random_timetable, problem)

    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.random_timetable)

    # As we now have our individual, we can create our population
    # by making a list of toolbox.individual (which we just created in last line).
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", crossover, n_cross_point=5)

    toolbox.register("mutate", mutate, problem=problem)
    toolbox.register("evaluate", eval, problem=problem)

    # Create uniform reference point
    NOBJ = 3
    P = 8
    H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
    MU = int(H + (4 - H % 4))
    N = 100

    ref_points = tools.uniform_reference_points(NOBJ, P)

    toolbox.register("select", selNSGA3WithMemory(ref_points=ref_points, nd='standard'))
    
    # Create population of size 100
    pop = toolbox.population(n=N)

    CXPB = 0.8
    MUTPB = 0.4
    NGEN = 1400

    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    alg = APNSGA3(toolbox, N, NGEN, cxpb=CXPB, mutbp=MUTPB)
    alg.run(pop, stats=stats, verbose=True)
    export_result(alg.result(), problem)
    pool.close()
    # print(pop[0])


if __name__ == "__main__":
    pause_flag = False


    run()
    #
    # # socket_.start_background_task(run)
    # http_server = WSGIServer(('',5001), app, handler_class=SocketHandler)
    # http_server.serve_forever()
    #

    


    
