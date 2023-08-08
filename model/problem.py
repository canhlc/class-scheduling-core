import copy
import random
import model.topic_class
# from model.schedule import Schedule
from model.timeslot import TimeSlot
from utils.utils import Utils
import pandas as pd
# import networkx as nx

def calculate_hour_difference(time_slot1, time_slot2):
    weekday_diff = min(abs(time_slot1[0] - time_slot2[0]), 7 - abs(time_slot1[0] - time_slot2[0]))
    hour_diff = abs(time_slot1[1] - time_slot2[1])
    return weekday_diff * 32 + hour_diff

class ClassSchedulingProblem:
    N_DAY = 7
    N_TIMESLOT_PER_DAY = 32

    def __init__(self):
        self.topic_list = None  # parsed topics
        self.class_list = None
        self.teacher_list = None  # parsed teachers
        self.cop_teacher_list = None
        self.fitness_weight = [-1.0, -1.0]
        self.ttype = 'TEX'

        # self.G = nx.Graph()

    def run(self):
        demand_path = 'dataset/demand.csv'
        teacher_path = 'dataset/available.csv'

        self.topic_list, self.class_list = Utils.read_input("demand", ttype=self.ttype, path=demand_path, method='csv')
        # self.heatmap = Utils.read_input("heatmap", heatmap_path)
        self.teacher_list = Utils.read_input("teacher", ttype=self.ttype, path=teacher_path, method='csv')

        # bind teachers to topic
        for topic in self.topic_list:
            for teacher in self.teacher_list:
                if Utils.can_teach(topic, teacher):
                    topic.add_possible_teacher(teacher)
        
        # for teacher in self.teacher_list:
        #     for teacher_slot in teacher.available_ts:
        #         self.G.add_node((teacher, teacher_slot[0], teacher_slot[1]))

        # for topic_class in self.class_list:
        #     for i in range(topic_class.topic.duration):
        #         self.G.add_node((topic_class,  topic_class.timeslot[0], topic_class.timeslot[1]))


        # for topic_class in self.class_list:
        #     for teacher in topic_class.topic.teachers:
        #         for teacher_slot in teacher.available_ts:
        #                 if topic_class.timeslot[0] == teacher_slot[0] and topic_class.timeslot[1] + i in range (teacher_slot[1] -1 , teacher_slot[1] + 1):
        #                     for i in range(topic_class.topic.duration):
        #                         self.G.add_edge((topic_class, topic_class.timeslot[0], topic_class.timeslot[1] + i),
        #                             (teacher, teacher_slot[0], teacher_slot[1] + i))
                        
                        
                    # hour_diff = calculate_hour_difference(teacher_slot, topic_class.timeslot)
                    # self.G.add_edge((topic_class, topic_class.timeslot[0], topic_class.timeslot[1]),
                    #                 (teacher, teacher_slot[0], teacher_slot[1]), weight=hour_diff)

        self.cop_teacher_list = [teacher for teacher in self.teacher_list if teacher.package > 0]


    # Returns number of parsed teachers
    def get_nof_teachers(self):
        return len(self.teacher_list)



def generate_schedule(problem, fitness):
    # problem = kwargs['problem']
    # fitness = kwargs['fitness']
    schedule = Schedule(problem, fitness)

    for tc in problem.class_list:
        topic = tc.topic
        class_ts = tc.timeslot
        class_duration = topic.duration
        topic_teachers = topic.teachers

        if not topic_teachers:
            schedule.genes.append(None)
            raise Exception(F'No teacher can teach the topic {topic.code}')
        else:
            # determine random teacher who can teach the class of topic
            teacher = topic_teachers[int(random.random() * topic.nof_teachers)]

            # determine timeslot
            # if class_ts:
            timeslot = (class_ts[0], class_ts[1])
            # else:
            #     # random a timeslot from available timeslot of chosen teacher
            #     ts_id = int(random.random() * (teacher.n_available_ts - class_duration))
            #     ts_id -= 1 if ts_id % 2 != 0 and class_duration > 1 and ts_id > 0 else 0
            #     timeslot = teacher.available_ts[ts_id]

            schedule.genes.append([tc.id, teacher.id, timeslot])
    return schedule


def generate_topic_schedule(problem, fitness):
    # problem = kwargs['problem']
    # fitness = kwargs['fitness']
    schedule = Schedule(problem, fitness)



def convert_genes_to_dataframe(problem, genes):
    arr = []
    for g in genes:
        class_duration = problem.class_list[g[0]].topic.duration
        for j in range(0, class_duration):
            obj = {
                'class_id': g[0]
                , 'class_day': g[2][0]
                , 'class_time': g[2][1] + j
                , 'teacher_id': g[1]
                , 'is_start_class': True if j == 0 else False
            }
            arr.append(obj)
    df = pd.DataFrame(arr).sort_values(by=['class_day', 'class_time', 'teacher_id'])
    return df


def demand_evaluate(schedule, problem):
    # score = 0
    # assigned_count = schedule_df.groupby(['class_day', 'class_time', 'teacher_id']).agg(distinct_count=("class_id", "nunique"))
    # overlapped = assigned_count[assigned_count['distinct_count'] > 1]
    # df_overlapped = schedule_df.merge(overlapped, on=['class_day', 'class_time', 'teacher_id'])
    # df_overlapped = df_overlapped[df_overlapped['is_start_class']]
    # count_ol = len(df_overlapped)
    # score = count_ol
    violate = 0
    timetable = {}
    for g in schedule.genes:
        tc = problem.class_list[g[0]]
        class_day = g[2][0]
        for j in range(tc.topic.duration):
            class_time = g[2][1] + j
            if class_day in timetable:
                if class_time in timetable[class_day]:
                    if g[1] in timetable[class_day][class_time]:
                        # If the teacher is already assigned to another class in the same time slot, penalize the individual
                        violate += 1
                else:
                    timetable[class_day][class_time] = []
            else:
                timetable[class_day] = {}
                timetable[class_day][class_time] = []

            timetable[class_day][class_time].append(g[1])

    return violate

    #


def evaluate(schedule, problem):
    # Initialize the fitness values
    df = convert_genes_to_dataframe(problem, schedule.genes)
    demand_score = demand_evaluate(schedule, problem)
    teacher_score = 0

    return demand_score, teacher_score


def crossover1(schedule0, schedule1):
    # print("crossover")
    size = min(len(schedule0.genes), len(schedule1.genes))
    cxpoint1 = random.randint(1, size - 2)
    cxpoint2 = random.randint(cxpoint1, size - 1)
    schedule0.genes[cxpoint1:cxpoint2], schedule1.genes[cxpoint1:cxpoint2] = schedule1.genes[
                                                                             cxpoint1:cxpoint2], schedule0.genes[
                                                                                                 cxpoint1:cxpoint2]
    del schedule1.fitness.values, schedule0.fitness.values
    return schedule0, schedule1


def crossover(parent0, parent1):
    new_child = parent0.copy(True)

    size = len(parent0.genes)
    cp = [False] * size
    n_cross_point = 5
    for i in range(n_cross_point, 0, -1):
        check_point = False
        while not check_point:
            p = random.randrange(size)
            if not cp[p]:
                cp[p] = check_point = True

    first = random.randrange(2) == 0
    new_child.genes = []
    for i in range(size):
        if first:
            teacher_id = parent0.genes[i][1]
        else:
            teacher_id = parent1.genes[i][1]
        if cp[i]:
            first = not first
        new_child.genes.append([parent0.genes[i][0], teacher_id, parent0.genes[i][2]])

    return new_child


# Define the genetic operators (continued)
def mutation(schedule, problem, mut_size):
    genes = schedule.genes
    for i in range(mut_size, 0, -1):
        # select random
        index = random.randint(0, len(genes) - 1)
        class_id = genes[index][0]
        tc = problem.class_list[class_id]
        topic = problem.class_list[class_id].topic
        # possible_teachers = topic.teachers
        # if possible_teachers:
        #     new_teacher = random.choice(possible_teachers)
        #     genes[index][1] = new_teacher.id

        if random.choice([True, False]):
            new_teacher = random.choice(topic.cop_teachers).id
        else:
            new_teacher = random.choice(topic.teachers).id
        genes[index][1] = new_teacher

        if not tc.fixed_class:
            variation = random.randrange(-1, 1)
            hour = tc.timeslot[1] + (topic.duration * variation)
            genes[index][2] = (genes[index][2][0], hour)

    return schedule,
