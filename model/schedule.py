import copy

import numpy as np
import pandas as pd
from deap import base


# class Individual:
#     def __init__(self, problem, fitness, genes):
#         self.problem = problem
#         self.fitness = fitness
#         self.score = 0
#         self.genes = []
#
#     # def __deepcopy__(self, memo):
#     def copy(self, setup):
#         if not setup:
#             n_indv = Individual(self.problem, copy.deepcopy(self.fitness))
#             n_indv.genes = copy.deepcopy(self.genes)
#             return n_indv
#         fitness = base.Fitness
#         fitness.weights = self.fitness.weights
#         return Individual(self.problem, fitness())
#
#     def crossover(self, other):
#         pass
#
#     def export_to(self):
#         pass
#
#
# class Schedule(Individual):
#     def __init__(self, problem, fitness):
#         super().__init__(problem, fitness)
#
#     def copy(self, setup):
#         return super().copy(setup)
#
#     def is_dominate(self, other):
#         better = False
#         count_greater = 0
#         count_lesser = 0
#         for f, w in enumerate(self.fitness.wvalues):
#             if w < other.fitness.wvalues[f]:
#                 count_lesser += 1
#                 return False
#             elif w > other.fitness.wvalues[f]:
#                 count_greater += 1
#                 better = True
#         # if count_lesser < count_greater:
#         #     return True
#         return better
#
#     def export_to(self):
#         arr = []
#         for g in self.genes:
#             topic = self.problem.class_list[g[0]].topic
#             teacher = self.problem.teacher_list[g[1]]
#             timeslot = g[2]
#             obj = {
#                 'code': topic.code,
#                 'theme': topic.theme,
#                 'class_type': topic.class_type,
#                 'teacher_type': topic.teacher_type,
#                 'n_teaching_hour': topic.conversion_teaching_hour,
#                 'teacher_email': teacher.email,
#                 'day': timeslot[0],
#                 'hour': timeslot[1]
#             }
#             arr.append(obj)
#
#         df = pd.DataFrame(arr).sort_values(by=["teacher_email", "day", "hour"])
#         df.to_csv("output_.csv", sep=';', encoding='utf-8', index=False)
#     # def crossover(self, other):
#     #     new_indv =
