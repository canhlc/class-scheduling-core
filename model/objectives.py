from core.objective import ObjectiveAbs
import numpy as np

from utils.utils import Utils

class ClassObjectives(ObjectiveAbs):
    def __init__(self, n_obj, individuals, problem) -> None:
        super().__init__(n_obj, individuals, problem)
        self._assigned = {}

    def check_teacher_ability(self, nve_topic, teacher_obj):
        result = Utils.can_teach(nve_topic, teacher_obj)
        return result

    def check_no_overlap_teacher(self, duration, teacher_idx, event_day, event_time):
        for j in range(duration):
            cls_ts_hour = event_time + j
            if event_day in self._assigned:
                if cls_ts_hour in self._assigned[event_day]:
                    if teacher_idx in self._assigned[event_day][cls_ts_hour]:
                        # self._criterias[individual_idx] = 0.0
                        return 0.0
                else:
                    self._assigned[event_day][cls_ts_hour] = []
            else:
                self._assigned[event_day] = {}
                self._assigned[event_day][cls_ts_hour] = []
            self._assigned[event_day][cls_ts_hour].append(teacher_idx)
        return 1.0
    

    def qualify(self):
        for i, reservation in enumerate(self._individuals):
            nve_class = self._problem.class_list[i]
            duration = nve_class.topic.duration
            teacher_idx = reservation[0]
            event_day = reservation[1][0]
            event_time = reservation[1][1]

            self._criterias[i] = not self.check_no_overlap_teacher(duration, teacher_idx, event_day, event_time)
            
            # teacher = self._problem.teacher_list[teacher_idx]
            # self._criterias[i + 1] = not self.check_teacher_ability(nve_class.topic, teacher)


class TeacherPreferenceObjective(ObjectiveAbs):
    def __init__(self, n_obj, individuals, problem=None) -> None:
        super().__init__(n_obj, individuals, problem)

    def calc_prefer_time(self, teacher_obj, event_day, event_time):
        pref_ts_by_day = np.array(teacher_obj.available_ts_by_day[event_day]) if event_day in teacher_obj.available_ts_by_day else None
        min_ts = np.min(np.abs(event_time - pref_ts_by_day))
        return 32 if pref_ts_by_day is None else min_ts
    

    
    def qualify(self):
        for i, reservation in enumerate(self._individuals):
            nve_class = self._problem.class_list[i]
            teacher_idx = reservation[0]
            event_day = reservation[1][0]
            event_time = reservation[1][1]
            teacher = self._problem.teacher_list[teacher_idx]
            self._criterias[i] = self.calc_prefer_time(teacher, event_day, event_time)

class TeacherObjectives(ObjectiveAbs):
    def __init__(self, n_obj, individuals, problem) -> None:
        super().__init__(n_obj, individuals, problem)
        # self._assigned = [[0 for _ in range(problem.N_DAY)] for _ in range(problem.get_nof_teachers())]
        self._assigned = np.full((problem.get_nof_teachers(), problem.N_DAY), 0)


    # def generate_individual_timetable(self, teacher_idx, problem):
    #     []
    def count_overload_daily(self, teacher_idx):
        MAX_DAILY = 8
        checking = self._assigned[teacher_idx].copy()
        checking -= MAX_DAILY
        total_overload = np.sum(np.maximum(checking, 0))
        
        return total_overload
    
    def qualify(self):
        for i in range(self._problem.get_nof_teachers()):
            self._criterias[i] = self.count_overload_daily(i)

    def add(self, nve_class, reservation):
        nve_class_idx = nve_class.id
        teacher_idx = reservation[0]
        event_day = reservation[1][0]
        self._assigned[teacher_idx][event_day]+=1