import pandas as pd

from utils.utils import Utils


class Teacher:
    _next_id = 0

    def __init__(self
                 , email
                 , teacher_type
                 , tag
                 , level
                 , package
                 # , remaining_ts
                 , n_commitment_ts):
        self._id = Teacher._next_id
        Teacher._next_id += 1
        self.email = email
        self._type = teacher_type
        self._tag = tag
        self._level = level
        self.package = package
        self._tag_bit = Utils.encode_param("teacher_tag", tag=tag)
        self._type_bit = Utils.encode_param("teacher_type", type=teacher_type)
        self._num_remaining_timeslot = 0
        self._num_committed_timeslot = n_commitment_ts
        self._available_slots = []
        self.available_ts_by_day = {}

        self.df_available_timeslot = None

    @property
    def id(self):
        return self._id

    @property
    def tag_bit(self):
        return self._tag_bit

    @property
    def type_bit(self):
        return self._type_bit

    @property
    def n_remaining_ts(self):
        return self._num_remaining_timeslot

    @property
    def n_commitment_ts(self):
        return self._num_committed_timeslot

    @property
    def available_ts(self):
        return self._available_slots

    @property
    def n_available_ts(self):
        return len(self._available_slots)
    
    def start_ts_of_day(self, day):
        a = min(self.available_ts_by_day[day])
        # print(self.email,a, self._available_slots)
        return a
    
    def last_ts_of_day(self, day):
        b = max(self.available_ts_by_day[day])
        # print(b)
        return b

    def add_available_ts(self, slot):
        self._available_slots.append(slot)
        day, time = slot[0], slot[1]
        if day not in self.available_ts_by_day:
            self.available_ts_by_day[day] = []
        self.available_ts_by_day[day].append(time)
