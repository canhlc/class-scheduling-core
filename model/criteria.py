class Criteria:
    @staticmethod
    def is_overlapped_teacher(cls, _class, reservation):
        teacher = reservation[1]
        timeslot = reservation[2]