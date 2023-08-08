class TimeSlot:
    def __init__(self, day, time):
        self.day = day
        self.time = time

    @classmethod
    def convert_to_1d_index(cls, n_day_hour, day, time):
        return day * n_day_hour + time
    #
    # def convert_to_1d_index(self, n_days):
    #     return self.time * n_days + self.day

    @classmethod
    def parse_1d_index(cls, index, n_day_hour):
        time = index % n_day_hour
        day = index // n_day_hour

        return TimeSlot(day, time)
