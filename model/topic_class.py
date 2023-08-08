class TopicClass:
    _next_id = 0

    def __init__(self
                 , topic
                 , timeslot
                 , is_fixed):
        self._id = TopicClass._next_id
        TopicClass._next_id += 1
        self.topic = topic
        self.timeslot = timeslot
        self.fixed_class = is_fixed

    @property
    def id(self):
        return self._id
