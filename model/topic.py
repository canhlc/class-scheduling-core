from utils.utils import Utils


class Topic:
    _next_id = 0

    def __init__(self
                 , code
                 , level
                 , theme
                 , class_type
                 , teacher_type
                 , duration
                 , conversion_duration
                 ):
        self._id = Topic._next_id
        Topic._next_id += 1

        self._code = code
        self._level = level
        self._theme = theme
        # return the type of class: standard | VIP | trial
        self._class_type = class_type
        # return the type of teacher: TEX | TVN
        self._teacher_type = teacher_type
        self._class_type_bit = Utils.encode_param("class_type", tag=class_type)
        self._teacher_type_bit = Utils.encode_param("teacher_type", type=teacher_type)
        self.duration = duration
        self.conversion_teaching_hour = conversion_duration

        # Relation to Teacher model
        self._teachers = None
        self.cop_teachers = []

    def add_possible_teacher(self, teacher):
        if not self._teachers:
            self._teachers = []
        self._teachers.append(teacher)
        if teacher.package > 0:
            self.cop_teachers.append(teacher)

    @property
    def code(self):
        return self._code
    
    @property
    def level(self):
        return self._level

    @property
    def theme(self):
        return self._theme

    @property
    def ctype_bit(self):
        return self._class_type_bit

    @property
    def ttype_bit(self):
        return self._teacher_type_bit

    @property
    def teachers(self):
        return self._teachers

    @property
    def nof_teachers(self):
        return len(self._teachers)

    @property
    def class_type(self):
        return self._class_type

    @property
    def teacher_type(self):
        return self._teacher_type


def set_class_duration(class_type):
    if class_type == 'trial':
        return 1
    return 2
