class Utils:
    @classmethod
    def can_teach(cls, topic, teacher):
        if (topic.ctype_bit & teacher.tag_bit) != topic.ctype_bit:
            return False
        if (topic.ttype_bit == teacher.type_bit) != topic.ttype_bit:
            return False
        return True

    @classmethod
    def read_input(cls, name, **kwargs):
        from utils import parse_demand_csv_file, parse_teach_csv_file, parse_package_csv_file, \
            parse_heat_map_csv_file

        INPUT_CSV = {
            'demand': parse_demand_csv_file,
            'teacher': parse_teach_csv_file,
            'package': parse_package_csv_file,
            'heatmap': parse_heat_map_csv_file
        }
        if name not in INPUT_CSV:
            raise Exception("Name {} not found.".format(name))

        return INPUT_CSV[name](**kwargs)

    @classmethod
    def encode_param(cls, name, **kwargs):
        from utils import encode_class_type, encode_teacher_tag, encode_teacher_type
        PARAM_NAME = {
            'teacher_tag': encode_teacher_tag,
            'teacher_type': encode_teacher_type,
            'class_type': encode_class_type,
        }

        if name not in PARAM_NAME:
            raise Exception("Name not found.")
        return PARAM_NAME[name](**kwargs)
