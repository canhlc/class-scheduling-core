def encode_teacher_tag(**kwargs):
    tag = kwargs["tag"].lower()
    enum = {
        'standard': int('0001', 2),
        'vip': int('0011', 2),
        'trial': int('0111', 2)
    }
    if tag not in enum:
        raise Exception("Please check teacher's tag parameter")

    return enum[tag]


def encode_class_type(**kwargs):
    tag = kwargs["tag"].lower()
    enum = {
        'standard': int('0001', 2),
        'vip': int('0010', 2),
        'trial': int('0100', 2)
    }
    if tag not in enum:
        raise Exception("Please check teacher's tag parameter")

    return enum[tag]


def encode_teacher_type(**kwargs):
    type = kwargs["type"]
    enum = {
        'TVN': 0,
        'TEX': 1,
        'OS-TEX': 1
    }
    return enum[type]
