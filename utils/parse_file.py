import os
import pandas as pd
from model.constant import Constant
from model.timeslot import TimeSlot


def generate_time_dict(start_hour, end_hour):
    time_dict = {}
    revert_dict = {}
    i = 0
    for hour in range(start_hour, end_hour):
        for minute in range(0, 60, 30):
            time = f'{hour:02}:{minute:02}'
            time_dict[time] = i
            revert_dict[i] = time
            i += 1
    return time_dict, revert_dict


def parse_demand_csv_file(**kwargs):
    from model.topic import Topic
    from model.topic_class import TopicClass

    path = kwargs.get("path")
    method = kwargs.get("src_input_type", "csv")
    ttype = kwargs.get("ttype")
    df = pd.DataFrame()

    if method == 'csv':
        df = pd.read_csv(path, delimiter=";")

    df = df[df['teacher_type'] == ttype]
    df_course = df.drop_duplicates(subset=['code', 'level', 'teacher_type', 'class_type']).reset_index()
    # df_course = df_course[['code', 'theme', 'level', 'teacher_type', 'class_type', 'duration']]
    df_course = df_course[['code', 'level', 'teacher_type', 'class_type', 'duration']]
    df_course["course_id"] = df_course.index

    # df = pd.merge(df, df_course, how="inner", on=['code', 'theme', 'level', 'teacher_type', 'class_type', 'duration'])
    df = pd.merge(df, df_course, how="inner", on=['code', 'level', 'teacher_type', 'class_type', 'duration'])

    # df = df.loc[:250, :]
    topic_list = []

    # convert each row in the Course DataFrame to on instance of Course
    for index, row in df_course.iterrows():
        topic = Topic(
            code=row["code"]
            , level=row["level"]
            # , theme=row["theme"]
            , theme = ''
            , class_type=row["class_type"]
            , teacher_type=row["teacher_type"]
            , duration=int(row["duration"] * 2)
            , conversion_duration=1 if row["duration"] == 1 else 0.75
        )
        topic_list.append(topic)

    time_dict, _ = generate_time_dict(8, 24)

    class_list = []
    for index, row in df.iterrows():
        topic = topic_list[row["course_id"]]
        n_instances = int(row["nof_class"])
        timeslot = (int(row['day'] - 1), int(time_dict[row['time_start']]))

        course_class = [TopicClass(topic=topic
                                   , timeslot=timeslot
                                   , is_fixed=bool(row['is_fixed'])) for _ in range(n_instances)]
        class_list.extend(course_class)

    return topic_list, class_list


def parse_package_csv_file(path, **kwargs):
    df = pd.read_csv(path)
    package_map = {}
    for i in range(len(df)):
        code = df.loc[i, "package_code"]
        n_commitment_hours = df.loc[i, "num_commitment_hours"]
        package_map[code] = {
            "num_commitment_hours": n_commitment_hours
        }
    return package_map


def parse_teach_csv_file(**kwargs):
    from model.teacher import Teacher
    method = kwargs.get("src_input_type", "csv")
    ttype = kwargs.get("ttype")
    path = kwargs.get("path")
    df = pd.DataFrame()
    if method == 'csv':
        df = pd.read_csv(path, delimiter=";")

    time_dict, _ = generate_time_dict(8, 24)
    df = df[df['type'] == ttype]
    # df["time_start"] = df['time_start'].str.split(":", expand=True, n=1).astype({0: 'int'})[0]

    df_info = df.drop_duplicates(subset=["email", "type", "package", "tag"]).sort_values(
        by=["package"], ascending=True, ignore_index=True)
    df_info["teacher_id"] = df_info.index
    df = pd.merge(df, df_info[["email", "teacher_id"]], on="email", how="inner")

    n_teachers = len(df_info)

    teachers = []
    for i in range(n_teachers):
        teachers.append(Teacher(
            email=df_info.loc[i, "email"],
            teacher_type=df_info.loc[i, "type"],
            tag=df_info.loc[i, "tag"],
            level=7,
            package=df_info.loc[i, "package"],
            # n_remaining_committed_slots=df_info.loc[i, "available_hour"] * 2,
            n_commitment_ts=df_info.loc[i, "commitment_hour"]
        ))
        teachers[i].df_available_timeslot = df[df['email']==df_info.loc[i, "email"]]

    n_available = len(df)
    for i in range(n_available):
        _d = int(df.loc[i, "day"]) - 1
        _id = df.loc[i, "teacher_id"]
        # _type = df.loc[i, "type"]
        start_ti = time_dict[df.loc[i, "time_start"]]
        end_ti = start_ti + 2
        for t in range(start_ti, end_ti):
            teachers[_id].add_available_ts((_d, t))


    return teachers


def parse_heat_map_csv_file(path, **kwargs):
    import numpy as np

    df = pd.read_csv(path, sep=";")
    ts_range = Constant.DAYS_NUM * len(Constant.TIME_SLOTS)
    ts_heatmap_lv = np.zeros(ts_range, dtype=int)
    len_df = len(df)

    if len_df == 0:
        raise Exception("Heatmap data is empty")

    for i in range(len_df):
        day = df.loc[i, "day"]
        time = df.loc[i, "time_slot"] - 1
        level = df.loc[i, "level"]

        ts_id = TimeSlot.convert_to_1d_index(len(Constant.TIME_SLOTS), day, time)
        next_ts_id = TimeSlot.convert_to_1d_index(len(Constant.TIME_SLOTS), day + 7, time)
        ts_heatmap_lv[ts_id] = get_num_slots_by_heatmap_level(int(level))
        ts_heatmap_lv[next_ts_id] = get_num_slots_by_heatmap_level(int(level))

    return ts_heatmap_lv


def get_num_slots_by_heatmap_level(heatmap_level):
    if heatmap_level == 0:
        return 0
    if heatmap_level == 1:
        return 10
    if heatmap_level == 2:
        return 20
    if heatmap_level == 3:
        return 40
