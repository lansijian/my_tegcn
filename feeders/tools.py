import random

import numpy as np


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)

def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    if mean == 0:
        return data_numpy
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    # Subtract mean from valid frames
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean

    # Perform IQR outlier deletion
    for c in range(C):
        for v in range(V):
            for m in range(M):
                # Flatten the data for each keypoint across all frames
                keypoints_data = data_numpy[c, :end, v, m]
                
                # Calculate Q1, Q3 and IQR
                Q1 = np.percentile(keypoints_data, 25, interpolation='midpoint')
                Q3 = np.percentile(keypoints_data, 75, interpolation='midpoint')
                IQR = Q3 - Q1
                
                # Define the lower and upper bounds for outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify and replace outliers with median value
                median_value = np.median(keypoints_data[keypoints_data >= lower_bound and keypoints_data <= upper_bound])
                outliers = (keypoints_data < lower_bound) | (keypoints_data > upper_bound)
                data_numpy[c, :end, v, m] = np.where(outliers, median_value, data_numpy[c, :end, v, m])

    return data_numpy
# 在mean_subtractor函数中引入IQR（四分位数范围）来识别和处理异常值，可以通过以下步骤实现：

# 计算每一帧中每个关键点的IQR。
# 确定异常值的阈值，通常是第一四分位数（Q1）减去1.5倍的IQR和第三四分位数（Q3）加上1.5倍的IQR。
# 将超出这个范围的值视为异常值，并进行处理，例如用中位数或均值替换。

# def mean_subtractor(data_numpy, mean):
#     # input: C,T,V,M
#     # naive version
#     if mean == 0:
#         return
#     C, T, V, M = data_numpy.shape
#     valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
#     begin = valid_frame.argmax()
#     end = len(valid_frame) - valid_frame[::-1].argmax()
#     data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
#     return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]

def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],  # 角度候选值
                scale_candidate=[0.9, 1.0, 1.1],  # 缩放候选值
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],  # 变换候选值
                move_time_candidate=[1]):  # 移动时间候选值
    # 输入: C, T, V, M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)  # 随机选择移动时间
    num_nodes = int(T * move_time) + 1  # 确保至少有一个节点

    # 为每个节点生成随机的变换参数
    A = np.random.choice(angle_candidate, num_nodes)  # 随机选择角度
    S = np.random.choice(scale_candidate, num_nodes)  # 随机选择缩放
    T_x = np.random.choice(transform_candidate, num_nodes)  # 随机选择x轴平移
    T_y = np.random.choice(transform_candidate, num_nodes)  # 随机选择y轴平移

    # 在帧之间插值变换参数
    a = np.interp(np.arange(T), np.linspace(0, T, num_nodes), A) * np.pi / 180  # 角度插值
    s = np.interp(np.arange(T), np.linspace(0, T, num_nodes), S)  # 缩放插值
    t_x = np.interp(np.arange(T), np.linspace(0, T, num_nodes), T_x)  # x轴平移插值
    t_y = np.interp(np.arange(T), np.linspace(0, T, num_nodes), T_y)  # y轴平移插值

    # 创建变换矩阵
    theta = np.stack([np.cos(a) * s, -np.sin(a) * s, np.sin(a) * s, np.cos(a) * s], axis=-1).reshape(T, 2, 2)

    # 执行变换
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]  # 提取x和y坐标
        new_xy = np.dot(theta[i_frame], xy.reshape(2, -1))  # 应用变换
        new_xy[0] += t_x[i_frame]  # 应用x轴平移
        new_xy[1] += t_y[i_frame]  # 应用y轴平移
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)  # 更新数据


    return data_numpy
# 改进说明：
# 向量化操作：使用np.interp进行参数插值，减少循环的使用，提高性能。
# 参数化：保留了自定义变换参数的功能，允许用户根据需要调整。
# 边界检查：在实际应用中，可以根据数据的具体范围添加边界检查，确保变换后的坐标不会超出数据的边界。
# 随机性增强：通过随机选择节点数量和参数，增加了数据增强的随机性。

# def random_move(data_numpy,
#                 angle_candidate=[-10., -5., 0., 5., 10.],
#                 scale_candidate=[0.9, 1.0, 1.1],
#                 transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
#                 move_time_candidate=[1]):
#     # input: C,T,V,M
#     C, T, V, M = data_numpy.shape
#     move_time = random.choice(move_time_candidate)
#     node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
#     node = np.append(node, T)
#     num_node = len(node)

#     A = np.random.choice(angle_candidate, num_node)
#     S = np.random.choice(scale_candidate, num_node)
#     T_x = np.random.choice(transform_candidate, num_node)
#     T_y = np.random.choice(transform_candidate, num_node)

#     a = np.zeros(T)
#     s = np.zeros(T)
#     t_x = np.zeros(T)
#     t_y = np.zeros(T)

#     # linspace
#     for i in range(num_node - 1):
#         a[node[i]:node[i + 1]] = np.linspace(
#             A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
#         s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
#                                              node[i + 1] - node[i])
#         t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
#                                                node[i + 1] - node[i])
#         t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
#                                                node[i + 1] - node[i])

#     theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
#                       [np.sin(a) * s, np.cos(a) * s]])  # xuanzhuan juzhen

#     # perform transformation
#     for i_frame in range(T):
#         xy = data_numpy[0:2, i_frame, :, :]
#         new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
#         new_xy[0] += t_x[i_frame]
#         new_xy[1] += t_y[i_frame]  # pingyi bianhuan
#         data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

#     return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M 偏移其中一段
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy
