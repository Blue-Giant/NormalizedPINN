import numpy as np
import torch
import scipy.stats.qmc as stqmc


# ---------------------------------------------- 数据集的生成 ---------------------------------------------------
#  方形区域[a,b]^n生成随机数, n代表变量个数
def rand_it(batch_size, variable_dim, region_a=0.0, region_b=1.0, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
            use_grad2x=False, lhs_sampling=True):

    if lhs_sampling:
        sampler = stqmc.LatinHypercube(d=variable_dim)
        x_it = (region_b - region_a) * sampler.random(batch_size) + region_a
    else:
        x_it = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    x_it = np.reshape(x_it, [batch_size, variable_dim])
    if to_float:
        x_it = x_it.astype(np.float32)

    if to_torch:
        x_it = torch.from_numpy(x_it)

        if to_cuda:
            x_it = x_it.cuda(device='cuda:' + str(gpu_no))

        x_it.requires_grad = use_grad2x

    return x_it


def rand_bd_1D(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
               use_grad2x=False):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    assert (int(variable_dim) == 1)

    region_a = float(region_a)
    region_b = float(region_b)

    x_left_bd = np.ones(shape=[batch_size, 1], dtype=np.float32) * region_a
    x_right_bd = np.ones(shape=[batch_size, 1], dtype=np.float32) * region_b
    if to_float:
        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)

    if to_torch:
        x_left_bd = torch.from_numpy(x_left_bd)
        x_right_bd = torch.from_numpy(x_right_bd)

        if to_cuda:
            x_left_bd = x_left_bd.cuda(device='cuda:' + str(gpu_no))
            x_right_bd = x_right_bd.cuda(device='cuda:' + str(gpu_no))

        x_left_bd.requires_grad = use_grad2x
        x_right_bd.requires_grad = use_grad2x

    return x_left_bd, x_right_bd


def rand_bd_2D(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
               use_grad=False):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    # np.random.random((100, 50)) 上方代表生成100行 50列的随机浮点数，浮点数范围 : (0,1)
    # np.random.random([100, 50]) 和 np.random.random((100, 50)) 效果一样
    region_a = float(region_a)
    region_b = float(region_b)
    assert (int(variable_dim) == 2)
    x_left_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a  # 浮点数都是从0-1中随机。
    x_right_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
    y_bottom_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
    y_top_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
    for ii in range(batch_size):
        x_left_bd[ii, 0] = region_a
        x_right_bd[ii, 0] = region_b
        y_bottom_bd[ii, 1] = region_a
        y_top_bd[ii, 1] = region_b

    if to_float:
        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)
        y_bottom_bd = y_bottom_bd.astype(np.float32)
        y_top_bd = y_top_bd.astype(np.float32)
    if to_torch:
        x_left_bd = torch.from_numpy(x_left_bd)
        x_right_bd = torch.from_numpy(x_right_bd)
        y_bottom_bd = torch.from_numpy(y_bottom_bd)
        y_top_bd = torch.from_numpy(y_top_bd)
        if to_cuda:
            x_left_bd = x_left_bd.cuda(device='cuda:' + str(gpu_no))
            x_right_bd = x_right_bd.cuda(device='cuda:' + str(gpu_no))
            y_bottom_bd = y_bottom_bd.cuda(device='cuda:' + str(gpu_no))
            y_top_bd = y_top_bd.cuda(device='cuda:' + str(gpu_no))

        x_left_bd.requires_grad = use_grad
        x_right_bd.requires_grad = use_grad
        y_bottom_bd.requires_grad = use_grad
        y_top_bd.requires_grad = use_grad

    return x_left_bd, x_right_bd, y_bottom_bd, y_top_bd


def rand_bd_3D(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
               use_grad=False):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    assert (int(variable_dim) == 3)

    bottom_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
    top_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
    left_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
    right_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
    front_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
    behind_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
    for ii in range(batch_size):
        bottom_bd[ii, 1] = region_a
        top_bd[ii, 1] = region_b
        left_bd[ii, 0] = region_a
        right_bd[ii, 0] = region_b
        front_bd[ii, 2] = region_a
        behind_bd[ii, 2] = region_b

    if to_float:
        bottom_bd = bottom_bd.astype(np.float32)
        top_bd = top_bd.astype(np.float32)
        left_bd = left_bd.astype(np.float32)
        right_bd = right_bd.astype(np.float32)
        front_bd = front_bd.astype(np.float32)
        behind_bd = behind_bd.astype(np.float32)
    if to_torch:
        bottom_bd = torch.from_numpy(bottom_bd)
        top_bd = torch.from_numpy(top_bd)
        left_bd = torch.from_numpy(left_bd)
        right_bd = torch.from_numpy(right_bd)
        front_bd = torch.from_numpy(front_bd)
        behind_bd = torch.from_numpy(behind_bd)

        if to_cuda:
            bottom_bd = bottom_bd.cuda(device='cuda:' + str(gpu_no))
            top_bd = top_bd.cuda(device='cuda:' + str(gpu_no))
            left_bd = left_bd.cuda(device='cuda:' + str(gpu_no))
            right_bd = right_bd.cuda(device='cuda:' + str(gpu_no))
            front_bd = front_bd.cuda(device='cuda:' + str(gpu_no))
            behind_bd = behind_bd.cuda(device='cuda:' + str(gpu_no))

        bottom_bd.requires_grad = use_grad
        top_bd.requires_grad = use_grad
        left_bd.requires_grad = use_grad
        right_bd.requires_grad = use_grad
        front_bd.requires_grad = use_grad
        behind_bd.requires_grad = use_grad

    return bottom_bd, top_bd, left_bd, right_bd, front_bd, behind_bd


def rand_bd_4D(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
               use_grad=False):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    assert (int(variable_dim) == 4)

    x0a = (region_b - region_a) * np.random.rand(batch_size, 4) + region_a
    x0b = (region_b - region_a) * np.random.rand(batch_size, 4) + region_a
    x1a = (region_b - region_a) * np.random.rand(batch_size, 4) + region_a
    x1b = (region_b - region_a) * np.random.rand(batch_size, 4) + region_a
    x2a = (region_b - region_a) * np.random.rand(batch_size, 4) + region_a
    x2b = (region_b - region_a) * np.random.rand(batch_size, 4) + region_a
    x3a = (region_b - region_a) * np.random.rand(batch_size, 4) + region_a
    x3b = (region_b - region_a) * np.random.rand(batch_size, 4) + region_a
    for ii in range(batch_size):
        x0a[ii, 0] = region_a
        x0b[ii, 0] = region_b
        x1a[ii, 1] = region_a
        x1b[ii, 1] = region_b
        x2a[ii, 2] = region_a
        x2b[ii, 2] = region_b
        x3a[ii, 3] = region_a
        x3b[ii, 3] = region_b

    if to_float:
        x0a = x0a.astype(np.float32)
        x0b = x0b.astype(np.float32)

        x1a = x1a.astype(np.float32)
        x1b = x1b.astype(np.float32)

        x2a = x2a.astype(np.float32)
        x2b = x2b.astype(np.float32)

        x3a = x3a.astype(np.float32)
        x3b = x3b.astype(np.float32)

    if to_torch:
        x0a = torch.from_numpy(x0a)
        x0b = torch.from_numpy(x0b)
        x1a = torch.from_numpy(x1a)
        x1b = torch.from_numpy(x1b)
        x2a = torch.from_numpy(x2a)
        x2b = torch.from_numpy(x2b)
        x3a = torch.from_numpy(x3a)
        x3b = torch.from_numpy(x3b)

        if to_cuda:
            x0a = x0a.cuda(device='cuda:' + str(gpu_no))
            x0b = x0b.cuda(device='cuda:' + str(gpu_no))
            x1a = x1a.cuda(device='cuda:' + str(gpu_no))
            x1b = x1b.cuda(device='cuda:' + str(gpu_no))
            x2a = x2a.cuda(device='cuda:' + str(gpu_no))
            x2b = x2b.cuda(device='cuda:' + str(gpu_no))
            x3a = x3a.cuda(device='cuda:' + str(gpu_no))
            x3b = x3b.cuda(device='cuda:' + str(gpu_no))

    return x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b


def rand_bd_5D(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
               use_grad=False):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    assert variable_dim == 5
    x0a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
    x0b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
    x1a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
    x1b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
    x2a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
    x2b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
    x3a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
    x3b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
    x4a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
    x4b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
    for ii in range(batch_size):
        x0a[ii, 0] = region_a
        x0b[ii, 0] = region_b

        x1a[ii, 1] = region_a
        x1b[ii, 1] = region_b

        x2a[ii, 2] = region_a
        x2b[ii, 2] = region_b

        x3a[ii, 3] = region_a
        x3b[ii, 3] = region_b

        x4a[ii, 4] = region_a
        x4b[ii, 4] = region_b

    if to_float:
        x0a = x0a.astype(np.float32)
        x0b = x0b.astype(np.float32)

        x1a = x1a.astype(np.float32)
        x1b = x1b.astype(np.float32)

        x2a = x2a.astype(np.float32)
        x2b = x2b.astype(np.float32)

        x3a = x3a.astype(np.float32)
        x3b = x3b.astype(np.float32)

        x4a = x4a.astype(np.float32)
        x4b = x4b.astype(np.float32)

    if to_torch:
        x0a = torch.from_numpy(x0a)
        x0b = torch.from_numpy(x0b)
        x1a = torch.from_numpy(x1a)
        x1b = torch.from_numpy(x1b)
        x2a = torch.from_numpy(x2a)
        x2b = torch.from_numpy(x2b)
        x3a = torch.from_numpy(x3a)
        x3b = torch.from_numpy(x3b)
        x4a = torch.from_numpy(x4a)
        x4b = torch.from_numpy(x4b)

        if to_cuda:
            x0a = x0a.cuda(device='cuda:' + str(gpu_no))
            x0b = x0b.cuda(device='cuda:' + str(gpu_no))
            x1a = x1a.cuda(device='cuda:' + str(gpu_no))
            x1b = x1b.cuda(device='cuda:' + str(gpu_no))
            x2a = x2a.cuda(device='cuda:' + str(gpu_no))
            x2b = x2b.cuda(device='cuda:' + str(gpu_no))
            x3a = x3a.cuda(device='cuda:' + str(gpu_no))
            x3b = x3b.cuda(device='cuda:' + str(gpu_no))
            x4a = x4a.cuda(device='cuda:' + str(gpu_no))
            x4b = x4b.cuda(device='cuda:' + str(gpu_no))

    return x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, x4a, x4b


def rand_bd_2D1(batch_size, variable_dim, region_a, region_b, init_l, init_r, to_torch=True, to_float=True,
                to_cuda=False, gpu_no=0, use_grad=False, lhs_sampling=True):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    # np.random.random((100, 50)) 上方代表生成100行 50列的随机浮点数，浮点数范围 : (0,1)
    # np.random.random([100, 50]) 和 np.random.random((100, 50)) 效果一样
    region_a = float(region_a)
    region_b = float(region_b)
    assert (int(variable_dim) == 2)
    if lhs_sampling:
        x_left_bd = (init_r - init_l) * lhs(2, batch_size) + init_l  # 浮点数都是从0-1中随机。
        x_right_bd = (init_r - init_l) * lhs(2, batch_size) + init_l  # 浮点数都是从0-1中随机。
        y_bottom_bd = (region_b - region_a) * lhs(2, batch_size) + region_a
        y_top_bd = (region_b - region_a) * lhs(2, batch_size) + region_a
    else:
        x_left_bd = (init_r - init_l) * np.random.random([batch_size, 2]) + init_l  # 浮点数都是从0-1中随机。
        x_right_bd = (init_r - init_l) * np.random.random([batch_size, 2]) + init_l  # 浮点数都是从0-1中随机。
        y_bottom_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
        y_top_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
    for ii in range(batch_size):
        x_left_bd[ii, 0] = init_l
        x_right_bd[ii, 0] = init_r
        y_bottom_bd[ii, 1] = region_a
        y_top_bd[ii, 1] = region_b

    if to_float:
        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)
        y_bottom_bd = y_bottom_bd.astype(np.float32)
        y_top_bd = y_top_bd.astype(np.float32)
    if to_torch:
        x_left_bd = torch.from_numpy(x_left_bd)
        x_right_bd = torch.from_numpy(x_right_bd)
        y_bottom_bd = torch.from_numpy(y_bottom_bd)
        y_top_bd = torch.from_numpy(y_top_bd)
        if to_cuda:
            x_left_bd = x_left_bd.cuda(device='cuda:' + str(gpu_no))
            x_right_bd = x_right_bd.cuda(device='cuda:' + str(gpu_no))
            y_bottom_bd = y_bottom_bd.cuda(device='cuda:' + str(gpu_no))
            y_top_bd = y_top_bd.cuda(device='cuda:' + str(gpu_no))

        x_left_bd.requires_grad = use_grad
        x_right_bd.requires_grad = use_grad
        y_bottom_bd.requires_grad = use_grad
        y_top_bd.requires_grad = use_grad

    return x_left_bd, x_right_bd, y_bottom_bd, y_top_bd


def rand_bd_2DV(batch_size, variable_dim, region_l, region_r, region_b, region_t,
                to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
                use_grad=False):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_l = float(region_l)
    region_r = float(region_r)
    region_b = float(region_b)
    region_t = float(region_t)

    assert (int(variable_dim) == 2)
    left_bd = np.random.rand(batch_size, 2)
    right_bd = np.random.rand(batch_size, 2)
    bottom_bd = np.random.rand(batch_size, 2)
    top_bd = np.random.rand(batch_size, 2)
    # 放缩过程
    left_bd = scale2D(left_bd, region_l, region_r, region_b, region_t)
    right_bd = scale2D(right_bd, region_l, region_r, region_b, region_t)
    bottom_bd = scale2D(bottom_bd, region_l, region_r, region_b, region_t)
    top_bd = scale2D(top_bd, region_l, region_r, region_b, region_t)

    left_bd[:, 0] = region_l
    right_bd[:, 0] = region_r
    bottom_bd[:, 1] = region_b
    top_bd[:, 1] = region_t

    if to_float:
        bottom_bd = bottom_bd.astype(np.float32)
        top_bd = top_bd.astype(np.float32)
        left_bd = left_bd.astype(np.float32)
        right_bd = right_bd.astype(np.float32)


    if to_torch:
        bottom_bd = torch.from_numpy(bottom_bd)
        top_bd = torch.from_numpy(top_bd)
        left_bd = torch.from_numpy(left_bd)
        right_bd = torch.from_numpy(right_bd)


        if to_cuda:
            bottom_bd = bottom_bd.cuda(device='cuda:' + str(gpu_no))
            top_bd = top_bd.cuda(device='cuda:' + str(gpu_no))
            left_bd = left_bd.cuda(device='cuda:' + str(gpu_no))
            right_bd = right_bd.cuda(device='cuda:' + str(gpu_no))


        bottom_bd.requires_grad = use_grad
        top_bd.requires_grad = use_grad
        left_bd.requires_grad = use_grad
        right_bd.requires_grad = use_grad

    return bottom_bd, top_bd, left_bd, right_bd


def scale2D(x, a, b, c, d):
    x[:, 0] = (b - a) * x[:, 0] + a
    x[:, 1] = (d - c) * x[:, 1] + c
    return x


def scale3D(x, a, b, c, d, e, f):
    x[:, 0] = (b - a) * x[:, 0] + a
    x[:, 1] = (d - c) * x[:, 1] + c
    x[:, 2] = (f - e) * x[:, 2] + e
    return x


def rand2D_2(batch_size, variable_dim, region_l, region_r, region_b, region_t,
             to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
             use_grad=False):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_l = float(region_l)
    region_r = float(region_r)
    region_b = float(region_b)
    region_t = float(region_t)

    assert (int(variable_dim) == 2)
    xy_it = np.random.rand(batch_size, 2)
    xy_it[:, 0] = (region_r - region_l) * xy_it[:, 0] + region_l
    xy_it[:, 1] = (region_t - region_b) * xy_it[:, 1] + region_b

    if to_float:
        xy_it = xy_it.astype(np.float32)

    if to_torch:
        xy_it = torch.from_numpy(xy_it)

        if to_cuda:
            xy_it = xy_it.cuda(device='cuda:' + str(gpu_no))
            if use_grad:
                xy_it.requires_grad = use_grad
    return xy_it


def rand_bd_1DV_neu_hard(batch_size, variable_dim, region_a, region_b, init_l, init_r, to_torch=True, to_float=True,
                         to_cuda=False, gpu_no=0, use_grad=False, lhs_sampling=True):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    # np.random.random((100, 50)) 上方代表生成100行 50列的随机浮点数，浮点数范围 : (0,1)
    # np.random.random([100, 50]) 和 np.random.random((100, 50)) 效果一样
    region_a = float(region_a)
    region_b = float(region_b)
    assert (int(variable_dim) == 2)
    if lhs_sampling:
        x_left_bd = (init_r - init_l) * lhs(2, batch_size) + init_l  # 浮点数都是从0-1中随机。
        x_right_bd = (init_r - init_l) * lhs(2, batch_size) + init_l  # 浮点数都是从0-1中随机。
        y_bottom_bd = (region_b - region_a) * lhs(2, batch_size) + region_a
        y_top_bd = (region_b - region_a) * lhs(2, batch_size) + region_a
    else:
        x_left_bd = (init_r - init_l) * np.random.random([batch_size, 2]) + init_l  # 浮点数都是从0-1中随机。
        x_right_bd = (init_r - init_l) * np.random.random([batch_size, 2]) + init_l  # 浮点数都是从0-1中随机。
        y_bottom_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
        y_top_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
    for ii in range(batch_size):
        x_left_bd[ii, 0] = init_l
        x_right_bd[ii, 0] = init_r
        y_bottom_bd[ii, 1] = region_a
        y_top_bd[ii, 1] = region_b

    if to_float:
        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)
        y_bottom_bd = y_bottom_bd.astype(np.float32)
        y_top_bd = y_top_bd.astype(np.float32)
    if to_torch:
        x_left_bd = torch.from_numpy(x_left_bd)
        x_right_bd = torch.from_numpy(x_right_bd)
        y_bottom_bd = torch.from_numpy(y_bottom_bd)
        y_top_bd = torch.from_numpy(y_top_bd)
        if to_cuda:
            x_left_bd = x_left_bd.cuda(device='cuda:' + str(gpu_no))
            x_right_bd = x_right_bd.cuda(device='cuda:' + str(gpu_no))
            y_bottom_bd = y_bottom_bd.cuda(device='cuda:' + str(gpu_no))
            y_top_bd = y_top_bd.cuda(device='cuda:' + str(gpu_no))

        x_left_bd.requires_grad = use_grad
        x_right_bd.requires_grad = use_grad
        y_bottom_bd.requires_grad = use_grad
        y_top_bd.requires_grad = use_grad

    return x_left_bd, x_right_bd, y_bottom_bd, y_top_bd


def rand2D_2(batch_size, variable_dim, region_l, region_r, region_b, region_t,
             to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
             use_grad=False):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_l = float(region_l)
    region_r = float(region_r)
    region_b = float(region_b)
    region_t = float(region_t)

    assert (int(variable_dim) == 2)
    xy_it = np.random.rand(batch_size, 2)
    xy_it[:, 0] = (region_r - region_l) * xy_it[:, 0] + region_l
    xy_it[:, 1] = (region_t - region_b) * xy_it[:, 1] + region_b

    if to_float:
        xy_it = xy_it.astype(np.float32)

    if to_torch:
        xy_it = torch.from_numpy(xy_it)

        if to_cuda:
            xy_it = xy_it.cuda(device='cuda:' + str(gpu_no))
        if use_grad:
                xy_it.requires_grad = use_grad
    return xy_it


def rand_bd_3DV(batch_size, variable_dim, region_l, region_r, region_b, region_t, region_f, region_be,
                to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
                use_grad=False):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_l = float(region_l)
    region_r = float(region_r)
    region_b = float(region_b)
    region_t = float(region_t)
    region_f = float(region_f)
    region_be = float(region_be)

    assert (int(variable_dim) == 3)

    left_bd =lhs(variable_dim, batch_size)
    right_bd = lhs(variable_dim, batch_size)
    bottom_bd = lhs(variable_dim, batch_size)
    top_bd = lhs(variable_dim, batch_size)
    front_bd = lhs(variable_dim, batch_size)
    behind_bd= lhs(variable_dim, batch_size)
    # 放缩过程
    left_bd = scale3D(left_bd, region_l, region_r, region_b, region_t, region_f, region_be)
    right_bd = scale3D(right_bd, region_l, region_r, region_b, region_t, region_f, region_be)
    bottom_bd = scale3D(bottom_bd, region_l, region_r, region_b, region_t, region_f, region_be)
    top_bd = scale3D(top_bd, region_l, region_r, region_b, region_t, region_f, region_be)
    front_bd = scale3D(top_bd, region_l, region_r, region_b, region_t, region_f, region_be)
    behind_bd = scale3D(top_bd, region_l, region_r, region_b, region_t, region_f, region_be)

    left_bd[:, 0] = region_l
    right_bd[:, 0] = region_r
    bottom_bd[:, 1] = region_b
    top_bd[:, 1] = region_t
    front_bd[:, 2] = region_f
    behind_bd[:, 2] = region_be

    if to_float:
        bottom_bd = bottom_bd.astype(np.float32)
        top_bd = top_bd.astype(np.float32)
        left_bd = left_bd.astype(np.float32)
        right_bd = right_bd.astype(np.float32)
        front_bd = front_bd.astype(np.float32)
        behind_bd = behind_bd.astype(np.float32)


    if to_torch:
        bottom_bd = torch.from_numpy(bottom_bd)
        top_bd = torch.from_numpy(top_bd)
        left_bd = torch.from_numpy(left_bd)
        right_bd = torch.from_numpy(right_bd)
        front_bd = torch.from_numpy(front_bd)
        behind_bd = torch.from_numpy(behind_bd)
        if to_cuda:
            bottom_bd = bottom_bd.cuda(device='cuda:' + str(gpu_no))
            top_bd = top_bd.cuda(device='cuda:' + str(gpu_no))
            left_bd = left_bd.cuda(device='cuda:' + str(gpu_no))
            right_bd = right_bd.cuda(device='cuda:' + str(gpu_no))
            front_bd = front_bd.cuda(device='cuda:' + str(gpu_no))
            behind_bd = behind_bd.cuda(device='cuda:' + str(gpu_no))


        bottom_bd.requires_grad = use_grad
        top_bd.requires_grad = use_grad
        left_bd.requires_grad = use_grad
        right_bd.requires_grad = use_grad
        front_bd.requires_grad = use_grad
        behind_bd.requires_grad = use_grad

    return bottom_bd, top_bd, left_bd, right_bd, front_bd, behind_bd


def rand_it_seq(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
                use_grad2x=False, lhs_sampling=True,i_epoch=None,Max_iter=None):
        # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
        # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
        x_it = np.zeros(variable_dim, batch_size)
        if lhs_sampling:
            x_it[0:batch_size/2] = (region_b - region_a) * lhs(variable_dim, batch_size/2) + region_a
            x_it[batch_size/2:batch_size-1] = (region_b - region_a) * lhs(variable_dim, batch_size / 2) + region_a

        else:
            x_it = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
        x_it = np.reshape(x_it, [batch_size, variable_dim])
        if to_float:
            x_it = x_it.astype(np.float32)

        if to_torch:
            x_it = torch.from_numpy(x_it)

            if to_cuda:
                x_it = x_it.cuda(device='cuda:' + str(gpu_no))

            x_it.requires_grad = use_grad2x

        return x_it


def rand_it_zeromore(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
            use_grad2x=False, lhs_sampling=True, portion=0.3,split= 0.1 ):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    if lhs_sampling:
        sampler = stqmc.LatinHypercube(d=variable_dim)
        x_it_1 = (region_b-region_a) * sampler.random(int(batch_size*(1-portion))) + region_a + split * (region_b-region_a)
        x_it_2 = sampler.random(batch_size-int(batch_size*(1-portion))) * split * (region_b-region_a) + region_a
        x_it = np.concatenate([x_it_1, x_it_2], axis=0)
    else:
        x_it = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    # x_it = np.reshape(x_it, [batch_size, variable_dim])
    if to_float:
        x_it = x_it.astype(np.float32)

    if to_torch:
        x_it = torch.from_numpy(x_it)

        if to_cuda:
            x_it = x_it.cuda(device='cuda:' + str(gpu_no))

        x_it.requires_grad = use_grad2x

    return x_it


if __name__ == "__main__":
    x_it = rand_it_zeromore(batch_size=3000, variable_dim=1, region_a=0, region_b=10, to_torch=False,
                            to_float=True, to_cuda=False, gpu_no=0, use_grad2x=False, lhs_sampling=True)