import math


def assert_mat(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            assert not math.isnan(data[i][j])
            assert not math.isinf(data[i][j])


def assert_vec(data):
    for i in range(len(data)):
        assert not math.isnan(data[i])
        assert not math.isinf(data[i])
