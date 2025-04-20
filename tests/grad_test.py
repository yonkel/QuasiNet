from modules.Quasi import Quasi
from modules.QuasiModule import QuasiModule
import torch
import torch.nn as nn


def test1():
    x = torch.tensor([
        [0.5836, 0.5555, 0.0326, 0.3965, 0.9331, 0.2323, 0.7176, 0.3473, 0.4592, 0.4075],
        [0.2215, 0.4602, 0.8460, 0.0724, 0.7131, 0.9075, 0.0559, 0.7249, 0.9910, 0.5915]],
        requires_grad=True)

    x1 = torch.clone(x)
    x2 = torch.clone(x)

    print("x1 == x2", x1 == x2)

    w = torch.tensor([
        [0.7875, 0.4008, 0.6059, 0.3469, 0.7291, 0.6232, 0.7926, 0.1019, 0.3405, 0.4138],
        [0.2748, 0.9126, 0.7338, 0.9343, 0.4337, 0.0290, 0.1881, 0.4549, 0.0079, 0.7771],
        [0.2789, 0.7845, 0.3458, 0.8256, 0.5474, 0.6373, 0.3323, 0.0291, 0.1438, 0.1089],
        [0.8338, 0.7457, 0.6759, 0.7751, 0.9579, 0.6779, 0.1361, 0.9070, 0.9394, 0.1012],
        [0.8971, 0.2670, 0.3104, 0.9343, 0.7357, 0.4645, 0.6462, 0.7961, 0.9404, 0.7647]
    ], requires_grad=True)

    q1 = Quasi(10, 5)
    q2 = QuasiModule(10, 5)

    q1.weight = nn.Parameter(torch.clone(w))
    q2.weight = nn.Parameter(torch.clone(w))

    print("q1.weight == q2.weight", q1.weight == q2.weight)

    d1 = q1(x1)
    d2 = q2(x2)

    print("d1", d1)
    print("d1 == d2", d1 == d2)

    y = torch.ones((2, 5))
    loss = torch.nn.MSELoss()
    e1 = loss(y, d1)
    e2 = loss(y, d2)

    print("e1", e1)
    print("e1 == e2", e1 == e2)

    e1.backward()
    e2.backward()

    torch.set_printoptions(precision=20)
    print("q1.weight.grad", q1.weight.grad)
    print("q2.weight.grad", q2.weight.grad)
    print("q2.weight.grad - q1.weight.grad", q2.weight.grad - q1.weight.grad)


def test2():
    x = torch.rand((2, 10))
    w = torch.rand((5, 10))

    q1 = Quasi(10, 5)
    q1.weight = nn.Parameter(torch.clone(w))

    q2 = QuasiModule(10, 5)
    q2.weight = nn.Parameter(torch.clone(w))

    d1 = q1(x)
    d2 = q2(x)

    y = torch.ones((2, 5))
    loss = torch.nn.MSELoss()

    e1 = loss(y, d1)
    e2 = loss(y, d2)

    e1.backward()
    e2.backward()

    # print(q1.weight.grad == q2.weight.grad)
    # print((q1.weight.grad == q2.weight.grad).shape)

    print(q1.weight.grad == q2.weight.grad)
    print(torch.abs(q1.weight.grad - q2.weight.grad))


def test3():
    outs = []
    for i in range(1000):
        outs.append(test2())

    outs = torch.stack(outs)
    print(1000 - torch.sum(outs, axis=0))


if __name__ == '__main__':
    test2()
