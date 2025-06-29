from typing import List
import torch
from torch.nn.functional import conv1d, conv2d, pad


class operator_base1D(object):
    '''
        # The base class of 1D finite difference operator
        ----
        * filter: the derivative operator
    '''

    def __init__(self, accuracy, device='cpu') -> None:
        super().__init__()
        self.mainfilter: torch.Tensor
        self.accuracy: int = accuracy
        self.device: torch.device = torch.device(device)
        self.centralfilters = [None, None,
                               torch.tensor(
                                   [[[1., -2., 1.]]], device=self.device),
                               None,
                               torch.tensor(
                                   [[[-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]]], device=self.device),
                               None,
                               torch.tensor(
                                   [[[1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]]], device=self.device),
                               None,
                               torch.tensor(
                                   [[[-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560]]],
                                   device=self.device)
                               ]

        self.centralfilters_4th_derivative = [None, None,
                                              torch.tensor([[[1., -4., 6., -4., 1.]]], device=self.device),
                                              None,
                                              torch.tensor([[[-1 / 6, 2, -13 / 2, 28 / 3, -13 / 2, 2, -1 / 6]]],
                                                           device=self.device),
                                              None,
                                              torch.tensor([[[7 / 240, -2 / 5, 169 / 60, -122 / 15, 91 / 8, -122 / 15,
                                                              169 / 60, -2 / 5, 7 / 240]]], device=self.device),
                                              ]

        self.forwardfilters: List(torch.Tensor) = [None,
                                                   torch.tensor(
                                                       [[[-1., 1.]]], device=self.device),
                                                   torch.tensor(
                                                       [[[-3 / 2, 2., -1 / 2]]], device=self.device),
                                                   torch.tensor(
                                                       [[[-11 / 6, 3, -3 / 2, 1 / 3]]], device=self.device),
                                                   ]

        self.backwardfilters: List(torch.Tensor) = [None,
                                                    torch.tensor(
                                                        [[[-1., 1.]]], device=self.device),
                                                    torch.tensor(
                                                        [[[1 / 2, -2., 3 / 2]]], device=self.device),
                                                    torch.tensor(
                                                        [[[-1 / 3, 3 / 2, -3., 11 / 6]]], device=self.device),
                                                    ]

        self.schemes = {'Central': self.centralfilters,
                        'Forward': self.forwardfilters,
                        'Backward': self.backwardfilters,
                        'Central_4th': self.centralfilters_4th_derivative, }

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        '''
            # The operator
            ----
            * u: the input tensor
            * return: the output tensor
        '''
        raise NotImplementedError


class diffusion1D(operator_base1D):
    def __init__(self, scheme='Central', accuracy=2, device='cpu') -> None:
        super().__init__(accuracy, device)

        assert accuracy % 2 == 0, 'diffusion operator precision must be even number'
        assert scheme == 'Central' or scheme == 'Central_4th', 'scheme must be one of the following: Central, Central_4th'
        self.filters = self.schemes[scheme]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        return conv1d(u, self.filters[self.accuracy])


class advection1d(operator_base1D):
    def __init__(self, scheme='Central', accuracy=2, device='cpu') -> None:
        super().__init__(accuracy, device)

        self.filters = self.schemes[scheme]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class dudx_1D(operator_base1D):
    def __init__(self, scheme='Upwind', accuracy=2, device='cpu') -> None:
        super().__init__(accuracy, device)
        self.schemes['Upwind'] = {
            'forward': [None,
                        torch.tensor(
                            [[[0, -1., 1.]]], device=self.device),
                        torch.tensor(
                            [[[0, 0, -3 / 2, 2., -1 / 2]]], device=self.device),
                        torch.tensor(
                            [[[0, 0, 0, -11 / 6, 3, -3 / 2, 1 / 3]]], device=self.device),
                        ],

            'backward': [None,
                         torch.tensor(
                             [[[-1., 1., 0.]]], device=self.device),
                         torch.tensor(
                             [[[1 / 2, -2., 3 / 2, 0, 0]]], device=self.device),
                         torch.tensor(
                             [[[-1 / 3, 3 / 2, -3., 11 / 6, 0, 0, 0]]], device=self.device),
                         ]}
        self.filter = self.schemes[scheme]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        '''
        only for periodic boundary condition
        '''
        inner = (u[:, :, self.accuracy:-self.accuracy] <= 0) * conv1d(u, self.filter['forward'][self.accuracy]) + \
                (u[:, :, self.accuracy:-self.accuracy] > 0) * conv1d(u, self.filter['backward'][self.accuracy])
        return inner


##################################### 2D #####################################

def permute_y2x(attr_y):
    attr_x = []
    for i in attr_y:
        if i is None:
            attr_x.append(i)
        elif isinstance(i, tuple):
            tmp = (j.permute(0, 1, 3, 2) for j in i)
            attr_x.append(tmp)
        else:
            attr_x.append(i.permute(0, 1, 3, 2))
    return attr_x


class operator_base2D(object):
    def __init__(self, accuracy=2, device='cpu') -> None:
        self.accuracy = accuracy
        self.device = device

        self.centralfilters_y_1 = [None, None,
                                   torch.tensor([[[[-.5, 0, 0.5]]]], device=self.device),
                                   None,
                                   torch.tensor([[[[1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]]]], device=self.device),
                                   ]

        self.centralfilters_y_2nd_derivative = [None, None,
                                                torch.tensor(
                                                    [[[[1., -2., 1.]]]], device=self.device),
                                                None,
                                                torch.tensor(
                                                    [[[[-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]]]], device=self.device),
                                                None,
                                                torch.tensor(
                                                    [[[[1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]]]],
                                                    device=self.device),
                                                None,
                                                torch.tensor(
                                                    [[[[-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5,
                                                        8 / 315, -1 / 560]]]], device=self.device)
                                                ]
        self.centralfilters_y_4th_derivative = [None, None,
                                                torch.tensor([[[[1., -4., 6., -4., 1.]]]], device=self.device),
                                                None,
                                                torch.tensor([[[[-1 / 6, 2, -13 / 2, 28 / 3, -13 / 2, 2, -1 / 6]]]],
                                                             device=self.device),
                                                None,
                                                torch.tensor([[[[7 / 240, -2 / 5, 169 / 60, -122 / 15, 91 / 8,
                                                                 -122 / 15, 169 / 60, -2 / 5, 7 / 240]]]],
                                                             device=self.device),
                                                ]
        self.forwardfilters_y: List(torch.Tensor) = [None,
                                                     torch.tensor(
                                                         [[[[-1., 1.]]]], device=self.device),
                                                     torch.tensor(
                                                         [[[[-3 / 2, 2., -1 / 2]]]], device=self.device),
                                                     torch.tensor(
                                                         [[[[-11 / 6, 3, -3 / 2, 1 / 3]]]], device=self.device),
                                                     ]

        self.backwardfilters_y: List(torch.Tensor) = [None,
                                                      torch.tensor(
                                                          [[[[-1., 1.]]]], device=self.device),
                                                      torch.tensor(
                                                          [[[[1 / 2, -2., 3 / 2]]]], device=self.device),
                                                      torch.tensor(
                                                          [[[[-1 / 3, 3 / 2, -3., 11 / 6]]]], device=self.device),
                                                      ]

        self.centralfilters_x_1: List(torch.Tensor) = permute_y2x(self.centralfilters_y_1)
        self.centralfilters_x_2nd_derivative: List(torch.Tensor) = permute_y2x(self.centralfilters_y_2nd_derivative)
        self.centralfilters_x_4th_derivative: List(torch.Tensor) = permute_y2x(self.centralfilters_y_4th_derivative)
        self.forwardfilters_x: List(torch.Tensor) = permute_y2x(self.forwardfilters_y)
        self.backwardfilters_x: List(torch.Tensor) = permute_y2x(self.backwardfilters_y)

        self.xschemes = {'Central1': self.centralfilters_x_1,
                         'Central2': self.centralfilters_x_2nd_derivative,
                         'Central4': self.centralfilters_x_4th_derivative,
                         'Forward1': self.forwardfilters_x,
                         'Backward1': self.backwardfilters_x}
        self.yschemes = {'Central1': self.centralfilters_y_1,
                         'Central2': self.centralfilters_y_2nd_derivative,
                         'Central4': self.centralfilters_y_4th_derivative,
                         'Forward1': self.forwardfilters_y,
                         'Backward1': self.backwardfilters_y}

        self.yschemes['Upwind1'] = [(None, None),
                                    (torch.tensor([[[[0, -1., 1.]]]], device=self.device),
                                     torch.tensor([[[[-1., 1., 0.]]]], device=self.device)),
                                    (torch.tensor([[[[0, 0, -3 / 2, 2., -1 / 2]]]], device=self.device),
                                     torch.tensor([[[[1 / 2, -2., 3 / 2, 0, 0]]]], device=self.device)),
                                    (torch.tensor([[[[0, 0, 0, -11 / 6, 3, -3 / 2, 1 / 3]]]], device=self.device),
                                     torch.tensor([[[[-1 / 3, 3 / 2, -3., 11 / 6, 0, 0, 0]]]], device=self.device)),
                                    ]

        self.xschemes['Upwind1'] = {}
        for i, v in enumerate(self.yschemes['Upwind1']):
            self.xschemes['Upwind1'][i] = permute_y2x(v)

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class d2udx2_2D(operator_base2D):
    def __init__(self, scheme='Central2', accuracy=2, device='cpu') -> None:
        super().__init__(accuracy, device)
        assert accuracy % 2 == 0, 'diffusion operator precision must be even number'
        assert scheme == 'Central2' or scheme == 'Central4' or scheme == 'Forward1' or scheme == 'Backward1', 'scheme must be one of the following: Central2, Central4, Forward1, Backward1'

        self.filter = self.xschemes[scheme][accuracy]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        return conv2d(u, self.filter)


class d2udy2_2D(operator_base2D):
    def __init__(self, scheme='Central2', accuracy=2, device='cpu') -> None:
        super().__init__(accuracy, device)
        assert accuracy % 2 == 0, 'diffusion operator precision must be even number'
        assert scheme == 'Central2' or scheme == 'Central4' or scheme == 'Forward1' or scheme == 'Backward1', 'scheme must be one of the following: Central2, Central4, Forward1, Backward1'
        self.filter = self.yschemes[scheme][accuracy]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        return conv2d(u, self.filter)


class dudx_2D(operator_base2D):
    def __init__(self, scheme='Upwind1', accuracy=1, device='cpu') -> None:
        super().__init__(accuracy, device)
        self.xscheme = scheme
        self.filter = self.xschemes[scheme][accuracy]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        if self.xscheme == 'Upwind1':
            return (u[:, :, self.accuracy:-self.accuracy] <= 0) * conv2d(u, self.filter[0]) + \
                (u[:, :, self.accuracy:-self.accuracy] > 0) * conv2d(u, self.filter[1])
        else:
            return conv2d(u, self.filter)


class dudy_2D(operator_base2D):
    def __init__(self, scheme='Upwind1', accuracy=1, device='cpu') -> None:
        super().__init__(accuracy, device)
        self.yscheme = scheme
        self.filter = self.yschemes[scheme][accuracy]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        if self.yscheme == 'Upwind1':
            return (u[:, :, :, self.accuracy:-self.accuracy] <= 0) * conv2d(u, self.filter[0]) + \
                (u[:, :, :, self.accuracy:-self.accuracy] > 0) * conv2d(u, self.filter[1])
        else:
            return conv2d(u, self.filter)


if __name__ == '__main__':
    from math import pi, exp
    import matplotlib.pyplot as plt
    import rhs

    # region #################### 2D Burgers ###########################
    torch.manual_seed(10)

    device = torch.device('cuda:0')

    para = 5
    repeat = 5
    mu = torch.linspace(0.025, 0.065, para, device=device).reshape(-1, 1)
    mu = mu.repeat(repeat, 1)
    num_para = para * repeat
    mu = mu.reshape(-1, 1, 1, 1)

    def padbcx(uinner):
        return torch.cat((uinner[:, :, -4:-1], uinner, uinner[:, :, 1:4]), dim=2)


    def padbcy(uinner):
        return torch.cat((uinner[:, :, :, -4:-1], uinner, uinner[:, :, :, 1:4]), dim=3)


    x = torch.linspace(0, 2 * pi, 65, device=device)
    y = torch.linspace(0, 2 * pi, 65, device=device)
    x, y = torch.meshgrid(x, y, indexing='ij')
    x = x.unsqueeze(0).unsqueeze(0).repeat([num_para, 1, 1, 1])
    y = y.unsqueeze(0).unsqueeze(0).repeat([num_para, 1, 1, 1])
    dt = 4e-4
    dx = 2 * pi / 64
    dy = 2 * pi / 64
    dx2 = dx ** 2
    dy2 = dy ** 2
    t = 0

    initu = torch.zeros_like(x)
    initv = torch.zeros_like(y)
    for k in range(-4, 5):
        for l in range(-4, 5):
            initu += torch.randn_like(mu) * torch.sin(k * x + l * y) + torch.randn_like(mu) * torch.cos(k * x + l * y)
            initv += torch.randn_like(mu) * torch.sin(k * x + l * y) + torch.randn_like(mu) * torch.cos(k * x + l * y)
    initu = (initu - initu.amin(dim=(1, 2, 3), keepdim=True)) / (
                initu.amax(dim=(1, 2, 3), keepdim=True) - initu.amin(dim=(1, 2, 3), keepdim=True)) + 0.1
    initv = (initv - initv.amin(dim=(1, 2, 3), keepdim=True)) / (
                initv.amax(dim=(1, 2, 3), keepdim=True) - initv.amin(dim=(1, 2, 3), keepdim=True)) + 0.1
    u = initu
    v = initv

    resultu = []
    resultv = []

    dudx = dudx_2D('Upwind1', accuracy=3, device=device)
    dudy = dudy_2D('Upwind1', accuracy=3, device=device)
    d2udx2 = d2udx2_2D('Central2', accuracy=6, device=device)
    d2udy2 = d2udy2_2D('Central2', accuracy=6, device=device)

    for i in range(10001):

        ux = padbcx(u)
        uy = padbcy(u)
        vx = padbcx(v)
        vy = padbcy(v)

        uv = u * u + v * v
        u = u + dt * rhs.burgers2Dpu(u, v, mu, dudx, dudy, d2udx2, d2udy2, ux, uy, dx, dy, dx2,
                                     dy2)  # (-u*dudx(ux)/dx - v*dudy(uy)/dy + mu*(d2udx2(ux)/dx2+d2udy2(uy)/dy2) + beta*((1-uv)*u+uv*v))
        v = v + dt * rhs.burgers2Dpv(u, v, mu, dudx, dudy, d2udx2, d2udy2, ux, uy, dx, dy, dx2,
                                     dy2)  # (-u*dudx(vx)/dx - v*dudy(vy)/dy + mu*(d2udx2(vx)/dx2+d2udy2(vy)/dy2) + beta*(-uv*u+(1-uv)*v))

        # if i%100==0:
        #     writer.add_figure('velocity',addplot(u,v),i)
        if i % 50 == 0:
            resultu.append(u.detach().cpu())
            resultv.append(v.detach().cpu())
            print(i)

        t += dt
    resultu = torch.cat(resultu, dim=1)
    resultv = torch.cat(resultv, dim=1)
    torch.save(torch.stack((resultu, resultv), dim=2), '/home/LJL/CSX/data5.9/2Dburgers0510.pt')
