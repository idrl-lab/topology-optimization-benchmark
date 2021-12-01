# encoding: utf-8
import copy

import torch
import yaml
import numpy as np
import torch.nn.functional as F
from scipy.stats import spearmanr


class Metric:

    def __init__(self,  input, target, boundary=None,
                 layout=None, data_config=None, hparams=None):
        """
        Args:
            input: (batch size x 1 x N x N or N x N) the predicted temperature field
            target: (batch size x 1 x N x N or N x N) the real temperature field
            boundary: 'all_walls' - all the dirichlet BCs
                    : 'rm_wall' - the neumann BCs for three sides and the dirichlet BCs for one side
                    : 'one_point' - all the neumann BCs except one tiny heatsink with dirichlet BC
            layout: Input layout
            data_config: Dataset parameter
            hparams: Model parameter
        """
        self.data_config = data_config
        self.layout = layout
        self.boundary = boundary
        self.input = input
        self.target = target
        self.hparams = hparams
        self.data = None
        self.data_info()
        self.metrics = self.all_metrics()

    def all_metrics(self):
        self.metrics = ["mae_global", "mae_boundary", "mae_component",
         "value_and_pos_error_of_maximum_temperature", "max_tem_spearmanr",
         "global_image_spearmanr"]
        return self.metrics

    def data_info(self):
        data_yaml = open(self.data_config, 'r', encoding='gbk')
        self.data = yaml.load(data_yaml, yaml.FullLoader)
        self.L = self.data['length']
        self.power = np.array(self.data['powers']) / self.hparams.std_layout
        self.comp_size = np.array(self.data['units'])
        self.comp_pixel_size = np.round(self.comp_size / self.L * 200).astype(int)

    # -------------------tool functions--------------------------#
    def identify_same_power(self, power):
        org_power = np.array(power)
        power1 = np.array(list(set(power)))  # 对元素去重
        indx1 = []
        indx2 = []
        indx3 = []
        for i in range(len(power1)):
            ind = np.where(org_power == power1[i])[0]
            if len(ind) == 1:  # 一个组件一个功率
                indx1 = indx1 + list(ind)
            elif len(ind) == 2:  # 两个组件一个功率
                indx2 = indx2 + list(ind)
            elif len(ind) == 3:  # 三个组件一个功率
                indx3 = indx3 + list(ind)
            else:
                print('There are four components with the same intensity!')
        return (indx1, indx2, indx3)

    def identify_component_boundary(self, layout, power, boundary):
        """
        find the pixel locations of components

        Args:
            layout: pixel-level representation
            power: the component dissapation power
            boundary: 'all_walls' - all the dirichlet BCs
                    : 'rm_wall' - the neumann BCs for three sides and the dirichlet BCs for one side
                    : 'one_point' - all the neumann BCs except one tiny heatsink with dirichlet BC
            When boundary is 'one_point', the input layout should be transposed and then read in an inverse-row order.
        Returns:
            location: -> Tensor: N * 4, pixel coordinates
        """
        if boundary == 'one_point':
            temp = layout.cpu().numpy().T[::-1].copy()
            layout = torch.from_numpy(temp)

        comp_num = len(power)
        location = torch.zeros(comp_num, 4)

        (indx1, indx2, indx3) = self.identify_same_power(power)

        for i in range(comp_num):
            [index_x, index_y] = torch.where(layout == power[i])
            if i in indx1:
                xmin, xmax = torch.min(index_x).item(), torch.max(index_x).item()
                ymin, ymax = torch.min(index_y).item(), torch.max(index_y).item()
                location[i, :] = torch.tensor([xmin, xmax, ymin, ymax])
            if i in indx2:  # [3, 8, 4, 10, 7, 11] # 4 和 9 号组件，P=8, 5和11，P=10, 8和12，P=12
                flag1 = 0
                layout_flag = torch.zeros_like(layout)
                layout_flag[index_x, index_y] = 1

                for j in range(int(len(indx2)/2)):
                    temp = indx2[(2*j): (2*j + 2)]
                    if i in temp:
                        comp_index = temp

                comp_coord = self.find_comp_coordinate(layout_flag, self.comp_pixel_size, comp_index)
                if comp_coord is None:
                    pass
                else:
                    location[comp_index[0], :] = torch.tensor(comp_coord[0])
                    location[comp_index[1], :] = torch.tensor(comp_coord[1])
                    flag1 = 1
                if flag1 == 0:
                    print("Something wrong! Cannot locate the component #", i)
            if i in indx3:  # [1, 6, 9]
                if i == 1:
                    flag2 = 0  # to indicate whether locate the components
                layout_flag = torch.zeros_like(layout)
                layout_flag[index_x, index_y] = 1
                xmin1, ymin1 = self.find_left_top_point(index_x, index_y)
                xmax1 = xmin1 + self.comp_pixel_size[i, 0] - 1
                ymax1 = ymin1 + self.comp_pixel_size[i, 1] - 1
                layout_flag[xmin1: (xmax1 + 1), ymin1: (ymax1 + 1)] = 0
                for j in range(int(len(indx3)/3)):
                    temp = indx3[(3*j): (3*j + 3)]
                    if i in temp:
                        comp_index = temp
                comp_index.remove(i)
                comp_coord = self.find_comp_coordinate(layout_flag, self.comp_pixel_size, comp_index)
                if comp_coord is None:
                    pass
                else:
                    location[i, :] = torch.tensor([xmin1, xmax1, ymin1, ymax1])
                    location[comp_index[0], :] = torch.tensor(comp_coord[0])
                    location[comp_index[1], :] = torch.tensor(comp_coord[1])
                    flag2 += 1
                if i == 9 and flag2 == 0:
                    print("Something wrong! Cannot locate components # 2, 7, 10")
        return location

    def find_left_top_point(self, index_x, index_y):
        x_min = torch.min(index_x).item()
        indx_min = torch.where(index_x == torch.min(index_x))[0]
        temp = index_y[indx_min]
        y_min = torch.min(temp).item()
        return (x_min, y_min)

    def find_comp_coordinate(self, layout, comp_pixel_size, comp_index):
        layout_flag = copy.deepcopy(layout)

        indx, indy = torch.where(layout_flag == 1)
        x_min1, y_min1 = self.find_left_top_point(indx, indy)
        x_max1 = x_min1 + comp_pixel_size[comp_index[0], 0] - 1
        y_max1 = y_min1 + comp_pixel_size[comp_index[0], 1] - 1
        layout_flag[x_min1: x_max1 + 1, y_min1: y_max1 + 1] = 0

        indx, indy = torch.where(layout_flag == 1)
        x_min2, y_min2 = self.find_left_top_point(indx, indy)
        x_max2 = x_min2 + comp_pixel_size[comp_index[1], 0] - 1
        y_max2 = y_min2 + comp_pixel_size[comp_index[1], 1] - 1
        layout_flag[x_min2: x_max2 + 1, y_min2: y_max2 + 1] = 0
        if torch.sum(layout_flag) == 0:
            return ([x_min1, x_max1, y_min1, y_max1], [x_min2, x_max2, y_min2, y_max2])
        else:
            layout_flag = copy.deepcopy(layout)
            x_max1 = x_min1 + comp_pixel_size[comp_index[1], 0] - 1
            y_max1 = y_min1 + comp_pixel_size[comp_index[1], 1] - 1
            layout_flag[x_min1: x_max1 + 1, y_min1: y_max1 + 1] = 0
            indx, indy = torch.where(layout_flag == 1)
            x_min2, y_min2 = self.find_left_top_point(indx, indy)
            x_max2 = x_min2 + comp_pixel_size[comp_index[0], 0] - 1
            y_max2 = y_min2 + comp_pixel_size[comp_index[0], 1] - 1
            layout_flag[x_min2: x_max2 + 1, y_min2: y_max2 + 1] = 0
            if torch.sum(layout_flag) == 0:
                return ([x_min2, x_max2, y_min2, y_max2], [x_min1, x_max1, y_min1, y_max1])
            else:
                return None
    # -------------------tool functions--------------------------#

    # --------------metric functions from here-------------------#
    def mae_global(self):
        """
        calculate the global temperature prediction mean absolute error between input and target.

        Returns:
            mae: the mean absolute error of the whole field for a batch of samples
        """
        return F.l1_loss(self.input, self.target, reduction='mean') * self.hparams.std_heat

    def mae_boundary(self, output_type='Dirichlet', reduction='mean'):
        """
        calculate the temperature perdiction mean abosolute error of the boundary of the domain.

        The input and target are tensors.

        Args:
            output_type: 'Dirichlet' for outputing the error of Dirichlet boundary
                         'Neumann' for outputing the error of Neumann boundary
        Returns:
            mae: (dirichlet, neumann) -> tuple: the specific (mean for batch > 1) mae in the boundary
        """
        if self.input.dim() == 2:
            [nx, ny] = self.input.shape
            batch = 1
            std_input = self.input.unsqueeze(0).unsqueeze(0).cpu()
            std_target = self.target.unsqueeze(0).unsqueeze(0).cpu()
        elif self.input.dim() == 4:
            [batch, channel, nx, ny] = self.input.shape
            std_input = self.input.cpu()
            std_target = self.target.cpu()
            if channel != 1:
                raise ValueError('Please input tensors with channel = 1.')
        else:
            raise ValueError("Please input four-dim or two-dim tensors with (batch * 1 *) N * N.")

        num_boundaryelement = 2*nx + 2*ny - 4  # 边界元素总数
        # 初始化边界总 mask
        mask = torch.zeros([nx, ny])
        mask[..., 0, :] = 1
        mask[..., -1, :] = 1
        mask[..., :, 0] = 1
        mask[..., :, -1] = 1
        if self.boundary == 'all_walls':
            num_dBC = num_boundaryelement
            num_nBC = 0
            dBC_mask = mask
            nBC_mask = mask - dBC_mask
        else:
            [index_x, index_y] = torch.where(self.target[0, 0, :, :] == torch.min(self.target[0, 0, :, :]))
            dBC_mask = torch.zeros_like(mask)
            num_dBC = torch.max(torch.tensor([index_x[-1] - index_x[0] + 1, (index_y[-1] - index_y[0] + 1)])).item()
            num_nBC = num_boundaryelement - num_dBC
            dBC_mask[index_x, index_y] = 1
            nBC_mask = mask - dBC_mask
        dBC_mask.unsqueeze_(0).unsqueeze_(0)
        nBC_mask.unsqueeze_(0).unsqueeze_(0)

        dBC_input = std_input * dBC_mask
        dBC_target = std_target * dBC_mask
        nBC_input = std_input * nBC_mask
        nBC_target = std_target * nBC_mask

        dirichletBC_mae = torch.sum(torch.abs(dBC_input - dBC_target), (1, 2, 3)) / num_dBC
        neumannBC_mae = (torch.sum(torch.abs(nBC_input - nBC_target), (1, 2, 3)) / num_nBC if num_nBC else torch.zeros([batch]))
        if reduction == 'mean':
            dir_mae = torch.mean(dirichletBC_mae)
            neu_mae = torch.mean(neumannBC_mae)
        elif reduction == 'max':
            dir_mae = torch.max(dirichletBC_mae)
            neu_mae = torch.max(neumannBC_mae)
        else:
            raise ValueError("Please input reduction with 'mean' or 'max'.")
        if output_type == 'Dirichlet':
            return dir_mae * self.hparams.std_heat
        elif output_type == 'Neumann':
            return neu_mae * self.hparams.std_heat
        else:
            raise ValueError("Please input the right boundary type ('Dirichlet' or 'Neumann').")

    def mae_component(self, xs=None, ys=None):
        """
        calculate the prediction mean absolute error of component-covering area

        Args:
            xs: meshgrid, N * N, when mesh = 'nonuniform', it is needed.
            ys: meshgrid, N * N, when mesh = 'nonuniform', it is needed.
        Returns:
            comp_mae: -> list: with N elements
        Note:
            xs and ys have been generated and added automatically and specifically.
        """
        if self.input.dim() != self.layout.dim():
            raise ValueError("Please input 'layout' with the same size as 'input' tensors.")

        if self.input.dim() == 2:
            [nx, ny] = self.input.shape
            batch = 1
            std_input = self.input.unsqueeze(0).unsqueeze(0).cpu()
            std_target = self.target.unsqueeze(0).unsqueeze(0).cpu()
            std_layout = self.layout.unsqueeze(0).unsqueeze(0).cpu()
        elif self.input.dim() == 4:
            [batch, channel, nx, ny] = self.input.shape
            std_input = self.input.cpu()
            std_target = self.target.cpu()
            std_layout = self.layout
            if channel != 1:
                raise ValueError('Please input tensors with channel = 1.')
        else:
            raise ValueError("Please input four-dim or two-dim tensors with (batch * 1 *) N * N.")

        domain_length = self.L

        mesh = 'uniform'
        if self.boundary == 'one_point':
            mesh = 'nonuniform'
        comp_mae_max_batch = torch.zeros(batch)
        for k in range(batch):
            single_input = std_input[k, 0, :, :]
            single_target = std_target[k, 0, :, :]
            single_layout = std_layout[k, 0, :, :]

            location = self.identify_component_boundary(single_layout, self.power, self.boundary)
            comp_num = len(self.power)
            comp_mae = []
            comp_mask = torch.zeros([comp_num, nx, ny])
            comp_mae = torch.zeros([comp_num])
            for i in range(comp_num):
                [xmin, xmax, ymin, ymax] = location[i, :].numpy().astype(int)
                mask = torch.zeros(nx, ny)
                if mesh == 'uniform':
                    mask[xmin:(xmax + 1), ymin:(ymax + 1)] = 1
                    num_element = (xmax - xmin + 1) * (ymax - ymin + 1)
                else:
                    if xs is None or ys is None:
                        xs = torch.linspace(0, domain_length, steps=200)  # 生成200个均匀排列的数
                        ys = torch.linspace(0, domain_length, steps=200)
                        # 对应有限差分计算过程中的网格自适应加密函数
                        xs = 4 / ((xs[-1] - xs[0])**2) * ((xs - (xs[-1] + xs[0]) / 2)**3) + (xs[0] + xs[-1]) / 2
                        ys = ys**2 / (ys[0] + ys[-1]) + ys[0] * ys[-1] / (ys[0] + ys[-1])
                        xs, ys = torch.meshgrid(xs, ys)
                    x_min = xmin * domain_length / nx
                    x_max = (xmax + 1) * domain_length / nx
                    y_min = ymin * domain_length / ny
                    y_max = (ymax + 1) * domain_length / ny
                    ind = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
                    mask[ind] = 1
                    num_element = torch.sum(mask).item()
                comp_mask[i, :, :] = mask

                comp_input = single_input * mask
                comp_target = single_target * mask

                mae = torch.sum(torch.abs(comp_input - comp_target)) / num_element
                comp_mae[i] = mae
            comp_mae_max = torch.max(comp_mae)
            comp_mae_max_batch[k] = comp_mae_max
        return torch.mean(comp_mae_max_batch) * self.hparams.std_heat

    def value_and_pos_error_of_maximum_temperature(self, output_type='value'):
        """
        calculate the absolute error of the maximum temperature between input and target

        Args:
            output_type: 'value' for outputing the value error of maximum temperature
                         'position' for outputing the position error of maximum temperature
        Returns:
            error_max_tem: batch : the error of the maximum temperature between input and target
            error_max_tem_pos: batch : the element error of the position of the maximum temperature
        """
        if self.input.dim() == 2:
            [nx, ny] = self.input.shape
            batch = 1
            std_input = self.input.unsqueeze(0)
            std_target = self.target.unsqueeze(0)
        elif self.input.dim() == 4:
            [batch, channel, nx, ny] = self.input.shape
            std_input = self.input.squeeze(1)
            std_target = self.target.squeeze(1)
            if channel != 1:
                raise ValueError('Please input tensors with channel = 1.')
        else:
            raise ValueError("Please input four-dim or two-dim tensors with (batch * 1 *) N * N.")

        [input_max_tem, input_ind] = torch.max(std_input.reshape(batch, -1), 1)
        [target_max_tem, target_ind] = torch.max(std_target.reshape(batch, -1), 1)
        # 计算最高温的误差
        error_max_temp = torch.abs(input_max_tem - target_max_tem)
        # 找出最高温对应位置
        input_max_tem_pos = torch.zeros(batch, 2)
        target_max_tem_pos = torch.zeros(batch, 2)
        for i in range(batch):
            ind1 = input_ind[i].item()
            ind2 = target_ind[i].item()
            flag = ind1 % ny
            ind1_x = ((ind1 // ny) if flag > 0 else (ind1 // ny - 1))
            ind1_y = ((flag - 1) if flag > 0 else (ny - 1))
            flag = ind2 % ny
            ind2_x = ((ind2 // ny) if flag > 0 else (ind2 // ny - 1))
            ind2_y = ((flag - 1) if flag > 0 else (ny - 1))
            input_max_tem_pos[i, :] = torch.Tensor([ind1_x, ind1_y])
            target_max_tem_pos[i, :] = torch.Tensor([ind2_x, ind2_y])
        diff_pos = input_max_tem_pos - target_max_tem_pos
        error_max_temp_pos = torch.sum(diff_pos * diff_pos, dim=1).sqrt_()
        if output_type == 'value':
            return torch.mean(error_max_temp) * self.hparams.std_heat
        elif output_type == 'position':
            return torch.mean(error_max_temp_pos)
        else:
            return ValueError("Please input the right output type ('value' or 'position').")

    def max_tem_spearmanr(self):
        """
        calculate the indicator (spearmanr) of the maximum temperature between input and target

        Returns:
            rho: [-1, 1]
            p_value: the smaller the better. (ideal: p_value < 0.05)
        """
        if self.input.dim() == 2:
            [nx, ny] = self.input.shape
            batch = 1
            std_input = self.input.unsqueeze(0)
            std_target = self.target.unsqueeze(0)
        elif self.input.dim() == 4:
            [batch, channel, nx, ny] = self.input.shape
            std_input = self.input.squeeze(1)
            std_target = self.target.squeeze(1)
            if channel != 1:
                raise ValueError('Please input tensors with channel = 1.')
        else:
            raise ValueError("Please input four-dim or two-dim tensors with (batch * 1 *) N * N.")

        if batch == 1:
            raise ValueError('please provide a batch of samples (batch > 1).')
        input_max_tem = torch.max(std_input.reshape(batch, -1), 1)[0].data.cpu().numpy()
        target_max_tem = torch.max(std_target.reshape(batch, -1), 1)[0].data.cpu().numpy()
        rho, p_value = spearmanr(target_max_tem, input_max_tem)
        return torch.tensor(rho)

    def global_image_spearmanr(self):
        """
        calculate the indicator (spearmanr) correlation coefficient between input and target

        Returns:
            rho: [-1, 1]
            p_value: the smaller the better. (ideal: p_value < 0.05)
        """
        if self.input.dim() == 2:
            [nx, ny] = self.input.shape
            batch = 1
            std_input = self.input.unsqueeze(0)
            std_target = self.target.unsqueeze(0)
        elif self.input.dim() == 4:
            [batch, channel, nx, ny] = self.input.shape
            std_input = self.input.squeeze(1)
            std_target = self.target.squeeze(1)
            if channel != 1:
                raise ValueError('Please input tensors with channel = 1.')
        else:
            raise ValueError("Please input four-dim or two-dim tensors with (batch * 1 *) N * N.")

        spear_batch = torch.zeros(batch)
        for i in range(batch):
            single_input = std_input[i, :, :].reshape(-1).data.cpu().numpy()
            single_target = std_target[i, :, :].reshape(-1).data.cpu().numpy()
            rho, p_value = spearmanr(single_input, single_target)
            spear_batch[i] = rho
        return torch.mean(spear_batch)


if __name__ == "__main__":

    data_config = Path(__file__).absolute().parent.parent.parent / "config/data.yml"
    data_yaml = open(data_config, 'r', encoding='gbk')
    data = yaml.load(data_yaml, Loader=yaml.FullLoader)
    L = data['length']
    power = data['powers']
    comp_size = data['units']

    print(np.array(L))
    print(np.array(power))
    print(np.array(comp_size))