# 拓扑优化研究基准

## 介绍

>  该项目主要用于实现卫星结构传热拓扑优化设计不同深度代理模型训练、测试以及高热导率材料分布预测作图.

## 环境要求

- ### 软件要求

  - python：
  - cuda：
  - pytorch：

- ### 硬件要求

  - 大约4GB显存的GPU


## 构建环境

- ``` pip install -r requirements.txt ```

## 快速开始

> 运行训练、测试以及热布局作图统一通过main.py入口.

  - 数据放在[百度网盘](https://pan.baidu.com/s/11LAVPSVq9fBQouIz0Aonkg) `提取码: u8fv`（详见[Readme](https://git.idrl.site/gongzhiqiang/supervised_layout_benchmark/blob/master/samples/README.md))，运行时请修改程序配置文件`config/config_complex_net.yml`中`data_root`输入变量为挂载服务器上数据地址.

  - 训练和测试

    ```python
    python main.py -m train 或者 python main.py --mode=train
    ```

- 测试

  ```python
  python main.py -m test --test_check_num=21 或者 python main.py --mode=test --test_check_num=21
  ```

  其中`test_check_num`是测试输入模型存储的编号.

- 热布局预测作图

  ```python
  python main.py -m plot --test_check_num=21 或者 python main.py --mode=plot --test_check_num=21
  ```

  其中`test_check_num`是作图输入模型存储的编号.

## 项目结构

- `benchmark`目录存放运行所需所有程序
  - `config`存放运行配置文件
  - `notebook`存放`notebook`测试文件
  - `outputs`用于存放`test`和`plot`作图输出结果，测试的输出结果保存在`outputs/*.csv`，`plot`结果保存在`outputs/predict_plot/`
  - `src`用于存放模型文件和测试训练文件
    - `test.py`测试程序
    - `train.py`训练程序
    - `plot.py`预测可视化程序
    - `data`文件夹存放数据预处理和读取程序
    - `models`深度代理模型
    - `utils`工具类文件

## 其他

* 训练测试examples
  * 训练样本测试样本存放于`samples/data`中
  * 原始文件配置环境后，直接运行`python main.py`，即运行example