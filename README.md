# supervised_layout_benchmark

## Introduction

This project aims to establish a deep neural network (DNN) surrogate modeling benchmark for the topology optimization of multi-component heat conduction problem, providing a set of representative DNN surrogates as baselines as well as the original code files for easy start and comparison.

## Running Requirements

- ### Software

  - python：
  - cuda：
  - pytorch：

- ### Hardware

  - A single GPU with at least 4GB.


## Environment construction

- ``` pip install -r requirements.txt ```

## A quick start

The training, test and visualization can be accessed by running `main.py` file.

  - The data is available at the server address: `\\192.168.2.1\mnt/share1/layout_data/v1.0/data/`（refer to [Readme for samples](https://git.idrl.site/gongzhiqiang/supervised_layout_benchmark/blob/master/samples/README.md)). Remember to modify variable `data_root` in the configuration file `config/config_complex_net.yml` to the right server address.

  - Training

    ```python
    python main.py -m train
    ```

    or

    ```python
    python main.py --mode=train
    ```

- Test

  ```python
  python main.py -m test --test_check_num=21
  ```
  
  or

  ```python
  python main.py --mode=test --test_check_num=21
  ```

  where variable `test_check_num` is the number of the saved model for test.

- Prediction visualization

  ```python
  python main.py -m plot -v 21
  ```

  or 
  ```python
  python main.py --mode=plot --test_check_num=21
  ```

  where variable `test_check_num` `v` is the number of the saved model for plotting.

## Project architecture

- `config`: the configuration file
- `notebook`: the test file for `notebook`
- `outputs`: the output results by `test` and `plot` module. The test results is saved at `outputs/*.csv` and the plotting figures is saved at `outputs/predict_plot/`.
- `src`: including surrogate model, training and testing files.
  - `test.py`: testing files.
  - `train.py`: training files.
  - `plot.py`: prediction visualization files.
  - `data`: data preprocessing and data loading files.
  - `metric`: evaluation metric file. (For details, see [Readme for metric](https://git.idrl.site/gongzhiqiang/supervised_layout_benchmark/blob/master/src/metric/README.md))
  - `models`: DNN surrogate models for the HSL-TFP task.
  - `utils`: useful tool function files.

## One tiny example

One tiny example for training and testing can be accessed based on the following instruction.
* Some training and testing data are available at `samples/data`.
* Based on the original configuration file, run `python main.py` directly for a quick experience of this tiny example.