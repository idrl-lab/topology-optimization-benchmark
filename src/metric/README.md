# Metrics for benchmark

## 介绍

>  本项目根据不同的需求构造了不同的metric准则，评价模型训练的好坏。

## Metrics准则

> 根据不同的需求，构造了pixel-level metrics，image-level metrics和batch-level metrics

* Pixel-level metrics
  * `value_and_pos_error_of_maximum_temperature`: 最高温的预测误差和最高温发生位置的预测误差
    * 可选参数`output_type`：`value`和`position`，默认`value`，其中`value`输出最高温预测误差，`position`输出最高温位置预测误差。
  
* Image-level metrics
  * `mae_global`: 全局温度平均预测误差
  * `mae_boundary`: 边界处温度平均预测误差
    * 可选参数`output_type`：`Dirichlet`和`Neumann`，默认`Dirichlet`，其中`Dirichlet`输出`Dirichlet`边界处温度平均预测误差，`Neumann`输出`Neumann`边界处温度平均预测误差。
  * `mae_component`: 最大的组件处温度平均预测误差
  * `global_image_spearmanr`: 预测温度场和真实温度场的Spearman相关系数

- Batch-level metrics
  - `max_tem_spearmanr`: 不同样本的预测最高温排序和真实最高温排序的Spearman相关系数，衡量代理模型对不同布局对应的最高温进行正确排序的能力

## 其他
