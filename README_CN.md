# SR4MDL

> 论文**Symbolic regression via MDLformer-guided search: from minimizing prediction error to minimizing description length**（ICLR 2025）的官方实现，以及其扩展的期刊提交版本：**An MDL-oriented Search Framework for Symbolic Regression**（提交至TPAMI）

> 注意：我们正在整理扩展版本（An MDL-oriented Search Framework for Symbolic Regression）的代码和数据，将在一周内更新（2025年9月18日之前）。

## 安装

在开始之前，您可能需要创建一个虚拟环境以避免与其他包发生冲突：
```bash
conda create --prefix ./venv python=3.12 -y
conda activate ./venv
```
我们的代码基于nd2py库，这是一个用纯Python编写的符号系统。您可以通过pip安装或克隆仓库：
```bash
# 通过pip安装
pip install git+https://github.com/yuzhTHU/nd2py

# 或克隆仓库
git clone https://github.com/yuzhTHU/nd2py nd2py_package
pip install ./nd2py_package
```

## 训练

要训练MDLformer模型，您可以运行以下命令：
```bash
python train.py --name demo
```
它将在合成数据集上训练模型，并将模型保存在`./results/train/demo/`目录中。

## 测试

要测试训练好的MDLformer模型，您可以运行以下命令：
```bash
python test.py --name demo --load_model ./results/train/demo/checkpoint.pth
```

## 符号回归

要使用训练好的MDLformer模型进行符号回归，您需要：
1. 将训练好的模型移动到`./weights/checkpoint.pth`。（我们在Github发布页面以及[Dropbox](https://www.dropbox.com/scl/fi/x1te3v1lmsrrr07r8uunr/checkpoint.pth?rlkey=v7ip8r6b4xuy4pdtk33jsyan5&st=iv36jfg2&dl=1)提供了训练好的模型）
2. 运行以下命令：
```bash
python search.py --load_model ./weights/checkpoint.pth --name demo --function "f=x1+x2*sin(x3)"
```
运行结果将显示在终端上，同时也保存在`./results/search/demo/`目录和`./results/aggregate.csv`文件中。

如果您想在Feynman & Strogatz数据集上测试此模型，您需要：
1. 从https://github.com/EpistasisLab/pmlb安装PMLB包（不推荐使用`pip install pmlb`，因为它不包含这些数据集，请参阅https://epistasislab.github.io/pmlb/using-python.html）
```bash
cd data
git clone https://github.com/EpistasisLab/pmlb pmlb
pip install ./pmlb
cd ..
```
2. 运行以下命令：
```bash
python search.py --load_model ./weights/checkpoint.pth --name demo --function "Feynman_II_27_18"
```
运行结果将显示在终端上，同时也保存在`./results/search/demo/`目录和`./results/aggregate.csv`文件中。

## 在SRBench中运行

要在SRBench基准测试上测试我们的方法，您需要：

1. 从[这里](https://github.com/cavalab/srbench)克隆SRBench仓库，保存到`./benchmark/srbench/`目录：
```bash
git clone https://github.com/cavalab/srbench ./benchmark/srbench/
```
2. 创建`./srbench/experiment/methods/sr4mdl`目录，并在其中创建一个空的`__init__.py`文件。
```bash
mkdir -p ./benchmark/srbench/experiment/methods/sr4mdl
touch ./benchmark/srbench/experiment/methods/sr4mdl/__init__.py
```
3. 将`regressor.py`移动到`./srbench/experiment/methods/sr4mdl/`，记得将`/path/to/weights/checkpoint.pth`替换为训练模型的路径。
```bash
cp ./regressor.py ./benchmark/srbench/experiment/methods/sr4mdl/regressor.py
# 将/path/to/weights/checkpoint.pth替换为，例如，./weights/checkpoint.pth
```
5. 从`./srbench/experiment/`目录开始运行以下脚本：
```bash
#!/bin/bash
method=sr4mdl
for seed in 29910 14423 28020 23654 15795 16850 21962 4426 5390 860; do
for noise in 0.000 0.001 0.01 0.1; do
for exp in strogatz_vdp2 feynman_I_6_2a strogatz_bacres2 strogatz_bacres1 feynman_II_27_18 feynman_II_3_24 feynman_I_6_2 feynman_II_8_31 feynman_I_12_1 feynman_I_12_5 feynman_I_14_4 feynman_I_39_1 strogatz_vdp1 feynman_I_25_13 feynman_I_26_2 feynman_I_29_4 strogatz_barmag1 feynman_II_11_28 feynman_II_38_14 strogatz_glider1 feynman_III_12_43 strogatz_shearflow2 strogatz_shearflow1 strogatz_predprey2 strogatz_barmag2 strogatz_predprey1 strogatz_lv2 strogatz_lv1 feynman_I_34_27 strogatz_glider2 feynman_I_12_4 feynman_III_17_37 feynman_I_43_31 feynman_I_14_3 feynman_III_15_27 feynman_I_15_10 feynman_I_16_6 feynman_I_18_12 feynman_I_39_11 feynman_III_15_14 feynman_III_15_12 feynman_II_13_34 feynman_II_13_23 feynman_I_27_6 feynman_II_10_9 feynman_I_30_3 feynman_I_30_5 feynman_I_37_4 feynman_I_34_1 feynman_III_8_54 feynman_I_47_23 feynman_I_10_7 feynman_II_15_4 feynman_II_34_2 feynman_II_34_29a feynman_II_34_2a feynman_test_10 feynman_II_37_1 feynman_I_48_2 feynman_III_7_38 feynman_II_4_23 feynman_I_34_14 feynman_I_6_2b feynman_II_27_16 feynman_II_24_17 feynman_II_8_7 feynman_II_15_5 feynman_I_43_16 feynman_test_5 feynman_I_34_8 feynman_I_50_26 feynman_test_3 feynman_I_38_12 feynman_I_39_22 feynman_test_15 feynman_test_11 feynman_I_8_14 feynman_I_43_43 feynman_test_8 feynman_III_10_19 feynman_I_24_6 feynman_II_13_17 feynman_II_34_11 feynman_II_11_27 feynman_I_32_5 feynman_III_4_33 feynman_III_21_20 feynman_II_38_3 feynman_II_6_11 feynman_II_6_15b feynman_I_12_2 feynman_III_4_32 feynman_I_29_16 feynman_I_13_4 feynman_I_15_3t feynman_I_18_4 feynman_III_13_18 feynman_I_18_14 feynman_I_15_3x feynman_I_12_11 feynman_II_2_42 feynman_test_7 feynman_test_4 feynman_II_34_29b feynman_II_11_3 feynman_II_11_20 feynman_test_18 feynman_II_35_18 feynman_I_44_4 feynman_test_14 feynman_test_13 feynman_test_12 feynman_II_35_21 feynman_test_9 feynman_I_41_16 feynman_III_19_51 feynman_I_13_12 feynman_III_14_14 feynman_II_21_32 feynman_III_9_52 feynman_I_32_17 feynman_test_2 feynman_test_19 feynman_test_17 feynman_II_6_15a feynman_I_11_19 feynman_I_40_1 feynman_test_16 feynman_test_20 feynman_test_1 feynman_test_6 feynman_II_36_38 feynman_I_9_18; do
    python evaluate_model.py ../../data/pmlb/datasets/$exp/$exp.tsv.gz \
        -ml $method \
        -seed $seed \
        -target_noise $noise \
        -results_path "./results-$method-$noise"
done
done
done
```
或者
```bash
python ./benchmark/srbench/experiment/analyze.py \
    ./data/pmlb/datasets/strogatz_* \  # Strogatz数据集
    -time_limit 00:15 \  # 15分钟
    -ml sr4mdl \ # 测试sr4mdl方法
    -n_jobs 2 \  # 2个核心
    -results ./results/srbench/ \  # 将结果保存到此目录
    -sym_data \
    --local \
    -script ./benchmark/srbench/experiment/evaluate_model 

python ./benchmark/srbench/experiment/analyze.py \
    ./data/pmlb/datasets/feynman_* \
    -ml sr4mdl \
    --local \
    -n_jobs 2 \
    -results ./results/srbench/ \
    -script ./benchmark/srbench/experiment/evaluate_model 
    -sym_data
```