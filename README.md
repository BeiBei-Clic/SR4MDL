# SR4MDL

> Official implementation of the paper: **Symbolic regression via MDLformer-guided search: from minimizing prediction error to minimizing description length** (ICLR 2025), as well as its extended journal submission version:**An MDL-oriented Search Framework for Symbolic Regression** (submitting to TPAMI)

> NOTE: We are organizing the code and data for the extended version (An MDL-oriented Search Framework for Symbolic Regression), which will be updated in a week (before Sep. 18, 2025).

## Installation

推荐直接用 `uv` 在仓库根目录创建虚拟环境：
```bash
export UV_CACHE_DIR=/tmp/uv-cache
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . \
  torch-geometric \
  openai \
  google-genai \
  "git+https://github.com/yuzhTHU/nd2py"
```
其中 `nd2py` 当前会间接依赖 `torch-geometric`、`openai` 和 `google-genai`，上面的命令已经一并安装。

## Train

To train the MDLformer model, you can run the following command:
```bash
python train.py --name demo
```
It will train the model on the synthetic dataset and save the model in the `./results/train/demo/` directory.

## Test

To test the trained MDLformer model, you can run the following command:
```bash
python test.py --name demo --load_model ./results/train/demo/checkpoint.pth
```

## Symbolic Regression

下面这组命令可以直接完成“下载官方权重 + 用 GPU 跑通最小示例”：
```bash
mkdir -p weights
curl -L "https://www.dropbox.com/scl/fi/x1te3v1lmsrrr07r8uunr/checkpoint.pth?rlkey=v7ip8r6b4xuy4pdtk33jsyan5&st=iv36jfg2&dl=1" \
  -o ./weights/checkpoint.pth
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python ./demos/search_mcts4mdl.py \
  --load_model ./weights/checkpoint.pth \
  --name demo \
  --device cuda:0 \
  --function "f=x1+x2*sin(x3)"
```

如果你只想先做一个更快的 smoke test，可以额外加上 `--n_iter 5 --sample_num 32`。

也可以改用 GP4MDL：
```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python ./demos/search_gp4mdl.py \
  --load_model ./weights/checkpoint.pth \
  --name demo \
  --device cuda:0 \
  --function "f=x1+x2*sin(x3)"
```
运行结果会打印到终端，并保存到 `./results/search/` 和 `./results/aggregate.csv`。

如果你已经把本地 PMLB 仓库放到 `./pmlb/datasets`，也可以直接对某个真实数据集做 GPU 推理：
```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python ./experiments/pmlb/pmlb_inference.py \
  --dataset 1027_ESL \
  --load_model ./weights/checkpoint.pth \
  --device cuda:0
```

如果要跑更正式一点的配置，可以显式指定搜索步数和采样行数：
```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python ./experiments/pmlb/pmlb_inference.py \
  --dataset 1027_ESL \
  --load_model ./weights/checkpoint.pth \
  --device cuda:0 \
  --n_iter 1000 \
  --sample_num 200
```
结果会追加保存到 `./experiments/pmlb/results/pmlb_inference.csv`，每次运行的详细日志和 `result.json` 会写到 `./experiments/pmlb/results/<name>/`。

先用很小参数验证本地 PMLB 批量 GPU 推理、无噪声流程和 CSV 落盘：
```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python ./experiments/pmlb/pmlb_batch_inference.py \
  --device cuda:0 \
  --dataset_limit 2 \
  --max_rows 64 \
  --max_input_points 64 \
  --n_iter 5
```

正式全量无噪声批跑时直接指定要用的 GPU：
```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python ./experiments/pmlb/pmlb_batch_inference.py \
  --device cuda:3 \
  --n_iter 100
```

带噪声实验时额外传目标值噪声强度，噪声按 `target` 标准差缩放：
```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python ./experiments/pmlb/pmlb_batch_inference.py \
  --device cuda:2 \
  --n_iter 100 \
  --noise_strength 0.1
```
批量结果默认写到 `./experiments/pmlb/results/pmlb_batch_inference_noise_<noise>.csv`，结果 CSV 也会额外记录 `noise_strength` 列。

If you wanna test this model on Feynman & Strogatz dataset, you have to:
1. Install PMLB package from https://github.com/EpistasisLab/pmlb (`pip install pmlb` is not recommended since it does not contains these datasets, see https://epistasislab.github.io/pmlb/using-python.html)
```bash
cd data
git clone https://github.com/EpistasisLab/pmlb pmlb
pip install ./pmlb
cd ..
```
2. Run the following command:
```bash
python ./demos/search_mcts4mdl.py --load_model ./weights/checkpoint.pth --name demo --function "Feynman_II_27_18"
```
or
```bash
python ./demos/search_gp4mdl.py --load_model ./weights/checkpoint.pth --name demo --function "Feynman_II_27_18"
```
The running result will be shown in the terminal, as well as saved in the `./results/search/` directory and `./results/aggregate.csv` file.

## Run in SRBench

To test our method on the SRBench benchmark, you have to:

1. Clone the SRBench repo from [here](https://github.com/cavalab/srbench), save it to `./benchmark/srbench/` directory:
```bash
git clone https://github.com/cavalab/srbench ./benchmark/srbench/
```

3. Copy the contents of `methods` to `./benchmark/srbench/experiment/methods/`, and remember to replace `/path/to/weights/checkpoint.pth` with the path to the trained model.
```bash
cp ./methods/* ./benchmark/srbench/experiment/methods/
# Replace the /path/to/weights/checkpoint.pth to, for example, ./weights/checkpoint.pth
```
