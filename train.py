import os
import json
import signal
import logging
import traceback
import numpy as np
import torch.nn as nn
from setproctitle import setproctitle
from nd2py.utils import seed_all, init_logger, AutoGPU
from sr4mdl.utils import get_args
from sr4mdl.model import MDLformer, FormulaEncoder, Trainer

# Get args
args = get_args(save_dir='./results/train/')

# Init
init_logger('sr4mdl', exp_name=args.name, log_file=os.path.join(args.save_dir, 'info.log'))
logger = logging.getLogger('sr4mdl.train')
logger.info(args)
def handler(signum, frame): raise KeyboardInterrupt
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)
seed_all(args.seed)
setproctitle(f'{args.name}@YuZihan')

# Refine Args
if args.num_workers: args.save_equations = int(np.ceil(args.save_equations / args.num_workers))
assert not (args.load_model and args.continue_from), 'Cannot use --load_model and --continue_from at the same time'
if ',' in args.device:  # e.g., "cuda:0,1,2,3"
    args.device_ids = [int(i) for i in args.device.removeprefix('cuda:').split(',')]
    args.device = f'cuda:{args.device_ids[0]}'
if args.device == 'auto':
    memory_MB = min(1150 + args.batch_size * 154, 80000)
    args.device = AutoGPU().choice_gpu(memory_MB=memory_MB)


def train():
    # Load dataset
    if args.dataset == 'default': # 检查参数，如果数据集类型是 'default'
        from sr4mdl.generator import Num2EqDataset # 从生成器模块导入 Num2EqDataset 类
        dataset = Num2EqDataset(args, beyond_token=True) # 实例化默认的数据集对象
    elif args.dataset == 'hard': # 如果数据集类型是 'hard'
        from sr4mdl.generator import Num2EqDatasetHard # 从生成器模块导入 Num2EqDatasetHard 类
        dataset = Num2EqDatasetHard(args, beyond_token=True) # 实例化困难模式的数据集对象
    elif args.dataset == 'pure': # 如果数据集类型是 'pure'
        from sr4mdl.generator import Num2EqDatasetPure # 从生成器模块导入 Num2EqDatasetPure 类
        dataset = Num2EqDatasetPure(args, beyond_token=True) # 实例化纯净模式的数据集对象
    elif args.dataset == 'keep': # 如果数据集类型是 'keep'
        from sr4mdl.generator import Num2EqDatasetKeep # 从生成器模块导入 Num2EqDatasetKeep 类
        dataset = Num2EqDatasetKeep(args, beyond_token=True) # 实例化保持模式的数据集对象
    elif args.dataset == 'load': # 如果数据集类型是 'load'
        from sr4mdl.generator import Num2EqDatasetLoad # 从生成器模块导入 Num2EqDatasetLoad 类
        from sr4mdl.env import sympy2eqtree, str2sympy # 从环境模块导入转换函数
        with open('./data/load_eqtrees.txt') as f: # 打开一个包含方程式的文本文件
            eqtrees = [sympy2eqtree(str2sympy(eq)) for eq in f.readlines()] # 读取每一行，将其从字符串转换为sympy表达式，再转换为eqtree对象
        dataset = Num2EqDatasetLoad(eqtrees, args, beyond_token=True) # 使用加载的方程式列表实例化数据集对象
    else: # 如果是任何其他未知的数据集类型
        raise ValueError(f'Unknown dataset: {args.dataset}') # 抛出一个值错误，提示数据集类型未知
    loader = dataset.get_dataloader(batch_size=args.batch_size, num_workers=args.num_workers) # 根据实例化的数据集对象，获取数据加载器

    # Load model
    mdlformer = MDLformer(args, dataset.get_token_list())  # 初始化MDLformer模型，它负责将数值数据（数据点）编码为嵌入向量
    eq_encoder = FormulaEncoder(args, dataset.get_token_list(all=True))  # 初始化FormulaEncoder模型，它负责将符号方程式（公式树）编码为嵌入向量
    if 'device_ids' in args:  # 检查命令行参数中是否指定了多个GPU设备ID
        mdlformer = nn.DataParallel(mdlformer, device_ids=args.device_ids)  # 如果是，则使用nn.DataParallel包装MDLformer模型，以在多个GPU上并行处理数据
        eq_encoder = nn.DataParallel(eq_encoder, device_ids=args.device_ids)  # 同样，包装FormulaEncoder模型以实现多GPU并行
    trainer = Trainer(args, mdlformer, eq_encoder)  # 初始化Trainer对象，该对象封装了训练循环、优化器、损失函数等逻辑
    logger.info(mdlformer)  # 使用日志记录器打印MDLformer模型的结构
    logger.info(eq_encoder)  # 使用日志记录器打印FormulaEncoder模型的结构

    # Recovery Checkpoint (if specified)
    if args.continue_from is not None:  # 检查是否设置了 'continue_from' 参数，该参数用于从之前的训练状态恢复
        result_dir = 'results'  # 定义结果目录的名称
        trainer.load(os.path.join(result_dir, args.continue_from, 'checkpoint.pth'), abs_path=True, model_only=False)  # 从指定的检查点文件加载完整的训练状态（模型权重、优化器状态等）
        with open(os.path.join(result_dir, args.continue_from, 'records.json'), 'r') as f:  # 打开之前训练的记录文件
            trainer.records = [json.loads(line) for line in f.readlines()]  # 读取并解析每一行的JSON记录
            trainer.records = trainer.records[:trainer.n_step]  # 截断记录，只保留到当前训练步数的记录
        if os.path.join(result_dir, args.continue_from) != args.save_dir:  # 检查旧的实验目录和新的保存目录是否不同
            with open(os.path.join(args.save_dir, 'records.json'), 'w') as f:  # 如果不同，则在新的保存目录中创建一个记录文件
                for record in trainer.records:  # 遍历截断后的记录
                    f.write(json.dumps(record) + '\n')  # 将记录写入新的文件
            trainer.plot(f'plot.png')  # 重新生成训练过程的图表
        logger.note(f'Continue from ./{result_dir}/{args.continue_from} at step {trainer.n_step}')  # 记录日志，说明从哪个实验的哪个步骤继续训练
    elif args.load_model is not None:  # 如果没有设置 'continue_from'，则检查是否设置了 'load_model' 参数
        trainer.load(args.load_model, abs_path=True, model_only=True)  # 只加载模型的权重，而不加载优化器等其他训练状态
        logger.note(f'Load model from {args.load_model}')  # 记录日志，说明从哪个文件加载了模型权重

    # Training
    try: # 使用 try...except...finally 结构来包裹训练过程，以便优雅地处理中断和错误
        for idx, (data, prefix, used_vars, length) in enumerate(loader): # 遍历数据加载器，每个迭代获取一个批次的数据
            log = trainer.step(data, prefix, used_vars, length, detail=not (trainer.n_step+1) % 10) # 调用 trainer 的 step 方法执行一个训练步骤，并获取日志信息
            if (not trainer.n_step % 10) or (idx < 10):  # 每10个步骤或在前10个步骤中
                logger.info(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items())) # 打印详细的日志信息，并对键进行下划线高亮
            if trainer.n_step >= args.tot_steps: break # 如果当前训练步数达到了总步数，则跳出循环
        logger.note('Training finished:' + (' | '.join(f'{k}:{v}' for k, v in log.items()))) # 训练结束后，记录一条日志，包含最后的日志信息
    except KeyboardInterrupt: # 如果捕获到键盘中断（如 Ctrl+C）
        logger.warning(f'Interrupted at step {trainer.n_step}') # 记录一条警告日志，说明训练在哪个步骤被中断
    except Exception: # 如果捕获到其他任何异常
        logger.error(traceback.format_exc()) # 记录详细的错误和堆栈跟踪信息
    finally: # 无论训练是否成功完成或被中断
        if trainer.n_step >= 100: # 如果训练步数超过100步
            trainer.save('checkpoint.pth') # 保存最终的检查点
            trainer.plot('plot.png') # 生成并保存训练过程的图表


if __name__ == '__main__':
    train()
