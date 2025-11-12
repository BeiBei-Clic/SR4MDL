import os  # 操作系统相关的工具（路径、文件等）
import json  # JSON 编码/解码
import yaml  # YAML 解析器，用来读取 PMLB 的 metadata
import time  # 时间相关函数（例如时间戳）
import torch  # PyTorch，用于加载模型和张量计算
import signal  # 处理系统信号（如 SIGINT）
import logging  # 日志记录模块
import traceback  # 捕获异常栈信息以便记录
import numpy as np  # 数值计算库
import nd2py as nd2  # 项目依赖的 nd2py 库（表达式/节点等）
import pandas as pd  # 数据处理库，用于读取/处理表格数据（PMLB）
from socket import gethostname  # 获取主机名，用于记录
from argparse import ArgumentParser  # 命令行参数解析器
from setproctitle import setproctitle  # 设置进程名，便于 ps/htop 中识别
from nd2py.utils import seed_all, init_logger, AutoGPU, AttrDict  # 工具函数：设置随机种子、初始化日志、自动 GPU 选择、属性字典
from sr4mdl.utils import parse_parser, RMSE_score, R2_score  # 项目工具：解析参数，评估指标
from sr4mdl.search import MCTS4MDL  # 搜索算法实现（MCTS for MDL）
from sr4mdl.model import MDLformer  # 模型定义
from sr4mdl.env import sympy2eqtree, str2sympy, Tokenizer  # 环境工具：符号到 eqtree、字符串到 sympy、分词器


parser = ArgumentParser()  # 创建命令行解析器
parser.add_argument('-f', '--function', type=str, default='f=x1+x2*sin(x3)', help='`f=...\' or `Feynman_xxx\'')  # 目标函数，支持直接表达式或 PMLB 名称
parser.add_argument('-n', '--name', type=str, default=None)  # 运行名字（用于保存目录和日志），默认为 None（由 parse_parser 自动生成时间名）
parser.add_argument('-s', '--seed', type=int, default=0)  # 随机种子，注意 parse_parser 中用 falsy 检查可能会把 0 当作未设置
parser.add_argument('--device', type=str, default='auto')  # 设备选择：'auto' 或 'cpu' 或 'cuda:0' 等
parser.add_argument('--sample_num', type=int, default=200)  # 采样点数量
parser.add_argument('--c', type=float, default=1.41)  # （未被本文件直接使用）可能为搜索超参数
parser.add_argument('--max_len', type=int, default=30)  # 最大长度（用于 token/序列限制）
parser.add_argument('--n_iter', type=int, default=10000)  # MCTS 迭代次数
parser.add_argument('--max_var', type=int, default=10)  # 最大变量数
parser.add_argument('--load_model', type=str, default='./weights/checkpoint.pth')  # 要加载的模型权重路径
parser.add_argument('--quiet', action='store_true')  # 静默模式 flag
parser.add_argument('--keep_vars', action='store_true')  # 是否保持变量不被替换的 flag，传给 MCTS
parser.add_argument('--normalize_y', action='store_true')  # 是否标准化 y
parser.add_argument('--normalize_all', action='store_true')  # 是否对所有输入输出进行标准化
parser.add_argument('--remove_abnormal', action='store_true')  # 是否移除异常样本
parser.add_argument('--use_old_model', action='store_true')  # 使用旧模型格式的兼容 flag
parser.add_argument('--cheat', action='store_true')  # 在 PMLB 情况下是否"作弊"使用真实方程信息


args = parse_parser(parser, save_dir='./results/search/')  # 解析命令行参数并补全（会创建 save_dir、生成 name、处理 unknown args）

init_logger('sr4mdl', args.name, os.path.join(args.save_dir, 'info.log'))  # 初始化日志系统并把日志文件放到 save_dir 下
logger = logging.getLogger('sr4mdl.search')  # 获取模块专用 logger
logger.info(args)  # 输出解析后的参数用于记录
seed_all(args.seed)  # 设置全局随机种子（影响 numpy/torch 等）
def handler(signum, frame): raise KeyboardInterrupt  # 把系统信号转换为 KeyboardInterrupt，方便统一中断处理
signal.signal(signal.SIGINT, handler)  # 绑定 Ctrl-C（SIGINT）信号处理
signal.signal(signal.SIGTERM, handler)  # 绑定终止信号（SIGTERM）处理
setproctitle(f'{args.name}@YuZihan')  # 设置进程标题，便于监控识别

if args.device == 'auto':
    args.device = AutoGPU().choice_gpu(memory_MB=1486, interval=15)  # 如果设备自动选择，使用 AutoGPU 根据可用内存挑选 GPU
args.function = args.function.replace(' ', '')  # 去掉 function 字符串中的空格，方便后续解析


def search():
    if '=' in args.function:
        f = sympy2eqtree(str2sympy(args.function.split('=', 1)[1]))
        binary = list(set(op.__class__ for op in f.iter_preorder() if op.n_operands == 2))  # 所用的二元算子类型
        unary = list(set(op.__class__ for op in f.iter_preorder() if op.n_operands == 1))  # 所用的一元算子类型
        leaf = list(set(op for op in f.iter_preorder() if isinstance(op, nd2.Number)))  # 出现的常数叶节点
        variables = list(set(op.name for op in f.iter_preorder() if isinstance(op, nd2.Variable)))  # 出现的变量名
        X = {var: np.random.uniform(-5, 5, (args.sample_num,)) for var in variables}
        y = f.eval(X)  # 用目标表达式计算标签
        log = {
            'target function': args.function,
            'binary operators': [op.__name__ for op in binary],
            'unary operators': [op.__name__ for op in unary],
            'leaf': [op.to_str(number_format=".2f") for op in leaf],
            'variables': list(X.keys()),
        }
    else:
        import pmlb
        # 只将第一个字母转换为小写，保持其余部分不变
        dataset_name = args.function[0].lower() + args.function[1:] if args.function else args.function
        logger.info(f'fetching {args.function} from PMLB...')
        os.makedirs('./data/pmlb/datasets', exist_ok=True)
        df = pmlb.fetch_data(dataset_name, local_cache_dir='./data/pmlb/datasets/')
        if df.shape[0] > args.sample_num: 
            df = df.sample(args.sample_num)
        else: 
            args.sample_num = df.shape[0]
        logger.info(f'Done, df.shape = {df.shape}')
        X = {col:df[col].values for col in df.columns}
        y = X.pop('target')
        binary = [nd2.Mul, nd2.Div, nd2.Add, nd2.Sub]
        unary = [nd2.Sqrt, nd2.Cos, nd2.Sin, nd2.Pow2, nd2.Pow3, nd2.Exp, nd2.Inv, nd2.Neg, nd2.Arcsin, nd2.Arccos, nd2.Cot, nd2.Log, nd2.Tanh]
        leaf = [nd2.Number(1), nd2.Number(2), nd2.Number(np.pi)]
        log = {
            'target function': args.function,
            'binary operators': [op.__name__ for op in binary],
            'unary operators': [op.__name__ for op in unary],
            'leaf': [op.to_str(number_format=".2f") for op in leaf],
            'variables': list(X.keys()),
        }
        try:
            metadata = yaml.load(open(f'./data/pmlb/datasets/{dataset_name}/metadata.yaml', 'r'), Loader=yaml.Loader)['description']
            metadata = [l.strip() for l in metadata.split('\n')]
            target, eq = metadata[metadata.index('')+1].split(' = ', 1)
            eq = sympy2eqtree(str2sympy(eq))
            log['target function'] = log['target function'] + ' ({} = {})'.format(target, eq.to_str(number_format=".2f"))
            if args.cheat:
                binary = list(set(op.__class__ for op in eq.iter_preorder() if op.n_operands == 2))
                unary = list(set(op.__class__ for op in eq.iter_preorder() if op.n_operands == 1))
                leaf = list(set(op for op in eq.iter_preorder() if isinstance(op, nd2.Number)))
                log['binary operators'] = [op.__name__ for op in binary]
                log['unary operators'] = [op.__name__ for op in unary]
                log['leaf'] = [op.to_str(number_format=".2f") for op in leaf]
        except Exception as e:
            logger.warning(e)
    logger.note('\n'.join(f'{k}: {v if not isinstance(v, list) else "[" + ", ".join(v) + "]"}' for k, v in log.items()))


    tokenizer = Tokenizer(-100, 100, 4, args.max_var)  # 构建符号/参数编码器
    state_dict = torch.load(args.load_model)  # 读取已训练模型权重
    model_args = AttrDict(dropout=0.1, d_model=512, d_input=64, d_output=512, n_TE_layers=8, max_len=50, max_param=5, max_var=args.max_var, uniform_sample_number=args.sample_num,device=args.device, use_SENet=True, use_old_model=args.use_old_model)  # 根据权重配置网络超参
    model = MDLformer(model_args, state_dict['xy_token_list'])  # 初始化模型结构
    model.load(state_dict['xy_encoder'], state_dict['xy_token_list'], strict=True)  # 加载参数到模型
    model.eval()  # 切换到推理模式
    est = MCTS4MDL(
        tokenizer=tokenizer,
        model=model,
        n_iter=args.n_iter,
        keep_vars=args.keep_vars,
        normalize_y=args.normalize_y,
        normalize_all=args.normalize_all,
        remove_abnormal=args.remove_abnormal,
        binary=binary,
        unary=unary,
        leaf=leaf,
        log_per_sec=5,
        save_path=os.path.join(args.save_dir, 'records.json'),
    )

    try:
        est.fit(X, y, use_tqdm=False)
        logger.info('Finished')
    except KeyboardInterrupt:
        logger.note('Interrupted')
    except Exception:
        logger.error(traceback.format_exc())

    y_pred = est.predict(X)
    rmse = RMSE_score(y, y_pred)
    r2 = R2_score(y, y_pred)
    logger.note(f'Result = {est.eqtree}, RMSE = {rmse:.4f}, R2 = {r2:.4f}')

    result = {
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'host': gethostname(),
        'name': args.name,
        'load_model': args.load_model,
        'success': str(rmse < 1e-6),
        'n_iter': len(est.records),
        'duration': est.records[-1]['time'],
        'model': 'MCTS4MDL',
        'exp': args.function,
        'result': str(est.eqtree),
        'rmse': rmse,
        'r2': r2,
        'sample_num': args.sample_num,
        'seed': args.seed,
    }
    json.dump(result, open(os.path.join(args.save_dir, 'result.json'), 'w'), indent=4)

    # aggregate results to aggregate.csv
    save_path = './results/aggregate.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            f.write('\t'.join([
                'success','name','exp','n_iter','duration','seed','rmse','r2',
                'result','target','date','host','load_model','model','sample_num'
            ]) + '\n')
    with open(save_path, 'a') as f:
        keys = open(save_path, 'r').readline().split('\t')
        f.write(','.join(str(result.get(k, '')) for k in keys) + '\n')


if __name__ == '__main__':
    search()
