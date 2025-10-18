# plot_pareto_fronts.py
import re
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import cycle

def load_fronts(path: Path):
    """Parse text file and return list[list[tuple[float,float]]] = fronts."""
    number = r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?'      # float regex
    pair_re = re.compile(rf'\(\s*({number})\s*,\s*({number})\s*\)')
    
    fronts = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Front'):
                fronts.append([
                    (float(x), float(y)) 
                    for x, y in pair_re.findall(line)
                ])
    return fronts

def plot_fronts(fronts, *, max_fronts=None, figsize=(8, 6),
                alpha=0.8, marker_size=20, logy=True, save=None):
    """Scatter-plot fronts with different colours."""
    if max_fronts is not None:
        fronts = fronts[:max_fronts]
    
    colors = cycle(plt.cm.tab20.colors)    # 20 distinct colours then repeat
    fig, ax = plt.subplots(figsize=figsize)

    xs, ys = zip(*fronts[0])
    ax.scatter(xs, ys, s=marker_size,
                color=next(colors), alpha=alpha, label=f'Front {0}')
    # for idx, front in enumerate(fronts):
    #     if not front:                 # skip empty layers
    #         continue
    #     xs, ys = zip(*front)
    #     ax.scatter(xs, ys, s=marker_size,
    #                color=next(colors), alpha=alpha, label=f'Front {idx}')
    
    ILP_solutions = [(604.26, 1050886.072227), (655.5, 406775.197227), (586.37,1532352.572227),(574.73,2481330.572227),(573.65,2481330.572227),(568.33,2656832.572227),(560.43,3081204.572227),(559.89,2959894.572227),(538.3,4289528.572227),(517.33,11502856.572227),(500.17,16204696.572227),(492.73,17253496.572227),(492.73,17253496.572227),(488.76,24360536.572227),(479.16,31356888.572227),(478.08,31356888.572227)]
    ILP_solutions = [(478.62, 3.13569e+07), (478.62, 3.13569e+07), (478.62, 3.13569e+07), (478.62, 3.13569e+07), (478.62, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (488.76, 2.43605e+07), (492.73, 1.72535e+07), (492.73, 1.72535e+07), (492.73, 1.72535e+07), (493.27, 1.72535e+07), (493.27, 1.72535e+07), (500.17, 1.62047e+07), (500.17, 1.62047e+07), (500.17, 1.62047e+07), (500.17, 1.62047e+07), (500.17, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (517.87, 1.15112e+07), (538.3, 4.28953e+06), (559.89, 2.95989e+06), (560.43, 2.9747e+06), (568.33, 2.65683e+06), (573.65, 2.48133e+06), (574.73, 2.48133e+06), (578.66, 0), (604.26, 1.03273e+06), (655.5, 412849)]
    ILP_solutions = [(478.62, 3.13569e+07), (478.62, 3.13569e+07), (478.62, 3.13569e+07), (478.62, 3.13569e+07), (478.62, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (479.16, 3.13569e+07), (488.76, 2.43605e+07), (492.73, 1.72535e+07), (492.73, 1.72535e+07), (492.73, 1.72535e+07), (493.27, 1.72535e+07), (493.27, 1.72535e+07), (500.17, 1.62047e+07), (500.17, 1.62047e+07), (500.17, 1.62047e+07), (500.17, 1.62047e+07), (500.17, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (500.71, 1.62047e+07), (517.87, 1.15112e+07), (538.3, 4.28953e+06), (559.89, 2.95989e+06), (560.43, 2.9747e+06), (568.33, 2.65683e+06), (573.65, 2.48133e+06), (574.73, 2.48133e+06), (578.66, 1.53235e+06), (604.26, 1.04829e+06), (655.5, 416374)]
    ILP_solutions = set(ILP_solutions)  # 去重
    xs, ys = zip(*ILP_solutions)
    ax.scatter(xs, ys,
                marker='*',         # 五角星
                s=160,              # 更醒目
                alpha=0.4,
                color='red',
                edgecolors='red',
                linewidths=0.6,
                label=f'ILP_solution (CAOE)')
    # ilp_x, ilp_y = 646.94, 448863.400352
    # ax.scatter(ilp_x, ilp_y,
    #         marker='*',         # 五角星
    #         s=160,              # 更醒目
    #         color='red',
    #         edgecolors='black',
    #         linewidths=0.6,
    #         label='ILP_solution (CoOpt=0.01)')
    # ilp_x, ilp_y = 686.19, 441_681.357384
    # ax.scatter(ilp_x, ilp_y,
    #         marker='*',         # 五角星
    #         s=160,              # 更醒目
    #         color='orange',
    #         edgecolors='black',
    #         linewidths=0.6,
    #         label='ILP_solution (CAOE=689)')
    ax.set_xlabel('Area')
    ax.set_ylabel('MED')
    if logy:
        ax.set_yscale('log')         # 常见目标值跨度大时用对数轴
    ax.legend(title='Pareto Layers', fontsize='small',
              loc='best')
    ax.set_title('Non-dominated Sorted Solutions')
    fig.tight_layout()
    
    if save:
        fig.savefig(save, dpi=300)
        print(f'Figure saved to {save}')
    else:
        plt.show()


def plot_fronts(fronts1, fronts2, *, max_fronts=None, figsize=(8, 6),
                alpha=0.8, marker_size=20, logy=True, save=None):
    """Scatter-plot fronts with different colours."""
    
    
    allSolutions1 = []
    for front in fronts1:
        allSolutions1.extend(front)
    allSolutions2 = []
    for front in fronts2:
        allSolutions2.extend(front)
    
    colors = cycle(plt.cm.tab20.colors)    # 20 distinct colours then repeat
    fig, ax = plt.subplots(figsize=figsize)

    xs, ys = zip(*allSolutions1)
    ax.scatter(xs, ys, s=marker_size,
                color=next(colors), alpha=alpha, label=f'AC_EWs')
    xs, ys = zip(*allSolutions2)
    ax.scatter(xs, ys, s=marker_size,
                color=next(colors), alpha=alpha, label=f'AC_General')

    ax.set_xlabel('Area')
    ax.set_ylabel('MED')
    if logy:
        ax.set_yscale('log')         # 常见目标值跨度大时用对数轴
    ax.legend(title='Pareto Layers', fontsize='small',
              loc='best')
    ax.set_title('Non-dominated Sorted Solutions')
    fig.tight_layout()
    
    if save:
        fig.savefig(save, dpi=300)
        print(f'Figure saved to {save}')
    else:
        plt.show()
    


if __name__ == '__main__':
    txt_path = Path("/Users/fch/Python/OPACT-Extension/Training_log_0.0001_ste_counts/non-dominated-sorting.txt")  # ← 改成自己的文件名
    fronts1 = load_fronts(txt_path)
    txt_path = Path("/Users/fch/Python/OPACT-Extension/Training_log_0.0001_ste_counts_general/non-dominated-sorting.txt")  # ← 改成自己的文件名
    fronts2 = load_fronts(txt_path)
    # ——  根据需要自行调整参数 ——
    plot_fronts(fronts1, fronts2,
                max_fronts=10,        # 只画前 10 层；改成 None 则画全部
                marker_size=15,
                logy=True,
                save=None)            # 改成 'pareto.png' 可直接保存
