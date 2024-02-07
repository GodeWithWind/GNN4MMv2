from munch import Munch

from config import setup_cfg, load_cfg, save_cfg, print_cfg
from data.loader import get_train_loader, get_test_loader, get_val_loader
from solver.activate_solver import Solver
import warnings
warnings.filterwarnings("ignore")


def main(args):
    if args.mode == 'train':
        solver = Solver(args)
        loaders = Munch(train=get_train_loader(**args), val=get_val_loader(**args))
        # 进行训练
        solver.train(loaders)
    elif args.mode == 'test':
        solver = Solver(args)
        test_loader = get_test_loader(**args)
        solver.test(test_loader)
    elif args.mode == 'nni':
        solver = Solver(args)
        loaders = Munch(train=get_train_loader(**args), val=get_val_loader(**args))
        solver.nni(loaders)


if __name__ == '__main__':
    # 测试修改
    cfg = load_cfg()
    setup_cfg(cfg)
    if cfg.mode == 'train':
        save_cfg(cfg)
        print_cfg(cfg)
        main(cfg)
    else:
        print_cfg(cfg)
        main(cfg)
