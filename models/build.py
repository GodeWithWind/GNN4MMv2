from data.dataset import HeterogeneousDataset
from models.gnn import GAT, BipGAT, HetGAT,GATReal

model_dict = {'GAT': GAT, 'BipGAT': BipGAT, 'HetGAT': HetGAT,"GATReal":GATReal}


def build_model(args):
    if args.model not in model_dict:
        raise ValueError('Invalid model name!')
    if args.model == "HetGAT":
        dataset = HeterogeneousDataset(args.train_path)
        data = dataset[0]
        model = HetGAT(data.metadata(), args)
    else:
        model_class = model_dict[args.model]
        model = model_class(args)
    return model
