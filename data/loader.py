from torch_geometric.data import DataLoader
from data.dataset import HomogeneousDataset

dataset_dict = {'GATReal': HomogeneousRealDataset}


def get_train_loader(train_path, user_num, batch_size, model, num_workers=4, **kwargs):
    dataset_class = dataset_dict[model]
    dataset = dataset_class(train_path)

    # 返回loader
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers)


def get_val_loader(val_path, user_num, batch_size, model, num_workers=4, **kwargs):
    dataset_class = dataset_dict[model]
    dataset = dataset_class(val_path)
    # 返回loader
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=num_workers)


def get_test_loader(test_path, user_num, batch_size, model, num_workers=4, **kwargs):
    dataset_class = dataset_dict[model]
    dataset = dataset_class(test_path)
    # 返回loader
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=num_workers)
