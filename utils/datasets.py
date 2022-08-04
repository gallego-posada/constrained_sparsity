import json
import logging
import os
import os.path as osp
import time

import torch
import torch.utils.data as torch_data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode


from .imagenet import presets

DEFAULT_NUM_WORKERS = 4


def create_loader(
    dataset,
    batch_size,
    is_train,
    num_workers,
    pin_memory,
    distributed,
):

    if distributed:
        if is_train:
            sampler = torch_data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch_data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        if is_train:
            sampler = torch_data.RandomSampler(dataset)
        else:
            sampler = torch_data.SequentialSampler(dataset)

    loader = torch_data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=None,
    )

    return loader


def get_dataset_path(task_name: str) -> str:
    """
    Retrieves the path in which a dataset is stored from paths.json file, also
    contained under the utils module.
    """

    with open(osp.join(osp.dirname(__file__), "paths.json")) as f:
        all_paths = json.load(f)
        data_path: str = all_paths[task_name]

    return data_path


def get_num_workers(distributed):
    if "SLURM_JOB_ID" in os.environ.keys() and distributed:
        num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        num_workers = DEFAULT_NUM_WORKERS

    return num_workers


def load_dataset(
    dataset_name: str,
    augment: bool = True,
    train_batch_size: int = 256,
    val_batch_size: int = 256,
    distributed: bool = False,
):

    kwargs = {
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
        "distributed": distributed,
    }

    if dataset_name == "mnist":
        # Augment is ignored for MNIST
        return mnist(**kwargs) + ((1, 28, 28),)
    elif dataset_name == "cifar10":
        return cifar10(augment=augment, **kwargs) + ((3, 32, 32),)
    elif dataset_name == "cifar100":
        return cifar100(augment=augment, **kwargs) + ((3, 32, 32),)
    elif dataset_name == "tiny_imagenet":
        return tiny_imagenet(augment=augment, **kwargs) + ((3, 64, 64),)
    elif dataset_name == "imagenet":
        return imagenet(**kwargs) + ((3, 224, 224),)
    elif dataset_name == "imagenet_validation":
        return imagenet_validation(val_batch_size, distributed=distributed) + (
            (3, 224, 224),
        )
    else:
        raise ValueError("Did not understand dataset_name")


def mnist(
    train_batch_size: int = 100, val_batch_size: int = 100, distributed: bool = False
):

    data_path = get_dataset_path("mnist")
    num_workers = get_num_workers(distributed)

    transf = [transforms.ToTensor()]
    transf.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform_data = transforms.Compose(transf)

    train_dataset = datasets.MNIST(
        data_path, train=True, download=True, transform=transform_data
    )
    train_loader = create_loader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        is_train=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        distributed=distributed,
    )

    val_dataset = datasets.MNIST(
        data_path, train=False, download=True, transform=transform_data
    )
    val_loader = create_loader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        is_train=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        distributed=distributed,
    )

    num_classes = 10

    return train_loader, val_loader, num_classes


def cifar10(
    augment: bool = True,
    train_batch_size: int = 128,
    val_batch_size: int = 128,
    distributed: bool = False,
):

    data_path = get_dataset_path("cifar10")
    num_workers = get_num_workers(distributed)

    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    log_text = "Using"
    if augment:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        log_text += " augmented"
    else:
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    logging.info("%s CIFAR 10", log_text)

    train_dataset = datasets.CIFAR10(
        data_path, train=True, download=True, transform=transform_train
    )
    train_loader = create_loader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        is_train=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        distributed=distributed,
    )

    val_dataset = datasets.CIFAR10(
        data_path, train=False, download=True, transform=transform_test
    )
    val_loader = create_loader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        is_train=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        distributed=distributed,
    )

    num_classes = 10

    return train_loader, val_loader, num_classes


def cifar100(
    augment: bool = True,
    train_batch_size: int = 128,
    val_batch_size: int = 128,
    distributed: bool = False,
):

    data_path = get_dataset_path("cifar100")
    num_workers = get_num_workers(distributed)

    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
        std=[x / 255.0 for x in [68.2, 65.4, 70.4]],
    )

    log_text = "Using"
    if augment:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        log_text += " augmented"
    else:
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    logging.info("%s CIFAR 100", log_text)

    train_dataset = datasets.CIFAR100(
        data_path, train=True, download=True, transform=transform_train
    )
    train_loader = create_loader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        is_train=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        distributed=distributed,
    )

    val_dataset = datasets.CIFAR100(
        data_path, train=False, download=True, transform=transform_test
    )
    val_loader = create_loader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        is_train=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        distributed=distributed,
    )

    num_classes = 100

    return train_loader, val_loader, num_classes


## ------------------------------------------------------------- Tiny Imagenet


def create_val_folder():
    """
    Used for Tiny-imagenet dataset
    Copied from https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-06/train.py
    This method is responsible for separating validation images into separate sub folders,
    so that test and val data can be read by the Pytorch dataloaders

    We do this pre-processing only once then store and copy on each compute node.
    """

    data_path = get_dataset_path("tiny_imagenet")

    path = osp.join(
        data_path, "val/images"
    )  # path where validation data is present now
    filename = osp.join(
        data_path, "val/val_annotations.txt"
    )  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = osp.join(path, folder)
        if not osp.exists(newpath):  # check if folder exists
            os.makedirs(newpath)

        if osp.exists(
            osp.join(path, img)
        ):  # Check if image exists in default directory
            os.rename(osp.join(path, img), osp.join(newpath, img))


def tiny_imagenet(
    augment: bool = True,
    train_batch_size: int = 128,
    val_batch_size: int = 128,
    distributed: bool = False,
):

    if "SLURM_JOB_ID" in os.environ.keys():
        data_path = get_dataset_path("tiny_imagenet")
        data_path = osp.join(os.environ["SLURM_TMPDIR"], data_path)
    else:
        data_path = get_dataset_path("tiny_imagenet_local")

    num_workers = get_num_workers(distributed)

    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [127.5, 127.5, 127.5]],
        std=[x / 255.0 for x in [127.5, 127.5, 127.5]],
    )

    if augment:
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=4),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Path to training images folder
    train_root = osp.join(data_path, "train")
    # Path to validation images folder
    validation_root = osp.join(data_path, "val/images")

    train_dataset = datasets.ImageFolder(train_root, transform=transform_train)
    val_dataset = datasets.ImageFolder(validation_root, transform=transform_test)

    train_loader = create_loader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        is_train=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        distributed=distributed,
    )

    val_loader = create_loader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        is_train=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        distributed=distributed,
    )

    num_classes = 200

    return train_loader, val_loader, num_classes


## ------------------------------------------------------------- ImageNet
def imagenet(train_batch_size: int = 32, val_batch_size: int = 32, distributed=False):
    # Data loading code
    print("Loading data")

    if "SLURM_JOB_ID" in os.environ.keys():
        data_path = get_dataset_path("imagenet")
        data_path = osp.join(os.environ["SLURM_TMPDIR"], data_path)
        num_workers = get_num_workers(distributed)
    else:
        data_path = get_dataset_path("imagenet_local")
        num_workers = 2

    traindir = os.path.join(data_path, "train")
    valdir = os.path.join(data_path, "val")

    # Following lines hard-coded based on Pytorch's baseline config
    val_resize_size, val_crop_size, train_crop_size = 256, 224, 224
    interpolation = InterpolationMode("bilinear")

    print("Loading training data")
    st = time.time()
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
        ),
    )
    print("Took", time.time() - st)

    print("Loading validation data")
    preprocessing = presets.ClassificationPresetEval(
        crop_size=val_crop_size,
        resize_size=val_resize_size,
        interpolation=interpolation,
    )
    dataset_test = torchvision.datasets.ImageFolder(valdir, preprocessing)

    # Added the creation of DataLoaders so that this function's returned
    # actual DataLoaders as opposed to the dataset and a Sampler.
    # Pytorch's code creates these loaders in another function.

    train_loader = create_loader(
        dataset,
        batch_size=train_batch_size,
        is_train=True,
        num_workers=num_workers,
        pin_memory=True,
        distributed=distributed,
    )
    test_loader = create_loader(
        dataset_test,
        batch_size=val_batch_size,
        is_train=False,
        num_workers=num_workers,
        pin_memory=True,
        distributed=distributed,
    )

    return train_loader, test_loader, 1000


## ImageNet Validation
def imagenet_validation(val_batch_size: int = 32, distributed=False):

    # Data loading code
    print("Loading data")

    if "SLURM_JOB_ID" in os.environ.keys():
        data_path = get_dataset_path("imagenet")
        data_path = osp.join(os.environ["SLURM_TMPDIR"], data_path)
        num_workers = get_num_workers(distributed)
    else:
        data_path = get_dataset_path("imagenet_local")
        num_workers = 2

    valdir = os.path.join(data_path, "val")

    # Following lines hard-coded based on Pytorch's baseline config
    val_resize_size, val_crop_size = 256, 224
    interpolation = InterpolationMode("bilinear")

    print("Loading validation data")
    preprocessing = presets.ClassificationPresetEval(
        crop_size=val_crop_size,
        resize_size=val_resize_size,
        interpolation=interpolation,
    )
    dataset_test = torchvision.datasets.ImageFolder(valdir, preprocessing)

    print("Creating data loaders")
    # This creates *samplers* for data, not loaders!
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=False
        )
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=val_batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader, 1000


if __name__ == "__main__":

    # create_val_folder()

    train_loader, val_loader, num_classes = imagenet()
