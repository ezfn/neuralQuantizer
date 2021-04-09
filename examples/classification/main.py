import os
from utils.transforms import get_cifar_train_transforms, get_test_transform
import torchvision
import torch
from architectures.mobilenets import get_mobilenet_parts
from examples.classification.modules import encoderMultiClassifier
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import hydra
from omegaconf import DictConfig
import sys
is_debug = sys.gettrace() is not None

@hydra.main( config_path='conf/config.yaml' )
def train(cfg: DictConfig) -> None:
    log_dir_name = f"{cfg.dataset.name}_{cfg.arch.name}_{cfg.quantization}_{cfg.ensemble}".replace(":",'-').replace('\'','').replace(' ', '')
    log_dir_name = log_dir_name.replace('{','')
    log_dir_name = log_dir_name.replace( '}', '')
    log_dir_name = log_dir_name.replace( ',', '_')
    log_dir = os.path.join(os.getcwd(), log_dir_name)
    pre_trained_path = None
    if is_debug:
        print( 'in debug mode!')
        training_params = cfg.training_debug
    else:
        print('in run mode!')
        training_params = cfg.training

    loss = torch.nn.CrossEntropyLoss()
    test_transform = get_test_transform()
    if 'cifar' in cfg.dataset['name']:
        train_transform = get_cifar_train_transforms()
    else:
        train_transform = test_transform

    if cfg.dataset['name'] == 'cifar10':
        trainset = torchvision.datasets.CIFAR10( root=cfg.params.data_path, train=True,
                                                 download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10( root=cfg.params.data_path, train=False,
                                                            download=True, transform=test_transform)
    elif cfg.dataset['name'] == 'cifar100':
        trainset = torchvision.datasets.CIFAR100( root=cfg.params.data_path, train=True,
                                                 download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100( root=cfg.params.data_path, train=False,
                                                            download=True, transform=test_transform)

    trainloader = torch.utils.data.DataLoader( trainset, batch_size=training_params.batch_size,
                                               shuffle=True, num_workers=training_params.num_workers)
    testloader = torch.utils.data.DataLoader( testset, batch_size=training_params.batch_size,
                                              shuffle=False, num_workers=training_params.num_workers)
    if cfg.arch.name == 'mobilenet':
        parts_dict = get_mobilenet_parts( width=cfg.arch.width, pretrained=True, num_classes=cfg.dataset.num_classes,
                                          part_idx=cfg.quantization.part_idx, decoder_copies=cfg.ensemble.n_ensemble,
                                          mobilenet_setup=cfg.arch.mobilenet_setup)
    from pl_bolts.models.detection import faster_rcnn
    module = encoderMultiClassifier.EncoderMultiClassifier(encoder=parts_dict['encoder'], decoder=parts_dict['decoders'],
                                                     primary_loss=loss, n_embed=cfg.quantization.n_embed, commitment=cfg.quantization.commitment_w)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)
    trainer = pl.Trainer(max_epochs=40, gpus=1, logger=tb_logger)
    trainer.fit(module, trainloader, testloader)


if __name__ == '__main__':
    train()

