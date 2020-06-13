from typing import Dict, Union

from config import model_config
from data_loader.data_loader import TFRecordShardLoader
from resample_dataset import ResampledTFRecordDataset
from models.vgg16 import VGG16, babyVGG16
from trainers.trainer import Trainer
from utils.utils import get_args, process_config


def init() -> None:
    """
    The main function of the project used to initialise all the required classes
    used when training the model
    """
    # get input arguments
    args = get_args()
    # get static config information
    config = process_config()
    # combine both into dictionary
    config = {**config, **args}
    config.update(model_config)

    # initialise model
    model = babyVGG16(config)
    # create your data generators for each mode
    loader = get_data_loader(config)
    train_data = loader(config, mode="train")

    val_data = loader(config, mode="val")

    test_data = loader(config, mode="test")

    # initialise the estimator
    trainer = Trainer(config, model, train_data, val_data, test_data)

    # start training
    trainer.run()

def get_data_loader(config: Dict[str, object]) -> Union[TFRecordShardLoader, ResampledTFRecordDataset]:
    """ Selects the appropriate Data loader, given config.

    Args:
        config: Parsed user inputted configuration

    Returns:
        The appropriate data loader to use for the trainer.
    """
    loader_name = config.get("data_loader")
    loader_map = {
        "TFRecordShard": TFRecordShardLoader,
        "ResampledTFRecord": ResampledTFRecordDataset,
        "default": TFRecordShardLoader
    }
    return loader_map.get(loader_name, loader_map["default"])

if __name__ == "__main__":
    init()
