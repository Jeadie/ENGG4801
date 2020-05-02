from data_loader.data_loader import TFRecordShardLoader
from models.vgg16 import VGG16
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

    # initialise model
    model = VGG16(config)
    # create your data generators for each mode
    train_data = TFRecordShardLoader(config, mode="train")

    val_data = TFRecordShardLoader(config, mode="val")

    test_data = TFRecordShardLoader(config, mode="test")

    # initialise the estimator
    trainer = Trainer(config, model, train_data, val_data, test_data)

    # start training
    trainer.run()


if __name__ == "__main__":
    init()
