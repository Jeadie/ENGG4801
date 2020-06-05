import tensorflow as tf
from datetime import datetime

from base.trainer import BaseTrain
from models.vgg16 import VGG16
from data_loader.data_loader import TFRecordShardLoader
import matplotlib.pyplot as plt


class Trainer(BaseTrain):
    def __init__(
        self,
        config: dict,
        model: VGG16,
        train: TFRecordShardLoader,
        val: TFRecordShardLoader,
        pred: TFRecordShardLoader,
    ) -> None:
        """ Constructor.

        Args:
            config: global configuration
            model: input function used to initialise model
            train: the training dataset
            val: the evaluation dataset
            pred: the prediction dataset
        """
        super().__init__(config, model, train, val, pred)

    def tensorboard_aids(self, image, label, log_file = "./"):
        """

        Args:
            image:
            label:
        returns
        """
        writer = tf.summary.create_file_writer(log_file)
        with writer.as_default():
            tf.summary.image("input_image", image, step=0)
            tf.summary.histogram("input_image", image,step=0)
            
            # Convert one-hot to class label to plot.
            tf.summary.histogram("class_label", tf.math.argmax(label), step=0)

        return (image, label)

    def run(self) -> None:

        # intialise the estimator with your model
        model = self.model.model()
        model.summary()

        dataset = self.train.input_fn()

        # Send input images to Tensorboard
        # dataset = dataset.map(lambda x,y: self.tensorboard_aids(x,y, log_file = self.config["job_dir"] + "train/" + datetime.now().strftime("%Y%m%d-%H%M%S")))

        model.fit(
            dataset,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=self.config["job_dir"],
                    histogram_freq=1,
                    # write_images=True # Odd error currently
                ),
                tf.keras.callbacks.History()
            ]
        )
        print(
            f"Model History: {model.history}"
        )

