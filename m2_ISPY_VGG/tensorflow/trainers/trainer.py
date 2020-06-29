from datetime import datetime
import os

import tensorflow as tf

from base.trainer import BaseTrain
from models.vgg16 import VGG16
from data_loader.data_loader import TFRecordShardLoader

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

    def tensorboard_aids(self, image, label, _histogram, i):
        """

        Args:
            image:
            label:
        returns
        """
        image = tf.transpose(image, [3, 1, 2, 0])
        writer = tf.summary.create_file_writer(self.log_name)
        with writer.as_default():
            if not _histogram:
                tf.summary.histogram("input_image", tf.identity(image), step=0)
            else:
                tf.summary.image("input_image", tf.identity(image), step=i, max_outputs=image.shape[0])

        with writer.as_default():
            # Convert one-hot to class label to plot.
            tf.summary.histogram("class_label", tf.math.argmax(label, axis=1), step=0)

        return (image, label)

    def run(self) -> None:
        dataset = self.train.input_fn()
        self.log_name = f"{self.config['job_dir']}{self.config['job_name']}/{datetime.now().strftime('%Y%m%d-%H%M%S')}_images"

        # Send images to tensorboard
        i=0

        for image, label in self.pred.input_fn().as_numpy_iterator():
            if i < 10:
                self.tensorboard_aids(image, label, True, i)
            elif i < 20:
                self.tensorboard_aids(image, label, False, i)
            else:
                break
            i+=1

        self.log_name = f"{self.config['job_dir']}{self.config['job_name']}/{datetime.now().strftime('%Y%m%d-%H%M%S')}_results"

        # intialise the estimator with your model
        model = self.model.model()
        model.summary()

        if self.config["use_stack"]:
            dataset = dataset.map(lambda img, label: ( tf.expand_dims(img[..., img.shape[-1]//2], axis=-1), label))

        model.fit(
            dataset,
            epochs=self.config["num_epochs"],
            verbose=2,
            callbacks=[
                tf.keras.callbacks.CSVLogger("training.logs"),
                tf.keras.callbacks.TensorBoard(
                    log_dir=self.log_name,
                    histogram_freq=1,
                    write_images=True # Odd error currently
                ),
                tf.keras.callbacks.History()
            ]
        )
        os.rename("training.logs", f"{self.log_name}/training.logs")
        print(
            f"Model History: {model.history}"
        )

