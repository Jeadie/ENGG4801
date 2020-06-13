import tensorflow as tf
from datetime import datetime

from base.trainer import BaseTrain
from models.vgg16 import VGG16
from data_loader.data_loader import TFRecordShardLoader
from resample_dataset import ResampledTFRecordDataset

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
        image = tf.transpose(image, [3, 1,2, 0])
        print(f"Image shape: {image.shape}")
        print(f"Label shape: {label.shape}")
        writer = tf.summary.create_file_writer(log_file)
        with writer.as_default():
            tf.summary.image("input_image", tf.identity(image), step=0, max_outputs=image.shape[0])
            tf.summary.histogram("input_image", tf.identity(image), step=0)
   
            # Convert one-hot to class label to plot.
            tf.summary.histogram("class_label", tf.math.argmax(label), step=0)

        return (image, label)

    def run(self) -> None:
        dataset = ResampledTFRecordDataset(self.config, mode="train") # self.train.input
        dataset = dataset.input_fn()

        logs = ResampledTFRecordDataset(self.config, mode="train").input_fn().map(self.tensorboard_aids)

        self.config["batch_size"] = self.train.batch_size

        # intialise the estimator with your model
        model = self.model.model()
        model.summary()

        log_name =  f"{self.config['job_dir']}{self.config['job_name']}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        if self.config["use_stack"]:
            dataset = dataset.map(lambda img, label: ( tf.expand_dims(img[..., img.shape[-1]//2], axis=-1), label))

        model.fit(
            dataset,
            epochs=self.config["num_epochs"],
            verbose=2,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=log_name,
                    histogram_freq=1,
                    write_images=True # Odd error currently
                ),
                tf.keras.callbacks.History()
            ]
        )
        print(
            f"Model History: {model.history}"
        )

