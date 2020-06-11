import tensorflow as tf
from datetime import datetime

from base.trainer import BaseTrain
from models.vgg16 import VGG16
from data_loader.data_loader import TFRecordShardLoader
from data_loader.view_data import visualise_image


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
            tf.summary.image("input_image", image, step=0, max_outputs=image.shape[0])
            tf.summary.histogram("input_image", image,step=0)
   
            # Convert one-hot to class label to plot.
            tf.summary.histogram("class_label", tf.math.argmax(label), step=0)

        return (image, label)

    def run(self) -> None:

        # intialise the estimator with your model
        # model = self.model.model()
        # model.summary()

        dataset = self.train.input_fn()
        img_dataset = self.train.input_fn()

        imgs = []
        lbls = []
        for img, lbl in img_dataset.take(10):
            imgs.append(img)
            lbls.append(lbl)

        log_name =  self.config["job_dir"] + "train/" + self.config["job_name"] + datetime.now().strftime("%Y%m%d-%H%M%S")

        # visualise_image("name", tf.transpose(tf.concat(imgs, -1), [3, 1,2, 0])[...,0])
        self.tensorboard_aids( tf.concat(imgs, -1), tf.stack(lbls), log_file= log_name)

        if not self.config["use_stack"]:
            dataset = dataset.map(lambda img, label: ( tf.expand_dims(img[..., img.shape[-1]//2], axis=-1), label))

        model.fit(
            dataset,
            epochs=self.config["num_epochs"],
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=log_name
                    histogram_freq=1,
                  write_images=True # Odd error currently
                ),
                tf.keras.callbacks.History()
            ]
        )
        print(
            f"Model History: {model.history}"
        )

