from typing import List, Union

import apache_beam as beam
from apache_beam.pvalue import PCollection
import tensorflow as tf
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
from tfx.components import StatisticsGen, SchemaGen
import tensorflow_data_validation as tfdv

def bytes_feature(value: Union[bytes, str]) -> tf.train.Feature:
    """ Converts a byte or string data type into a bytes_list Tensorflow feature.
    """
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value: float) -> tf.train.Feature:
    """ Converts a float data type into a float_list Tensorflow feature.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_feature(value: int) -> tf.train.Feature:
    """ Converts an integer into a int64_list Tensorflow feature.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64List_feature(value: List[int]) -> tf.train.Feature:
    """ Converts an integer List into a int64_list Tensorflow feature.
        """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def test_infer(x):
    print(x)
    print(tfdv.infer_schema(x))
    return tfdv.infer_schema(x)

def create_schema(examples: PCollection[tf.train.Example]) -> SchemaGen:
    """ Automatically create the Proto schema based on the tf.Examples.

    Args:
        examples:

    Returns:
    """
    statistics = (
        examples
        | "decode" >> tfdv.DecodeTFExample()
        | 'GenerateStatistics' >> tfdv.GenerateStatistics()

        | "infer schema" >> beam.Map(test_infer)
       )
    return statistics
    # statistics_gen = StatisticsGen(
    #     examples=example_gen.outputs['examples'],
    # )
    # return SchemaGen(statistics=statistics_gen.outputs['statistics'])

