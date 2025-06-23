import numpy as np
import tensorflow as tf
from .sequence_data_flow import DataSequencePipeline



class TFDataPipeline(DataSequencePipeline):

    def data_generator(self):
        batch_images = []
        batch_labels = []
        batch_weights = []

        for idx, sample in enumerate(self.dataset):
            image, label = self.load_data(sample)
            batch_images.append(image)
            batch_labels.append(label)

            if self.class_weights:
                weight = self.class_weights.get(label, 1.0)
            else:
                weight = 1.0
                
            batch_weights.append(weight)
        
            if len(batch_images) == self.batch_size or idx == self.N - 1:
                yield (
                    np.array(batch_images, dtype=np.float32),
                    np.array(batch_labels, dtype=np.int32),
                    np.array(batch_weights, dtype=np.float32),
                )
                batch_images = []
                batch_labels = []
                batch_weights = []

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.target_size), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            )
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        if self.phase == "train":
            dataset = dataset.repeat()
        
        return dataset
