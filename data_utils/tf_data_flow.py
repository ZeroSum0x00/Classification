import numpy as np
import tensorflow as tf
from .sequence_data_flow import DataSequencePipeline


class TFDataPipeline(DataSequencePipeline):

    def data_generator(self):
        batch_data = []

        for idx, sample in enumerate(self.dataset):
            metadata = self.load_data(sample)
            batch_data.append(metadata)

            if len(batch_data) == self.batch_size or idx == self.N - 1:
                batch_data = self.collate_batch(batch_data)
                if self.class_weights:
                    batch_weights = np.array([self.class_weights.get(label, 1.0) for label in batch_data["label"]], dtype=np.float32)
                else:
                    batch_weights = np.ones_like(batch_data["label"], dtype=np.float32)

                images = batch_data["image"]
                labels = batch_data["label"]
                sample_weight = batch_weights
                yield images, labels, sample_weight
                batch_data = []

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
