import time
import argparse
import numpy as np
import tensorflow as tf
import os
import sys
from pathlib import Path

# ensure project root is importable when running this script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils.sequence_data_flow import DataSequencePipeline
from data_utils.tf_data_flow import TFDataPipeline

# A simple pure-tf.data pipeline implementation for comparison

def build_pure_tf_dataset_from_images(image_array, labels, batch_size, target_size, phase='train', num_workers=1):
    ds = tf.data.Dataset.from_tensor_slices((image_array, labels))
    if phase == 'train':
        ds = ds.shuffle(buffer_size=len(image_array)).repeat()

    def _proc(img, lbl):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, target_size[:2])
        # simple augmentation (only for train)
        img = tf.image.random_flip_left_right(img)
        return img, lbl, tf.cast(1.0, tf.float32)

    ds = ds.map(_proc, num_parallel_calls=num_workers if num_workers > 0 else tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def synthetic_dataset(num_samples, target_size, n_classes):
    images = np.random.randint(0, 256, size=(num_samples, *target_size), dtype=np.uint8)
    labels = np.random.randint(0, n_classes, size=(num_samples,), dtype=np.int32)
    dataset_list = []
    for i in range(num_samples):
        dataset_list.append({
            'image': images[i],
            'label': int(labels[i]),
            'path': '',
            'filename': ''
        })
    return images, labels, dataset_list


def bench_sequence(seq, steps=200):
    # warmup
    for i in range(5):
        _ = seq[i % len(seq)]
    t0 = time.time()
    cnt = 0
    for i in range(steps):
        _ = seq[i % len(seq)]
        cnt += seq.batch_size if hasattr(seq, 'batch_size') else 1
    t1 = time.time()
    secs = t1 - t0
    samples_per_sec = (steps * seq.batch_size) / secs
    return secs, samples_per_sec


def bench_tf_dataset(ds, steps=200):
    it = iter(ds)
    # warmup
    for _ in range(5):
        _ = next(it)
    t0 = time.time()
    samples = 0
    for _ in range(steps):
        batch = next(it)
        samples += batch[0].shape[0]
    t1 = time.time()
    secs = t1 - t0
    return secs, samples / secs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--target_size', type=int, nargs=3, default=[224,224,3])
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    if args.mode == 'synthetic':
        images, labels, dataset_list = synthetic_dataset(args.num_samples, tuple(args.target_size), args.n_classes)

        # DataSequencePipeline
        seq = DataSequencePipeline(
            dataset=dataset_list,
            target_size=tuple(args.target_size),
            batch_size=args.batch_size,
            augmentor=None,
            normalizer='divide',
            phase='train',
            num_workers=args.num_workers
        )

        # TFDataPipeline (py_function/opencv augmentor path preserved)
        tfp = TFDataPipeline(
            dataset=dataset_list,
            target_size=tuple(args.target_size),
            batch_size=args.batch_size,
            augmentor=None,
            normalizer='divide',
            phase='train',
            num_workers=args.num_workers
        )
        ds_tfp = tfp.get_dataset()
        print('Running synthetic benchmark: samples=', args.num_samples, 'batch_size=', args.batch_size, "num_workers=", args.num_workers)

        # pure tf.data from numpy images
        ds_pure = build_pure_tf_dataset_from_images(images, labels, args.batch_size, tuple(args.target_size), phase='train', num_workers=args.num_workers)

        s_pure = bench_tf_dataset(ds_pure, steps=args.steps)
        print(f'Pure tf.data: {s_pure[0]:.3f}s, {s_pure[1]:.1f} samples/s,', "default benchmark percent: 100%")

        s_seq = bench_sequence(seq, steps=args.steps)
        print(f'DataSequencePipeline: {s_seq[0]:.3f}s, {s_seq[1]:.1f} samples/s,', f"percent of pure tf.data: {(s_seq[1]/s_pure[1])*100:.1f}%")

        s_tfp = bench_tf_dataset(ds_tfp, steps=args.steps)
        print(f'TFDataPipeline (py_function): {s_tfp[0]:.3f}s, {s_tfp[1]:.1f} samples/s,', f"percent of pure tf.data: {(s_tfp[1]/s_pure[1])*100:.1f}%")

    else:
        print('Real data mode not implemented in this benchmark script. Use synthetic mode or adapt script to your dataset.')


if __name__ == '__main__':
    main()
