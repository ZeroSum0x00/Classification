import tensorflow as tf



class PatchConv2DWithResampleWeights(tf.keras.layers.Conv2D):
    """
      Resample the weights of the patch embedding kernel to target resolution.
    
      We resample the patch embedding kernel by approximately inverting the effect
      of patch resizing. Colab with detailed explanation:
      (internal link)
      With this resizing, we can for example load a B/8 filter into a B/16 model
      and, on 2x larger input image, the result will match.
      
      # From FlexiViT https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/flexi/vit.py#L30
      # Paper [PDF 2212.08013 FlexiViT: One Model for All Patch Sizes](https://arxiv.org/pdf/2212.08013.pdf)
    """
    def load_resized_weights(self, source_layer, method="bilinear"):
        import numpy as np

        if isinstance(source_layer, dict):
            source_kernel, source_bias = list(source_layer.values())  # weights
        else:
            source_kernel, source_bias = source_layer.get_weights()  # layer

        # channels_last source_kernel shape `[patch_size, patch_size, in_channel, out_channel]`
        # channels_first source_kernel shape `[out_channel, in_channel, patch_size, patch_size]`
        source_kernel, source_bias = np.array(source_kernel).astype("float32"), np.array(source_bias).astype("float32")
        source_shape, target_shape = source_kernel.shape[:2], self.kernel_size  # source_kernel is from h5, must be channels_last format

        # get_resize_mat(old_shape, target_shape)
        # NOTE: we are using tf.image.resize here to match the resize operations in
        # the data preprocessing pipeline.
        mat = []
        for idx in range(source_shape[0] * source_shape[1]):
            basis_vec = np.zeros(source_shape).astype("float32")
            basis_vec[np.unravel_index(idx, source_shape)] = 1.0
            vec = np.expand_dims(basis_vec, axis=-1)
            vec = tf.image.resize(vec, target_shape, method=method)
            vec = vec.numpy()
            mat.append(vec.reshape(-1))
            
        resize_mat_pinv = np.linalg.pinv(np.stack(mat))

        # v_resample_kernel = jax.vmap(jax.vmap(lambda kernel: (resize_mat_pinv @ kernel.reshape(-1)).reshape(new_hw), 2, 2), 3, 3)
        # cc = v_resample_kernel(old)
        # As it's only one weight, just using two loop here, instead of `jax.vmap`
        target_weights = np.stack([
            [(resize_mat_pinv @ jj.reshape(-1)).reshape(target_shape) for jj in ii]
            for ii in source_kernel.transpose([3, 2, 0, 1])
        ])
        
        target_weights = target_weights.transpose([2, 3, 1, 0])
        self.set_weights([target_weights, source_bias])
        