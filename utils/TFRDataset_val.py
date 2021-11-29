import numpy as np
import tensorflow as tf
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

def read_tfrecord(example):
    features = {
        'label': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = example['image']
    image = tf.io.decode_raw(image, tf.uint8)
    # image = tf.reshape(image, [64, 128, 128, 3])
    # Downsized dataset
    image = tf.reshape(image, [64, 32, 32, 3])
    # image = tf.reshape(image, [128, 16, 16, 3])
    # image = tf.reshape(image, [64, 128, 128, 3])

    label  = example['label']
    # height = example['height']
    # width  = example['width']

    return image, label #, height, width
    

def tfr_data_loader_val(data_dir="", batch_size=32, drop_remainder=True):
    '''
    Function that takes path to tfrecord files (allows regular expressions), 
    and returns a tensorflow dataset that can be iterated upon, 
    using loops or enumerate()
    '''

    if data_dir is None:
        raise ValueError("Missing path to data directory!")
    else:
        data_dir=data_dir
    
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.io.gfile.glob(data_dir)
    dataset = tf.data.TFRecordDataset(dataset, compression_type='GZIP') # , cycle_length=batch_size, num_parallel_calls=8)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.shuffle(10)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset




# start=time.time()
# for x in a:
#     tt=torch.from_numpy(x[0].numpy())
#     tt=tt.permute(0,4,1,2,3)
#     print(tt.shape)
# print(time.time()-start)
# np.array(list(map(ord, p[1][1].numpy())))
# np.vectorize(ord)(p[1][1].numpy())