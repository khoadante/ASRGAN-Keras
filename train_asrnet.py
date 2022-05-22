import argparse
import tensorflow as tf
import time
import os
from glob import glob
from utils.train import no_gan_inner_step, AverageModelWeights

from keras.optimizer_v2.adam import Adam

from networks.models import RRDBNet
from a_networks.a_models import ARRDBNet

from utils.data import load_function, feed_data, PoolData
from utils.data import feed_props_1, feed_props_2
from utils.data import usm_sharpener

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=32, help='Number batches')
parser.add_argument('--restore_checkpoint', default=False, action='store_true', help='Use training checkpoints to continue training the model')
parser.add_argument('--save_gan_weights', default=False, action='store_true', help='Save the generator weights')
parser.add_argument('--save_ema_model', default=False, action='store_true', help='Save the generator ema model')
parser.add_argument('--use_anet', default=False, action='store_true', help='Use the A network (may crash)')

args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = 2e-4

BETA_1 = 0.9
BETA_2 = 0.99

POOL_SIZE = BATCH_SIZE * 5
RESTORE_CHECKPOINT = args.restore_checkpoint
SAVE_GAN_WEIGHTS = args.save_gan_weights
SAVE_EMA_MODEL = args.save_ema_model

USE_ANET = args.use_anet

optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)

no_gan_model = ARRDBNet() if USE_ANET else RRDBNet()
no_gan_model.build((None, 256, 256, 3))

ema_no_gan_model = ARRDBNet() if USE_ANET else RRDBNet()
ema_no_gan_model.build((None, 256, 256, 3))

ema_api = AverageModelWeights(ema_no_gan_model, no_gan_model.get_weights())

data_path = os.path.abspath("./DIV2K_train_HR/*.png")
train_images_paths = sorted(glob(data_path))

train_dataset = tf.data.Dataset.from_tensor_slices((train_images_paths))
train_dataset = train_dataset.shuffle(len(train_images_paths))
train_dataset = train_dataset.map(
    load_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_generator = train_dataset.batch(BATCH_SIZE)

pool_train_data = PoolData(POOL_SIZE, BATCH_SIZE)

checkpoint_dir = './training_checkpoints'
ema_checkpoint_dir = './ema_training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=no_gan_model, optimizer=optimizer)

ema_checkpoint_prefix = os.path.join(ema_checkpoint_dir, "ema_ckpt")
ema_checkpoint = tf.train.Checkpoint(model=ema_no_gan_model)

if RESTORE_CHECKPOINT:
    print("loading training checkpoints: ")
    print(tf.train.latest_checkpoint(checkpoint_dir))
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    print("loading EMA training checkpoints: ")
    print(tf.train.latest_checkpoint(ema_checkpoint_dir))
    ema_checkpoint.restore(tf.train.latest_checkpoint(ema_checkpoint_dir))

@tf.function
def train_step(gt_images, lq_images):
    return no_gan_inner_step(gt_images, lq_images, no_gan_model, optimizer)

epochs = EPOCHS
start_epoch = 0

train_steps = int(len(train_images_paths) // BATCH_SIZE)

pool_train_data = PoolData(POOL_SIZE, BATCH_SIZE)

train_loss_metric = tf.keras.metrics.Mean()
loss_results = []

def train(epochs):
    print("Start Training")
    for epoch in range(start_epoch, epochs):
        train_loss_metric.reset_states()
        epoch_time = time.time()
        batch_time = time.time()
        step = 0

        epoch_count = f"0{epoch + 1}/{epochs}" if epoch < 9 else f"{epoch + 1}/{epochs}"

        for img, first_kernel, second_kernel, sinc_kernel in train_generator:
            gt_img, lq_img = feed_data(img, first_kernel, second_kernel, sinc_kernel, [
                                       feed_props_1, feed_props_2])
            gt_img, lq_img = pool_train_data.get_pool_data(gt_img, lq_img)
            gt_img = usm_sharpener.sharp(gt_img)
            loss = train_step(gt_img, lq_img)

            print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
                  '| Loss:', f"{loss:.5f}", "| Step Time:", f"{time.time() - batch_time:.2f}", end='')

            train_loss_metric.update_state(loss)
            loss = train_loss_metric.result().numpy()
            step += 1

            loss_results.append(loss)

            batch_time = time.time()

        checkpoint.save(file_prefix=checkpoint_prefix)
        ema_api.compute_ema_weights(no_gan_model)
        ema_checkpoint.save(file_prefix=ema_checkpoint_prefix)

        print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
              '| Loss:', f"{loss:.5f}", "| Epoch Time:", f"{time.time() - epoch_time:.2f}")

train(epochs)

if SAVE_GAN_WEIGHTS:
    no_gan_model.save_weights('./checkpoint_weights/last_weights')

if SAVE_EMA_MODEL:
    ema_no_gan_model.save("./no_gan_ema_model")
