# ─── 1. SETUP & IMPORTS ─────────────────────────────────────────────────────────
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Reshape, Conv2D, UpSampling2D, LeakyReLU, Dropout,
    Flatten, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import Callback

# ─── 2. TPU INIT & MIXED PRECISION ───────────────────────────────────────────────
if not hasattr(tf, "_tpu_initialized"):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    tf._tpu_initialized = True
    print("🔥 TPU initialized")
else:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    print("⚠️ TPU already initialized")

strategy = tf.distribute.TPUStrategy(resolver)
tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

# ─── 3. DATA LOADING & PIPELINE ─────────────────────────────────────────────────
print("Loading & preprocessing Fashion MNIST…")
(x_train, _), _ = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 127.5 - 1.0   # [0,255]→[-1,1]
x_train = x_train[..., None]

BUFFER = 60000
GLOBAL_BATCH_SIZE = 256 * strategy.num_replicas_in_sync  # 256×8=2048

dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
      .shuffle(BUFFER)
      .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
      .prefetch(tf.data.AUTOTUNE)
)

# ─── 4. MODEL DEFINITIONS ─────────────────────────────────────────────────────────
LATENT_DIM = 256

def build_generator():
    return Sequential([
        Dense(7 * 7 * 640, input_shape=(LATENT_DIM,)),
        BatchNormalization(),
        LeakyReLU(0.2),
        Reshape((7, 7, 640)),
        UpSampling2D(),
        Conv2D(320, 5, padding="same"),
        BatchNormalization(),
        LeakyReLU(0.2),
        UpSampling2D(),
        Conv2D(160, 5, padding="same"),
        BatchNormalization(),
        LeakyReLU(0.2),
        Conv2D(1, 5, padding="same", activation="tanh"),
    ], name="generator")

def build_discriminator():
    return Sequential([
        Conv2D(160, 5, strides=2, padding="same", input_shape=(28,28,1)),
        LeakyReLU(0.2),
        Dropout(0.3),
        Conv2D(320, 5, strides=2, padding="same"),
        LeakyReLU(0.2),
        Dropout(0.3),
        Flatten(),
        Dense(1, activation="sigmoid"),
    ], name="discriminator")

class FashionGAN(Model):
    def __init__(self, G, D, latent_dim=LATENT_DIM):
        super().__init__()
        self.G = G
        self.D = D
        self.latent_dim = latent_dim

    def build(self, input_shape):
        self.G.build((None, self.latent_dim))
        self.D.build((None, input_shape[1], input_shape[2], input_shape[3]))
        super().build(input_shape)

    def call(self, inputs, training=False):
        return self.D(inputs, training=training)

    def compile(self, g_opt, d_opt, loss_fn):
        super().compile()
        self.g_opt, self.d_opt, self.loss_fn = g_opt, d_opt, loss_fn

    def train_step(self, real):
        batch = tf.shape(real)[0]
        real_labels = tf.ones((batch,1)) * 0.9
        fake_labels = tf.zeros((batch,1))

        # Discriminator
        noise = tf.random.normal((batch, self.latent_dim))
        with tf.GradientTape() as td:
            fake = self.G(noise, training=True)
            d_real = self.D(real, training=True)
            d_fake = self.D(fake, training=True)
            d_loss = 0.5 * (
                self.loss_fn(real_labels, d_real) +
                self.loss_fn(fake_labels, d_fake)
            )
        grads_d = td.gradient(d_loss, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grads_d, self.D.trainable_variables))

        # Generator
        noise = tf.random.normal((batch, self.latent_dim))
        misleading = tf.ones((batch,1))
        with tf.GradientTape() as tg:
            gen = self.G(noise, training=True)
            g_loss = self.loss_fn(misleading, self.D(gen, training=True))
        grads_g = tg.gradient(g_loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grads_g, self.G.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

class ModelMonitor(Callback):
    def __init__(self, num_img=4, latent_dim=LATENT_DIM, save_freq=5):
        super().__init__()
        self.num_img, self.latent_dim, self.save_freq = num_img, latent_dim, save_freq
        os.makedirs("generated_images", exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            z = tf.random.normal((self.num_img**2, self.latent_dim))
            imgs = (self.model.G(z) * 127.5 + 127.5).numpy().astype("uint8")
            fig, axes = plt.subplots(self.num_img, self.num_img, figsize=(6,6))
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(imgs[i, :, :, 0], cmap="gray")
                ax.axis("off")
            plt.tight_layout()
            path = f"generated_images/epoch_{epoch+1}.png"
            plt.savefig(path)
            plt.close(fig)
            print(f"Saved {path}")

# ─── 5. BUILD, COMPILE & TRAIN UNDER TPU STRATEGY ───────────────────────────────
with strategy.scope():
    G = build_generator()
    D = build_discriminator()
    g_opt = Adam(2e-4, beta_1=0.5)
    d_opt = Adam(2e-4, beta_1=0.5)
    bce   = BinaryCrossentropy()
    gan = FashionGAN(G, D)
    gan.compile(g_opt, d_opt, bce)

print("🚀 Training on TPU…")
history = gan.fit(
    dataset,
    epochs=30,  # a few more epochs for better fidelity
    callbacks=[ModelMonitor(num_img=4, latent_dim=LATENT_DIM, save_freq=5)],
)

# ─── 6. VISUALIZE & SAVE LOSSES ─────────────────────────────────────────────────
plt.figure()
plt.plot(history.history["d_loss"], label="Discriminator")
plt.plot(history.history["g_loss"], label="Generator")
plt.title("Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_loss.png")
plt.show()

# ─── 7. FINAL GENERATED IMAGES & SAVE ───────────────────────────────────────────
final_z = tf.random.normal((16, LATENT_DIM))
final_imgs = (G(final_z) * 127.5 + 127.5).numpy().astype("uint8")
fig, axes = plt.subplots(4, 4, figsize=(8,8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(final_imgs[i, :, :, 0], cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.savefig("final_generated_images.png")
plt.show()
