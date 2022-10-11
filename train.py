import os
import tensorflow as tf
from tensorflow import keras
import models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dataloader
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-epoch','--epoch', type=int)
parser.add_argument('-batch_size','--batch_size', type=int)
parser.add_argument('-save_path', '--save_path', type=str)
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epoch

# Gen, Disc, Extractor 정의
modules = models.Modules()

generator = modules.generator()
discriminator = modules.disriminator()
vgg_extractor = modules.vgg_extractor()


# DACON DATA 기준 ['LR', 'HR']
train_df = pd.read_csv('c:/tasks/data/train.csv')
train_lrs = train_df.LR.to_list()
train_hrs = train_df.HR.to_list()



bce = keras.losses.BinaryCrossentropy(from_logits = False)
mse = keras.losses.MeanSquaredError()
gene_opt = keras.optimizers.Adam()
disc_opt = keras.optimizers.Adam()
gene_losses = tf.keras.metrics.Mean()
disc_losses = tf.keras.metrics.Mean()


def get_gene_loss(fake_out):
    return bce(tf.ones_like(fake_out), fake_out)

def get_disc_loss(real_out, fake_out):
    return bce(tf.ones_like(real_out), real_out) + bce(tf.zeros_like(fake_out), fake_out)


@tf.function
def get_content_loss(hr_real, hr_fake):

    hr_real_feature = vgg_extractor(hr_real)
    hr_fake_feature = vgg_extractor(hr_fake)

    return mse(hr_real_feature, hr_fake_feature)


@tf.function
def step(lr, hr_real):
    with tf.GradientTape() as gene_tape , tf.GradientTape() as disc_tape:
        hr_fake = generator(lr, training=True) # training을 위한 lr 이미지를 를 넣어 hr_real과 비교할 fake 이미지 생성

        real_out = discriminator(hr_real, training=True) # hr 고화질 이미지를 넣어 disciriminator로 판별 한 output값
        fake_out = discriminator(hr_fake, training=True) # Generator를 통해 생성된 hr_fake 이미지를 discriminator로 판별한 output 값
        perceptual_loss = get_content_loss(hr_real, hr_fake) + 1e-3 * get_gene_loss(fake_out)
        # vgg를 통과 한 실제 이미지와, 생성 이미지간의 mse 값 + discriminator로 판별한 fake_out의 sigmoid output 값 + 0.001
        
        discriminator_loss = get_disc_loss(real_out, fake_out)
        # discriminaotr output 값 real_out은 모두 1인 동일 shape 텐서와의 비교값이며 fake_out 은 모두 0인 텐서와의 비교값 => BinaryCrossEntropy

        # Define gradient
        gene_gradient = gene_tape.gradient(perceptual_loss, generator.trainable_variables)
        disc_gradient = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)

        gene_opt.apply_gradients(zip(gene_gradient, generator.trainable_variables))
        disc_opt.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))

        return perceptual_loss, discriminator_loss


# loss 
for epoch in range(1,50):
    images = dataloader.Dataloader(train_lrs, train_hrs, count=1)
    print(f'#################\nEPOCH : {epoch}\n#################')
    
    try:
        for idx in range(10000):
            lr, hr = images.cut_image()
            for i in range(int(len(lr)/batch_size -1)):
                g_loss, d_loss = step(lr[batch_size * i : batch_size * (i+1)], hr[batch_size * i : batch_size * (i+1)])

                gene_losses.update_state(g_loss)
                disc_losses.update_state(d_loss)

        
    except Exception as e:
        print(e)
            
    print(f'Epoch : {epoch}, Step : {idx+1} \nGen_loss : {gene_losses.result():.4f} \nDisc_loss : {disc_losses.result():.4f}', end='\n')
    
    gene_losses.reset_states()
    disc_losses.reset_states()
    generator.save(f'{args.save_path}/srgan_{epoch}.h5')