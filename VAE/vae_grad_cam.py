#Importing relevant libraries
import os
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from cv2 import cv2
import pickle
from IPython.display import Image, display
import matplotlib.cm as cm
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras import backend as K
from tensorflow import keras
from PIL import Image
from PIL import ImageDraw 
import imageio
#Creating folder to save anbd load weights
WEIGHTS_FOLDER = './weights_praveen/'
DATA_FOLDER = './data/img_align_celeba/'
RESULT_FOLER = './vae_grad_1/'
if not os.path.exists(RESULT_FOLER):
  os.mkdir(RESULT_FOLER)
if not os.path.exists(WEIGHTS_FOLDER):
  os.makedirs(os.path.join(WEIGHTS_FOLDER,"AE"))
  os.makedirs(os.path.join(WEIGHTS_FOLDER,"VAE"))

#Reading images from Dataset
filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))

#Setting up the iunput image dimension
INPUT_DIM = (128,128,3) # Image dimension
BATCH_SIZE = 256
Z_DIM = 200 # Dimension of the latent vector (z)
data_flow = ImageDataGenerator(rescale=1./255).flow_from_directory(DATA_FOLDER, 
                                                                   target_size = INPUT_DIM[:2],
                                                                   batch_size = BATCH_SIZE,
                                                                   shuffle = True,
                                                                   class_mode = 'input',
                                                                   subset = 'training'
                                                                   )

# Building the VAE Encoder 
def build_vae_encoder(input_dim, output_dim, conv_filters, conv_kernel_size, 
                  conv_strides, use_batch_norm = False, use_dropout = False):
  
  # Clear tensorflow session
  global K
  K.clear_session()
  
  # Number of Conv layers
  n_layers = len(conv_filters)

  # Define model input
  encoder_input = Input(shape = input_dim, name = 'encoder_input')
  x = encoder_input

  # Add convolutional layers
  for i in range(n_layers):
      x = Conv2D(filters = conv_filters[i], 
                  kernel_size = conv_kernel_size[i],
                  strides = conv_strides[i], 
                  padding = 'same',
                  name = 'encoder_conv_' + str(i)
                  )(x)
      if use_batch_norm:
        x = BathcNormalization()(x)
  
      x = LeakyReLU()(x)

      if use_dropout:
        x = Dropout(rate=0.25)(x)

  # Required for reshaping latent vector while building Decoder
  shape_before_flattening = K.int_shape(x)[1:] 
  x = Flatten()(x)
  #Defining mean and Variance needed for VAE
  mean_mu = Dense(output_dim, name = 'mu')(x)
  log_var = Dense(output_dim, name = 'log_var')(x)

  # Defining a function for sampling
  def sampling(args):
    mean_mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mean_mu), mean=0., stddev=1.) 
    return mean_mu + K.exp(log_var/2)*epsilon   
  
  # Using a Keras Lambda Layer to include the sampling function as a layer 
  # in the model
  encoder_output = Lambda(sampling, name='encoder_output')([mean_mu, log_var])

  return encoder_input, encoder_output, mean_mu, log_var, shape_before_flattening, Model(encoder_input, encoder_output)

#Using function to build Eecoder
vae_encoder_input, vae_encoder_output,  mean_mu, log_var, vae_shape_before_flattening, vae_encoder  = build_vae_encoder(input_dim = INPUT_DIM,
                                    output_dim = Z_DIM, 
                                    conv_filters = [32, 64, 64, 64],
                                    conv_kernel_size = [3,3,3,3],
                                    conv_strides = [2,2,2,2])

# Building the VAE Decoder
def build_decoder(input_dim, shape_before_flattening, conv_filters, conv_kernel_size, 
                  conv_strides):

  # Number of Conv layers
  n_layers = len(conv_filters)

  # Define model input
  decoder_input = Input(shape = (input_dim,) , name = 'decoder_input')

  # To get an exact mirror image of the encoder
  x = Dense(np.prod(shape_before_flattening))(decoder_input)
  x = Reshape(shape_before_flattening)(x)

  # Add convolutional layers
  for i in range(n_layers):
      x = Conv2DTranspose(filters = conv_filters[i], 
                  kernel_size = conv_kernel_size[i],
                  strides = conv_strides[i], 
                  padding = 'same',
                  name = 'decoder_conv_' + str(i)
                  )(x)
      
      # Adding a sigmoid layer at the end to restrict the outputs 
      # between 0 and 1
      if i < n_layers - 1:
        x = LeakyReLU()(x)
      else:
        x = Activation('sigmoid')(x)

  # Define model output
  decoder_output = x

  return decoder_input, decoder_output, Model(decoder_input, decoder_output)
#Using function to build Decoder
vae_decoder_input, vae_decoder_output, vae_decoder = build_decoder(input_dim = Z_DIM,
                                        shape_before_flattening = vae_shape_before_flattening,
                                        conv_filters = [64,64,32,3],
                                        conv_kernel_size = [3,3,3,3],
                                        conv_strides = [2,2,2,2]
                                        )
#Creating the VAE model
vae_input = vae_encoder_input
vae_output = vae_decoder(vae_encoder_output)
vae_model = Model(vae_input, vae_output)

#Loading previously saved weights
vae_model.load_weights("Simple_VAE.hdf5")
#Extracting random image for inference
example_batch = next(data_flow)
example_batch = example_batch[0]
example_images = example_batch[:10]
img_array = example_images[0]
#Saving the inpput image at a location 
with open('im_5_data', 'wb') as f:
  pickle.dump(img_array, f)
with open('img_array','rb') as f: img_array = pickle.load(f)
plt.imsave('VAE_Input_Image_5.jpg', img_array)

#Expanding the image dimension to include the batch dimension
img_array = np.expand_dims(img_array, axis=0)

#Add colour, superimpose and save geenrated heatmap
def save_and_display_gradcam(img, heatmap,cam_path="superimposed_img.jpg", alpha=0.4):
    #Craeting blank palette for heatmap
    heatmap = np.uint8(255 * heatmap)
    #getting colour map
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    #Superimposing input image
    superimposed_img = cv2.addWeighted(jet_heatmap, 0.005, img, 0.995, 0)
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    #Saving the final heatmap
    superimposed_img.save(cam_path)

#Generates heatmap using Grad-CAM
def save_individual_heatmaps(i,img_array, model, last_conv_layer_name, encoder_out, heatmap_images, pred_index=None):
  #Extracting Encoder from VAE model 
  grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.get_layer(encoder_out).output]
    )
  # Calculating Gradients
  with tf.GradientTape(persistent=True) as tape:
    last_conv_layer_output, preds = grad_model(img_array)
    y = preds[0][i]
  grads = tape.gradient(y, last_conv_layer_output)
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
  last_conv_layer_output = last_conv_layer_output[0]
  #Calculating heatmap from Gradients
  heatmap = tf.matmul(last_conv_layer_output, pooled_grads[..., tf.newaxis])
  heatmap = tf.squeeze(heatmap)
  heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
  ##Add colour and save geenrated heatmap
  save_and_display_gradcam(img_array.squeeze(), heatmap, cam_path=heatmap_images+"/heatmap_"+str(i)+".jpg")
  return heatmap

#Iterating through all 200 Latent space nodes
for i in range(200):
  #Creating folder to save images and heatmaps after tweaking of Latent space 
  parent_image_folder = "folder_generated_images_5"
  gen_images_folder_path = parent_image_folder+"/fol_gen_image_"+str(i)
  heatmap_images = "heatmap_images_5"
  if not os.path.exists(parent_image_folder):
    os.mkdir(parent_image_folder)
  if not os.path.exists(gen_images_folder_path):
    os.mkdir(gen_images_folder_path)
  if not os.path.exists(heatmap_images):
    os.mkdir(heatmap_images)
  #variable to store names of all generated images, later used to create animation
  filenames = []
  #Tweaking of individual nodes
  for j in range(20):
    #Encoder prediction
    res = vae_encoder.predict(img_array)
    #Modifying latent node
    res[0][i] = res[0][i] + (j*0.05)
    #Sending the modified vector to Decoder for reconstruction
    res = vae_decoder.predict(res)

    #Saving the generated image
    res = res.squeeze()
    res = res*255
    res = res.astype(np.uint8)
    res = Image.fromarray(res)
    draw = ImageDraw.Draw(res)
    draw.rectangle((0, 0, (res.size)[0], 11), outline='red', fill='white')
    #Adding text to generated image
    imText = "Latent: "+str(i)+", Tweak: "+str(j)
    draw.text((0, 0),imText,(0,0,0))
    res.save(gen_images_folder_path+"/gen_image_"+str(j)+".jpeg")
    #Appending generated image name to a list, used for creating animation from all images
    filenames.append(gen_images_folder_path+"/gen_image_"+str(j)+".jpeg")
  
  #Creating animation of all generated images for a particular node
  images = []
  for filename in filenames:
    images.append(imageio.imread(filename))
  imageio.mimsave(gen_images_folder_path+'/latent_space_'+str(i)+'.gif', images, fps=2)
  #Last Encoder layer name for Grad-CAM
  last_conv_layer_name = "encoder_conv_3"
  #Encoder output layer name
  encoder_out = "encoder_output"
  #Craeting heatmap using Grad-CAM and saving it as .jpg file
  h_map = save_individual_heatmaps(i,img_array, vae_model, last_conv_layer_name, encoder_out, heatmap_images)
