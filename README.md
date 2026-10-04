# cat-and-dog-cnn-
# Cat vs Dog Classifier

A web app that tells you whether an image is a cat or a dog. Upload a photo and it predicts which one it is.

## How it works

It's built on a convolutional neural network using transfer learning. I used MobileNetV2 (pretrained on ImageNet) as the base, froze its layers, and added my own classification head on top to tell cats and dogs apart. The model was trained on about 25,000 cat and dog images.

Since MobileNetV2 already learned to recognize general visual features (edges, shapes, textures) from millions of images, I only needed to train the final layers on the cat/dog data, which made training fast and the results accurate.

## Dataset

Cat and dog images from Kaggle (the classic 25k cats and dogs dataset). The dataset had a few corrupt images that I had to filter out before training.

## Tech stack

Python, TensorFlow/Keras, Streamlit. Trained on Google Colab (GPU), deployed with Streamlit.

## Running it

pip install -r requirements.txt
streamlit run app.py


Then upload an image and it'll predict cat or dog.

## Notes

The trickiest part was making sure images are preprocessed the exact same way during training and prediction. MobileNetV2 expects pixel values scaled a specific way, and mismatching that between training and inference gives completely wrong results.

## Files
The model reaches around 97% accuracy on validation data.

- `app.py` — the Streamlit app
- `cat_dog_training.ipynb` — the training notebook (Colab)
- `cat_dog_model.keras` — the trained model
