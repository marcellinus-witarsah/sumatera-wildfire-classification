from json import load
from multiprocessing import allow_connection_pickling
from unicodedata import name
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

MODEL_PATH = "./models/landsat_8_b7_b5_b2/best_unet_model_opt_adam_lr_0.001_batch_16_epochs_20_filters_32_size_176_date_20220925"
SCALE = 0.5

def dice_coef():
    """
    :return: constant of 1
    """
    return 1

@st.cache(allow_output_mutation=True)
def load_model():
    """
    :return: return a U-Net Model
    """
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={'dice_coef': dice_coef})

def predict(image_input, model, size):
    """
    :image_input: image data input for prediction
    :model: model being used for prediction 
    :size: use to resize the image_input 
    :return: return a prediction result
    """
    image = ImageOps.fit(image_input, size, Image.ANTIALIAS)
    st.image(image)
    img = np.asarray(image)
    image_reshape = img[np.newaxis,...]
    res = model.predict(image_reshape)

    return res

def main():    
    # load model
    img_type = "Landsat" if "landsat" in MODEL_PATH else "Sentinel"
    unet_model = load_model()

    # Returns pretty much every information about your model
    config = unet_model.get_config()
    _, height, width, channel = config["layers"][0]["config"]["batch_input_shape"]

    # Title of the App
    st.title("Sumatra Wildfire Detection App")
    st.write("This web application is a final product of U-Net model trained using {} satellite images".format(img_type))


    # File uploader for Image
    uploaded_file = st.file_uploader("Please Input Wildfire Image", type="png")

    # Display Image
    if uploaded_file:
        # read file as byte
        image = Image.open(uploaded_file)
        img_height, img_width, channel = np.asarray(image).shape

        # prepare image for display
        image_for_display = image.resize((int(img_height * SCALE), int(img_width * SCALE)))

        # display image
        st.image(image_for_display, channels="RGB", caption="Uploaded file")

        # Predict Button
        generate_prediction = st.button("Generate Prediction")

        # Prediction Result
        if generate_prediction:
            result = predict(
                image_input=image, 
                model=unet_model,
                size=(height, width) 
            )

            # apply threshold to the result prediction
            np_mask = np.array(result[0])
            np_mask = np.where(np_mask < 0.5, 0, 1)
            np_mask = np_mask * 255
            np_mask = np_mask.astype(np.uint8)
            st.write(np_mask.shape)
            
            # return an array of unique value along with respectives amount
            unique, counts = np.unique(np_mask, return_counts=True)
            
            # turn into dictionary => {0: ..., 255: ...}
            unique_count_dict = dict(zip(unique, counts))
            percentage_burned = unique_count_dict[255]/(unique_count_dict[0]+unique_count_dict[255])
            
            # display the mask and the caption
            st.image(np_mask, width=400, caption='{:.2%} of Land Burned'.format(percentage_burned))

if __name__ == "__main__":
    main()





