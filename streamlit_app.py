import streamlit as st
import tensorflow as tf
import numpy as np
import cv2


def dice_coef(y_true, y_pred):
    """
    :param y_true: tensors contain ground truth values
    :param y_pred: tensors contain predicted values
    :return: dice coefficient value
    """
    return 1


# load model
model_path = "models/sentinel_2/unet_model_opt_adam_lr_0.001_batch_32_epochs_20_filters_16_size_176_date_20220810"
unet_model = tf.keras.models.load_model(model_path, custom_objects={'dice_coef': dice_coef})

# Returns pretty much every information about your model
config = unet_model.get_config()
_, height, width, channel = config["layers"][0]["config"]["batch_input_shape"]

# Title of the App
st.title("Sumatra Wildfire Detection App")
st.write("This web application is a final product of U-Net model trained using satellite images")


# File uploader for Image
uploaded_file = st.file_uploader("Please Input Wildfire Image", type="png")

# Display Image
if uploaded_file:
    # read file as byte
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # decode file
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # prepare image for display
    scale_percentage = 0.6
    scaled_height = int(image.shape[0]*scale_percentage)
    scaled_width = int(image.shape[1]*scale_percentage)
    image_for_display = cv2.resize(image, (scaled_height, scaled_width))

    # display image
    st.image(image_for_display * 3, channels="RGB", caption="Uploaded file")

    # prepare image for input
    resized_image = cv2.resize(image, (height, width))
    normalized_image = tf.cast(resized_image / 255, tf.dtypes.float32)
    input_data = tf.expand_dims(normalized_image, 0)

    # Predict Button
    generate_prediction = st.button("Generate Prediction")

    # Prediction Result
    if generate_prediction:
        result = unet_model.predict(input_data)

        # apply threshold to the result prediction
        np_mask = np.array(result[0])
        np_mask = np.where(np_mask < 0.5, 0, 1)
        np_mask = np_mask * 255
        np_mask = np_mask.astype(np.uint8)

        # prepare mask for display
        mask_for_display = cv2.resize(np_mask, (scaled_height, scaled_width), interpolation=cv2.INTER_AREA)
        st.image(mask_for_display, caption="Prediction Result")




