import gradio as gr
import tensorflow as tf
#from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("indian_traffic_sign_cnn.h5")

# Define your class names (use the same ones you used during training)
class_names = ['ALL_MOTOR_VEHICLE_PROHIBITED',
               'AXLE_LOAD_LIMIT',
               'BARRIER_AHEAD',
               'BULLOCK_AND_HANDCART_PROHIBITED',
               'BULLOCK_PROHIBITED',
               'CATTLE',
               'COMPULSARY_AHEAD',
               'COMPULSARY_AHEAD_OR_TURN_LEFT',
               'COMPULSARY_AHEAD_OR_TURN_RIGHT',
               'COMPULSARY_CYCLE_TRACK',
               'COMPULSARY_KEEP_LEFT',
               'COMPULSARY_KEEP_RIGHT',
               'COMPULSARY_MINIMUM_SPEED',
               'COMPULSARY_SOUND_HORN',
               'COMPULSARY_TURN_LEFT',
               'COMPULSARY_TURN_LEFT_AHEAD',
               'COMPULSARY_TURN_RIGHT',
               'COMPULSARY_TURN_RIGHT_AHEAD',
               'CROSS_ROAD',
               'CYCLE_CROSSING',
               'CYCLE_PROHIBITED',
               'DANGEROUS_DIP',
               'DIRECTION',
               'FALLING_ROCKS',
               'FERRY',
               'GAP_IN_MEDIAN',
               'GIVE_WAY',
               'GUARDED_LEVEL_CROSSING',
               'HANDCART_PROHIBITED',
               'HEIGHT_LIMIT',
               'HORN_PROHIBITED',
               'HUMP_OR_ROUGH_ROAD',
               'LEFT_HAIR_PIN_BEND',
               'LEFT_HAND_CURVE',
               'LEFT_REVERSE_BEND',
               'LEFT_TURN_PROHIBITED',
               'LENGTH_LIMIT',
               'LOAD_LIMIT',
               'LOOSE_GRAVEL',
               'MEN_AT_WORK',
               'NARROW_BRIDGE',
               'NARROW_ROAD_AHEAD',
               'NO_ENTRY',
               'NO_PARKING',
               'NO_STOPPING_OR_STANDING',
               'OVERTAKING_PROHIBITED',
               'PASS_EITHER_SIDE',
               'PEDESTRIAN_CROSSING',
               'PEDESTRIAN_PROHIBITED',
               'PRIORITY_FOR_ONCOMING_VEHICLES',
               'QUAY_SIDE_OR_RIVER_BANK',
               'RESTRICTION_ENDS',
               'RIGHT_HAIR_PIN_BEND',
               'RIGHT_HAND_CURVE',
               'RIGHT_REVERSE_BEND',
               'RIGHT_TURN_PROHIBITED',
               'ROAD_WIDENS_AHEAD',
               'ROUNDABOUT',
               'SCHOOL_AHEAD',
               'SIDE_ROAD_LEFT',
               'SIDE_ROAD_RIGHT',
               'SLIPPERY_ROAD',
               'SPEED_LIMIT_15',
               'SPEED_LIMIT_20',
               'SPEED_LIMIT_30',
               'SPEED_LIMIT_40',
               'SPEED_LIMIT_5',
               'SPEED_LIMIT_50',
               'SPEED_LIMIT_60',
               'SPEED_LIMIT_70',
               'SPEED_LIMIT_80',
               'STAGGERED_INTERSECTION',
               'STEEP_ASCENT',
               'STEEP_DESCENT',
               'STOP',
               'STRAIGHT_PROHIBITED',
               'TONGA_PROHIBITED',
               'TRAFFIC_SIGNAL',
               'TRUCK_PROHIBITED',
               'TURN_RIGHT',
               'T_INTERSECTION',
               'UNGUARDED_LEVEL_CROSSING',
               'U_TURN_PROHIBITED',
               'WIDTH_LIMIT',
               'Y_INTERSECTION']

# Define the prediction function
def classify_image(image):
    # Preprocess image
    image = image.convert("RGB")
    image = tf.image.convert_image_dtype(tf.constant(np.array(image)), tf.float32)
    image = tf.image.resize(image, (64, 64))
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(image)
    score = tf.nn.softmax(predictions[0])

    predicted_class_index = np.argmax(score)
    predicted_class_name = class_names[predicted_class_index]
    #confidence = float(np.max(score))

    return predicted_class_name

# Create Gradio interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload a Traffic Sign"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="Traffic Sign Classifier",
    description="Upload an image of a traffic sign to classify it."
)

# Launch the app
demo.launch(share=True)