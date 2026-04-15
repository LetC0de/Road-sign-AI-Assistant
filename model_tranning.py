# Import Libraries
import os
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# 🔥 Label Map (IMPORTANT)
label_map = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicle > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing vehicle > 3.5 tons'
}

# Ask to Train or Load Model
choice = input("Train new model? (y/n): ").strip().lower()

if choice == 'y':
    # CNN Model
    model = Sequential([
        Input(shape=(64, 64, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(43, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Data
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(
        'archive/TRAIN',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'
    )

    test_set = test_datagen.flow_from_directory(
        'archive/TEST',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'
    )

    # Train
    model.fit(
        training_set,
        steps_per_epoch=len(training_set),
        epochs=10,
        validation_data=test_set,
        validation_steps=len(test_set)
    )

    # Save
    model.save("traffic_sign_model.keras")
    with open("class_labels.json", "w") as f:
        json.dump(training_set.class_indices, f)

    class_indices = training_set.class_indices
    print("Model trained & saved.")

else:
    model = load_model("traffic_sign_model.keras")
    with open("class_labels.json", "r") as f:
        class_indices = json.load(f)

    print("Model loaded.")

# 🔥 Reverse mapping (index → folder)
class_labels = {v: k for k, v in class_indices.items()}

# Predict
img_path = input("Enter image path: ").strip()

img = load_img(img_path, target_size=(64, 64))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

result = model.predict(img_array)

predicted_class = np.argmax(result)
confidence = np.max(result)

# 🔥 Folder → Actual label
predicted_folder = class_labels[predicted_class]
predicted_name = label_map[int(predicted_folder)]

# Final Output
print("Predicted Sign:", predicted_name)
print("Confidence:", round(confidence * 100, 2), "%")