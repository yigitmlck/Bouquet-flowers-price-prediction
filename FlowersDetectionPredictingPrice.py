import cv2
import pandas as pd
import numpy as np
from roboflow import Roboflow
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import firebase_admin
from firebase_admin import credentials, db
import datetime
import paho.mqtt.client as mqtt


# Configure Firebase Admin SDK
cred = credentials.Certificate("your firebase's j.son file path")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'your database url'
})

def publish_to_mqtt(message):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connection successful")
        else:
            print("Connection error, kod:", rc)

    def on_publish(client, userdata, mid):
        print("Message published")

    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_publish = on_publish

    try:
        # Connection to HiveMQ public broker
        mqtt_client.connect("broker.hivemq.com", 1883, 60)
        mqtt_client.loop_start()

        # Publish Message
        topic = "Prefered Topic"
        result = mqtt_client.publish(topic, str(message))  # Send message in string format
        result.wait_for_publish()  # Wait until the publish is complete

        mqtt_client.loop_stop()  # Stop the loop
        mqtt_client.disconnect()
        print("MQTT message sent:", message)
    except Exception as e:
        print("Error sending MQTT message:", e)

# Roboflow API and other required libraries are considered installed

# Describe the model

rf = Roboflow(api_key="AOYdXKLeZqaFh74QaORw")
project = rf.workspace().project("annot-0abet")
model_rf = project.version(1).model

# Open the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera can not founded!")
    exit()

print("Camera opened!")

# A list of all flower names
flowers = ["Rose", "sunflower", "champaka", "chitrak", "Common Lanthana", "Hibiscus", "honeysuckle", "indian mallow", "Jatropha", "malabar melastome", "Marigold", "shankupushpam", "spider lily"]

# Create an empty array to store flower numbers
flower_counts_array = []

# Load dataset
data = pd.read_csv("Bouquets")

# Separate feature and target variables
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split into training and test datasets %20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating a CNN model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Optimizing the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Saving data function in Firebase
def save_to_firebase(flower_counts, price_prediction):
    ref = db.reference('flower_predictions')
    timestamp = datetime.datetime.now().isoformat()
    data = {
        'timestamp': timestamp,
        'flower_counts': flower_counts,
        'predicted_price': float(price_prediction)
    }
    ref.push(data)

# A waiting loop when the camera is opened
while True:
    # Take a photo frame from the camera
    ret, frame = camera.read()

    # Show camera on the screen
    cv2.imshow("Camera", frame)

    # Capture a photo when the 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Detect using the model
        result = model_rf.predict(frame, confidence=10, overlap=80).json()

        # If there is an object detected
        if "predictions" in result:
            # Get labels of detected objects
            labels = [item["class"] for item in result["predictions"]]

            # Calculate the number of times all flowers cross and add only the numbers to the array
            flower_counts = [len(re.findall(rf'\b{flower}\b', ' '.join(labels))) for flower in flowers]

            # Add the calculated flower counts to the array
            flower_counts_array.append(flower_counts)

            # Show flower counts in console
            print("Flower Counts:", flower_counts)

            # Convert flower counts to numpy array and scale
            flower_counts_np = np.array(flower_counts).reshape(1, -1)
            flower_counts_np = sc.transform(flower_counts_np)

            # Make a price predict
            price_prediction = model.predict(flower_counts_np)
            predicted_price = price_prediction[0][0]
            print("Predicted Price:", predicted_price)

            # Save to firebase
            save_to_firebase(flower_counts, predicted_price)

            # Share with MQTT
            publish_to_mqtt(predicted_price)

    # End loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
camera.release()
cv2.destroyAllWindows()

# Print the resulting flower numbers as an array
print("All Flowers Counts:", flower_counts_array)
