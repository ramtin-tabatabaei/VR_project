import numpy as np
import pandas as pd  # Import pandas
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model




#Parameters
sequence = 30
delta = 1
K = 2
K_max = 10
epoch = 1
early = 10
lstm1 = 64
lstm2 = 64
resolution = 0.1
t0 = 5
tf = 620.8



directory = r"/Users/stabatabaeim/Downloads/Data"  
txt_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
print("Number of txt files:", len(txt_files))

#To make the size of all txt data files similar to each other I choose a big number like 50000
data_array = np.zeros((len(txt_files), 50000, 4), dtype=np.float32) 

for i, file_path in enumerate(txt_files):
    try:
        # Use pandas to read the file. It automatically handles different delimiters.
        df = pd.read_csv(file_path, delimiter=None, header=None, engine='python')
        
        # Replace any non-numeric values with NaN, then fill with 0s
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        data = df.values.astype(np.float32)  # Convert DataFrame to a numpy array
        
        # Determine the number of rows to fill in the pre-allocated array
        num_rows = data.shape[0]
        
        # Fill the corresponding slice of the pre-allocated array with data
        data_array[i, :num_rows, :] = data[:num_rows, :]

    except Exception as e:  # Catch any error
        print(f"Error reading file {file_path}: {e}")

# You can print the array shape or specific elements to verify
# print(data_array.shape)
        
# Additional code to load Excel file and print selected properties
excel_file_path = r"/Users/stabatabaeim/Downloads/Properties.xlsx"
df_properties = pd.read_excel(excel_file_path, sheet_name='Sheet1')

# Selecting values from row 3 onwards (df.values[1:] because of zero-based indexing)
properties = df_properties.values[1:]


# Convert 'Properties' to a numpy array if it isn't one already
Properties = np.array(properties, dtype='object')  # Use 'object' dtype to accommodate mixed types

# Vectorized replacement of "N" with 1 and "F" with 2
Properties[Properties == "N"] = 1
Properties[Properties == "F"] = 2
# Convert to float
Properties = Properties.astype(float)


# # Calculate the dimension based on the new resolution
Dim = round((tf - t0) / resolution)
# # Initialize the resampled Properties array
Properties_res = np.zeros((Dim, Properties.shape[1]))
# # Initialize the resampled Data array
Data_res = np.zeros((len(txt_files), Dim, data_array.shape[2]))

t = np.arange(t0, tf, resolution)  # Generate time steps based on resolution

# # Initialize indices for interpolation
Data_inter = np.zeros(len(txt_files), dtype=int)
Properties_inter = 0

# # Pre-compute indices for Data array where time >= t0
for j in range(len(txt_files)):
    Data_inter[j] = np.argmax(data_array[j, :, 0] >= t0) - 1
print(Data_inter)

# Pre-calculate all time points based on the given resolution.
time_points = np.arange(t0, tf, resolution)  # Ensure tf is included

# Loop through each time point to update Properties_res and Data_res
for i, t in enumerate(time_points):
    # Update Properties for the current time step
    Properties_res[i] = Properties[Properties_inter]
    Properties_res[i, 0] = t  # Set the current time
    
    # Check if it's time to move to the next property based on the time condition
    if t >= Properties[Properties_inter + 1, 0] and Properties_inter + 1 < len(Properties):
        Properties_inter += 1
    
    # Update Data_res for each file based on the current time step
    for j in range(len(txt_files)):
        # Skip if the next data point is zero (assuming data is zero-padded)
        if data_array[j, Data_inter[j] + 2, 0] == 0:
            continue
        
        # Assign data from the current index
        Data_res[j, i] = data_array[j, Data_inter[j]]
        Data_res[j, i, 0] = t  # Set the current time
        
        # Increment Data_inter if the next data point's time is passed
        while Data_inter[j] + 1 < data_array[j].shape[0] and t >= data_array[j, Data_inter[j] + 1, 0]:
            Data_inter[j] += 1


# Initialize list to track changes in situation
Sit_change = [-1]

# Delete angle columns from the resampled Properties
array_without_angle_col = np.delete(Properties_res, [0, 6, 12, 18], axis=1)

# Detect changes in situation, excluding the angle columns
for i in range(Properties_res.shape[0] - 1):
    if not np.array_equal(array_without_angle_col[i + 1], array_without_angle_col[i]):
        Sit_change.append(i)

# Append the last time frame, adjusted for resolution
Sit_change.append(tf / resolution)
Sit_change = np.array(Sit_change)

# Calculate target number for determining Start and Stop
# target_number = int(K / resolution * tf / K_max)
target_number = int(K / K_max * Properties_res.shape[0])
nearest_index = np.abs(Sit_change - target_number).argmin()
Stop_K = int(Sit_change[nearest_index])
Stop_t = Properties_res[Stop_K, 0]

target_number = int((K-1) / K_max * Properties_res.shape[0])
nearest_index = np.abs(Sit_change - target_number).argmin()
Start_K = int(Sit_change[nearest_index]+1)
Start_t = Properties_res[Start_K, 0]
print("Start time: ", Start_t)
print("Stop time: ", Stop_t)

print("Start k: ", Start_K)
print("Stop k: ", Stop_K)


def get_boundaries_for_person(P):
    """Return the boundary variables for a given person P."""
    # Boundaries defined for each person
    boundaries = {
        0: (120, 160, 150, 170),  # Number 1: XBminN1, XBmaxN1, XBminF1, XBmaxF1
        1: (160, 195, 170, 190),  # Number 2: XBminN2, XBmaxN2, XBminF2, XBmaxF2
        2: (195, 230, 190, 220),  # Number 3: XBminN3, XBmaxN3, XBminF3, XBmaxF3
    }
    # Return the corresponding boundaries for the given person P
    return boundaries.get(P, (None, None, None, None))  # Return None values if P is not found


Model_data = np.zeros((Data_res.shape[0]*Data_res.shape[1],Properties_res.shape[1]))
Model_label = np.zeros((Data_res.shape[0]*Data_res.shape[1]), dtype="O")
inter = 0

def check_conditions(P, j, z):
    """
    Function to check conditions based on person P, time index j, and data index z.
    """
    # Get boundaries for the current person
    XBminN, XBmaxN, XBminF, XBmaxF = get_boundaries_for_person(P)
    
    # Check presence
    if Properties_res[j, 6*P+1] == 1:  
        # Near or Far, Stationary or moving
        near_far = Properties_res[j, 6*P+2]
        stationary_moving = Properties_res[j, 6*P+3]
        body_position = Data_res[z, j, 2]
        direction = Properties_res[j, 6*P+6]  # Direction for entering/leaving
        
        # Check Near
        if near_far == 1:
            if stationary_moving != 7 and XBminN < body_position < XBmaxN:
                return f"Number {P+1}"
            elif stationary_moving == 7 and direction <= 0 and 100 < body_position < 180:
                return f"Number {P+1}"
            elif stationary_moving == 7 and direction >= 0 and 180 < body_position < 250:
                return f"Number {P+1}"

        # Check Far
        elif near_far == 2:
            if stationary_moving != 7 and XBminF < body_position < XBmaxF:
                return f"Number {P+1}"
            elif stationary_moving == 7 and direction <= 0 and 100 < body_position < 180:
                return f"Number {P+1}"
            elif stationary_moving == 7 and direction >= 0 and 180 < body_position < 250:
                return f"Number {P+1}"

    return None

for z in range(Data_res.shape[0]):
    for j in range(Data_res.shape[1]):
        if Data_res[z, j, 0] == 0:  # Skip if no data
            continue
        Model_data[inter] = Properties_res[j]
        # Process each person
        for P in range(3):  # Assuming 3 people, adjust as necessary
            label = check_conditions(P, j, z)
            if label:
                Model_label[inter] = label
                break  # Assuming only one label per data point, adjust if needed
            else:
                Model_label[inter] = "No Data"
        inter += 1


Model_data = np.array(Model_data[0:inter])
Model_label = np.array(Model_label[0:inter])

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to the Model_data
scaler = scaler.fit(Model_data)

# Transform Model_data to get the normalized version
Model_data_normal = scaler.transform(Model_data)

# Remove the first column from the normalized data
Model_data_normal = np.delete(Model_data_normal, 0, axis=1)

def Labell(String):
    if String == "Number 1":
        return 0
    elif String == "Number 2":
        return 1
    elif String == "Number 3":
        return 2
    

# Initialize sequences for training and testing data and labels
train_data_normal_sequence = np.zeros((Model_data_normal.shape[0] - sequence, sequence, Model_data_normal.shape[1]), dtype='float')
train_label_sequence = np.zeros((Model_data_normal.shape[0] - sequence, 1), dtype='int')
test_data_normal_sequence = np.zeros((Model_data_normal.shape[0] - sequence, sequence, Model_data_normal.shape[1]), dtype='float')
test_label_sequence = np.zeros((Model_data_normal.shape[0] - sequence, 1), dtype='int')

train_inter = 0
test_inter = 0

for i in range(Model_data_normal.shape[0] - sequence):
    # Check for non-"No Data" labels and valid sequence time span
    if Model_label[i + sequence] != "No Data" and (Model_data[i, 0] - Model_data[i - sequence, 0]) > 0:
        if Start_t <= Model_data[i, 0] < Stop_t - resolution * sequence:
            # Assign to testing set
            test_label_sequence[test_inter] = Labell(Model_label[i + sequence])
            test_data_normal_sequence[test_inter, :] = Model_data_normal[i:i + sequence]
            test_inter += 1
        elif Model_data[i, 0] < Start_t - resolution * sequence or Model_data[i, 0] >= Stop_t:
            # Assign to training set
            train_label_sequence[train_inter] = Labell(Model_label[i + sequence])
            train_data_normal_sequence[train_inter, :] = Model_data_normal[i:i + sequence]
            train_inter += 1


# Resize arrays to the actual number of sequences processed
train_data_normal_sequence = train_data_normal_sequence[:train_inter]
train_label_sequence = train_label_sequence[:train_inter]
test_data_normal_sequence = test_data_normal_sequence[:test_inter]
test_label_sequence = test_label_sequence[:test_inter]
    

print("ratio of the training data to testing data: ",train_inter/test_inter)


train_data = np.array(train_data_normal_sequence)
train_label = np.array(train_label_sequence)
test_data = np.array(test_data_normal_sequence)
test_label = np.array(test_label_sequence)

print("train_data:",train_data.shape)
print("train_label:",train_label.shape)
print("test_data:",test_data.shape)
print("test_label:",test_label.shape)


#X_train = train_data.reshape(train_data.shape[0],sequence, -1)
one_hot_encoded_labels_train = to_categorical(train_label,3)

#X_test = test_data.reshape(test_data.shape[0],sequence, -1)
one_hot_encoded_labels_test = to_categorical(test_label,3)


import tensorflow as tf
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )

    def call(self, inputs):
        # Assuming inputs' shape: `(batch_size, sequence_length, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = tf.keras.Sequential(
            [tf.keras.layers.Dense(dense_dim, activation='swish'),
             tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, mask=None):
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

def get_compiled_model(sequence_length, num_features, num_classes):
    inputs = tf.keras.Input(shape=(sequence_length, num_features))
    x = PositionalEmbedding(sequence_length, num_features)(inputs)
    x = TransformerEncoder(num_features, dense_dim=1024, num_heads=2)(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Assuming `train_data`, `one_hot_encoded_labels_train`, `test_data`, and `one_hot_encoded_labels_test` are already defined
sequence_length = train_data.shape[1]
num_features = train_data.shape[2]
num_classes = one_hot_encoded_labels_train.shape[1]

model = get_compiled_model(sequence_length, num_features, num_classes)
model.summary()

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)

model_training_history = model.fit(
        x=train_data,
        y=one_hot_encoded_labels_train,
        epochs=epoch,
        batch_size=20,
        shuffle=True,
        validation_data=(test_data, one_hot_encoded_labels_test),
        callbacks=[early_stopping_callback]
    )



        
y_pred_train = model.predict(train_data)
y_pred_test = model.predict(test_data)

def calculate_top_k_accuracy(y_pred, y_true, k=1):
    """
    Calculate top-k accuracy for predictions.
    
    :param y_pred: Predicted probabilities (num_samples, num_classes)
    :param y_true: True labels (num_samples, 1)
    :param k: Consider prediction correct if true label is among top k predictions
    :return: Top-k accuracy
    """
    top_k_predictions = np.argsort(y_pred, axis=1)[:, -k:]
    correct_predictions = np.any(top_k_predictions == y_true, axis=1)
    return np.mean(correct_predictions)

# Calculate accuracies
train_accuracies = [calculate_top_k_accuracy(y_pred_train, train_label, k) for k in range(1, 4)]
test_accuracies = [calculate_top_k_accuracy(y_pred_test, test_label, k) for k in range(1, 4)]

A = np.array([train_accuracies, test_accuracies])
print(A)

# try:
#     os.makedirs("Results/", exist_ok=True)
#     print("Directory was created successfully.")
# except OSError as error:
#     pass

# Define the name and path for saving results and model
Name = f"Transformers-K{K}-KMax{K_max}"

results_dir = "Results/"

# Ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)

results_path = os.path.join(results_dir, Name)
np.savetxt(results_path + '.csv', A, delimiter=',')  # Save accuracy results
model.save(results_path + '.h5')  # Save the model
