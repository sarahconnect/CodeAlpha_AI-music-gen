import pickle
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# -----------------------------
# Settings
# -----------------------------

NOTES_FILE = "processed/notes.pkl"
MODEL_FILE = "models/music_lstm.keras"

SEQUENCE_LENGTH = 50


# -----------------------------
# Load notes
# -----------------------------

print("Loading notes...")

with open(NOTES_FILE, "rb") as file:
    notes = pickle.load(file)

print(f"Total notes/chords: {len(notes)}")


# -----------------------------
# Create vocabulary
# -----------------------------

unique_notes = sorted(set(notes))

print(f"Unique notes/chords: {len(unique_notes)}")


note_to_int = {
    note: number
    for number, note in enumerate(unique_notes)
}


# -----------------------------
# Create training sequences
# -----------------------------

network_input = []
network_output = []

print("Creating training sequences...")

for i in range(len(notes) - SEQUENCE_LENGTH):
    sequence = notes[i:i + SEQUENCE_LENGTH]
    target = notes[i + SEQUENCE_LENGTH]

    network_input.append(
        [note_to_int[note] for note in sequence]
    )

    network_output.append(
        note_to_int[target]
    )


# -----------------------------
# Convert to numpy arrays
# -----------------------------

n_patterns = len(network_input)

X = np.reshape(
    network_input,
    (n_patterns, SEQUENCE_LENGTH, 1)
)

# Normalize input
X = X / float(len(unique_notes))

y = to_categorical(
    network_output,
    num_classes=len(unique_notes)
)

print(f"Training sequences: {n_patterns}")
print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")


# -----------------------------
# Build LSTM model
# -----------------------------

print("\nBuilding LSTM model...")

model = Sequential()

model.add(
    LSTM(
        128,
        input_shape=(X.shape[1], X.shape[2]),
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(
    LSTM(128)
)

model.add(Dropout(0.3))

model.add(
    Dense(len(unique_notes), activation="softmax")
)


# -----------------------------
# Compile
# -----------------------------

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)


model.summary()


# -----------------------------
# Train
# -----------------------------

print("\nStarting training...")

model.fit(
    X,
    y,
    epochs=10,
    batch_size=128
)


# -----------------------------
# Save model
# -----------------------------

os.makedirs("models", exist_ok=True)

model.save(MODEL_FILE)

print("\n=============================")
print("TRAINING COMPLETE!")
print("=============================")
print(f"Model saved to: {MODEL_FILE}")
