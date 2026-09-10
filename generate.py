import pickle
import numpy as np
import random
import os

from tensorflow.keras.models import load_model


# -----------------------------
# Settings
# -----------------------------

MODEL_FILE = "models/music_lstm.keras"
NOTES_FILE = "processed/notes.pkl"

SEQUENCE_LENGTH = 50


# -----------------------------
# Load notes
# -----------------------------

print("Loading notes...")

with open(NOTES_FILE, "rb") as file:
    notes = pickle.load(file)

unique_notes = sorted(set(notes))

note_to_int = {
    note: number
    for number, note in enumerate(unique_notes)
}

int_to_note = {
    number: note
    for number, note in enumerate(unique_notes)
}


# -----------------------------
# Load trained model
# -----------------------------

print("Loading trained model...")

model = load_model(MODEL_FILE)

print("Model loaded successfully!")


# -----------------------------
# Select random starting sequence
# -----------------------------

start = random.randint(
    0,
    len(notes) - SEQUENCE_LENGTH - 1
)

pattern = notes[start:start + SEQUENCE_LENGTH]

print("Starting music generation...")


# -----------------------------
# Generate notes
# -----------------------------

prediction_output = []

for _ in range(200):

    input_sequence = [
        note_to_int[note]
        for note in pattern
    ]

    prediction_input = np.reshape(
        input_sequence,
        (1, SEQUENCE_LENGTH, 1)
    )

    prediction_input = (
        prediction_input / float(len(unique_notes))
    )

    prediction = model.predict(
        prediction_input,
        verbose=0
    )

    index = np.argmax(prediction)

    result = int_to_note[index]

    prediction_output.append(result)

    pattern.append(result)
    pattern = pattern[1:]


print("Music generation complete!")

print(f"Generated {len(prediction_output)} notes/chords.")


# -----------------------------
# Save generated notes
# -----------------------------

os.makedirs("output", exist_ok=True)

with open("output/generated_notes.pkl", "wb") as file:
    pickle.dump(prediction_output, file)

print("Saved generated notes to:")
print("output/generated_notes.pkl")
