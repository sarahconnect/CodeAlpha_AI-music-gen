# CodeAlpha_AI-music-gen
AI music generation using LSTM neural networks and MIDI data with Python and music21. 
🎵 AI Music Generator

An AI-based music generation project that uses a Long Short-Term Memory (LSTM) neural network to learn musical patterns from MIDI files and generate new music.

📌 Project Overview

This project uses MIDI music data to train an LSTM neural network. The model learns patterns in notes and chords and then generates new musical sequences.

The generated sequences are converted back into a MIDI file and can be played using Python.

🧠 How It Works
MIDI Dataset
     ↓
Music21 Preprocessing
     ↓
Notes & Chords
     ↓
Training Sequences
     ↓
LSTM Neural Network
     ↓
Model Training
     ↓
Generate New Notes
     ↓
Create MIDI File
     ↓
Play Generated Music

**Features**
MIDI music data processing
Note and chord extraction using music21
LSTM-based deep learning model
Training on musical sequences
AI-generated musical sequences
MIDI file generation
MIDI playback using Pygame
 **Technologies Used**
Python
TensorFlow / Keras
LSTM Neural Networks
music21
NumPy
Pygame
MIDI
📁 Project Structure
ai-music-gen/
│
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── generate.py
│   └── midi_to_audio.py
│
├── dataset/
│   └── midi/
│
├── processed/
├── models/
├── output/
│
├── requirements.txt
├── .gitignore
└── README.md

**Installation**

Clone or download this repository and create a Python virtual environment.

Create virtual environment
python -m venv venv

Activate the virtual environment

Windows PowerShell:

.\venv\Scripts\Activate.ps1

Install dependencies
pip install -r requirements.txt

🎹 Dataset

The model is trained using MIDI piano music from the MAESTRO dataset.

The MIDI files should be placed inside:

dataset/midi/


The dataset should not be uploaded directly to this repository. Follow the MAESTRO dataset's terms and license when obtaining and using the data.

⚙️ Usage
1. Prepare the MIDI data

Place MIDI files in:

dataset/midi/


Then run:

python src/prepare_data.py


This extracts notes and chords and saves the processed data to:

processed/notes.pkl

2. Train the LSTM model

Run:

python src/train.py


The trained model will be saved as:

models/music_lstm.keras

3. Generate new music

Run:

python src/generate.py


The generated musical sequence will be saved as:

output/generated_notes.pkl

4. Create the MIDI file

Run:

python src/midi_to_audio.py


The generated music will be saved as:

output/generated_music.mid


The MIDI file can then be played using the project's MIDI playback functionality.

 **Current Model**

The current version uses:

LSTM neural network
128 LSTM units
Sequence length of 50
10 training epochs
Batch size of 128

The model was trained on approximately 57,996 musical notes/chords extracted from the MIDI dataset.
