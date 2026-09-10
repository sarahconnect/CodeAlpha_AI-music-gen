from music21 import converter, instrument, note, chord
import glob
import os
import pickle

MIDI_FOLDER = "dataset/midi"
OUTPUT_FILE = "processed/notes.pkl"


def extract_notes():
    notes = []

    midi_files = glob.glob(os.path.join(MIDI_FOLDER, "*.mid"))
    midi_files += glob.glob(os.path.join(MIDI_FOLDER, "*.midi"))

    print(f"Found {len(midi_files)} MIDI files.")

    for i, file in enumerate(midi_files):
        print(f"Processing {i + 1}/{len(midi_files)}: {os.path.basename(file)}")

        try:
            midi = converter.parse(file)

            parts = instrument.partitionByInstrument(midi)

            if parts:
                elements = parts.parts[0].recurse()
            else:
                elements = midi.flat.notes

            for element in elements:

                # Single note
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))

                # Chord
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))

        except Exception as e:
            print(f"Error processing {file}: {e}")

    os.makedirs("processed", exist_ok=True)

    with open(OUTPUT_FILE, "wb") as file:
        pickle.dump(notes, file)

    print("\n-----------------------------")
    print("DATA PROCESSING COMPLETE!")
    print("-----------------------------")
    print(f"Total notes/chords: {len(notes)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_notes()
