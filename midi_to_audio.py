import pygame
import time
import os

MIDI_FILE = "output/generated_music.mid"

print("Starting MIDI player...")

if not os.path.exists(MIDI_FILE):
    print("MIDI file not found!")
    exit()

pygame.mixer.init()

try:
    pygame.mixer.music.load(MIDI_FILE)
    print("MIDI loaded successfully!")
    print("Playing generated music...")

    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    print("Playback finished!")

except Exception as e:
    print("Could not play MIDI:")
    print(e)

finally:
    pygame.mixer.quit()

 
