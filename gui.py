import sys

def remove_previous_line():
    erase_sequence = "\033[A" + "\033[2K"
    # Erase the entire line "You: {text_input}"
    sys.stdout.write(erase_sequence)
    sys.stdout.flush()