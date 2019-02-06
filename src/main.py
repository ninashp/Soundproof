"""
Main file, run to compare speaker in two calls.
See config.py for system parameters
"""
import speaker_verfier
from configuration import get_config

config = get_config()

if __name__ == "__main__":
    # verify speaker in two  calls 
    if speaker_verfier.speaker_verifier(config.calls_to_compare_path):
        print("Result: Same Customer!")
    else:
        print("Result: Different Customers!")        