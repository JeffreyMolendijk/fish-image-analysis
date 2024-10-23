import logging
from ml import machine_learning
from setup_dirs import setup_dirs
from rotate_crop import rotate_crop
from measure_magenta import measure_magenta

def main():
    try:
        logging.info("Setting up directories...")
        setup_dirs()
        
        logging.info("Rotating and cropping images...")
        rotate_crop()
        
        logging.info("Measuring magenta content...")
        measure_magenta()
        
        logging.info("Running machine learning analysis...")
        machine_learning()
        
        logging.info("Process completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()