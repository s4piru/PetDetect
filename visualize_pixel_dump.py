import pandas as pd
import numpy as np
from PIL import Image

def main():
    # Load pixel data from CSV
    csv_path = 'pixel_dump.csv'  # Update to the path of the file exported from iOS
    pixel_df = pd.read_csv(csv_path, header=None)
    
    # Convert pixel data to [224, 224, 3] NumPy array
    pixel_values = pixel_df.values  # Shape: (224*224, 3)
    pixel_array = pixel_values.reshape((224, 224, 3))
    
    # Scale from [0,1] to [0,255]
    pixel_array = np.clip(pixel_array * 255.0, 0, 255).astype(np.uint8)
    
    # Generate and save the image
    img = Image.fromarray(pixel_array, 'RGB')
    img.save('reconstructed_image.png')
    img.show()

if __name__ == "__main__":
    main()
