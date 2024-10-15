# Step 1: Install Required Libraries
!pip install diffusers torch torchvision matplotlib transformers accelerate

# Step 2: Import Necessary Libraries
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Step 3: Function to Generate Image from Text
def generate_image_from_text(prompt):
    # Load the Stable Diffusion model (using a publicly available model)
    model_id = "stabilityai/stable-diffusion-2-1"  # Updated model ID
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Use GPU if available

    # Generate image
    with torch.no_grad():
        image = pipe(prompt).images[0]

    return image

# Step 4: Display the Generated Image
def display_image(image):
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()

# Step 5: Create 3D Shape from Image
def create_3d_shape_from_image(image):
    # Convert image to grayscale
    grayscale_image = image.convert("L")
    img_array = np.array(grayscale_image)

    # Create 3D surface
    x = np.arange(0, img_array.shape[1], 1)
    y = np.arange(0, img_array.shape[0], 1)
    x, y = np.meshgrid(x, y)
    z = img_array  # Use grayscale values as height

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')

    # Show the 3D plot
    plt.show()

# Step 6: Main Function to Generate and Visualize
def main():
    text_prompt = "A sunset view from the top of the mountain"  # Specify your prompt here
    generated_image = generate_image_from_text(text_prompt)
    display_image(generated_image)
    create_3d_shape_from_image(generated_image)

# Run the main function
if __name__ == "__main__":
    main()
