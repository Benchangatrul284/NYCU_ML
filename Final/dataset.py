import torch
import torchvision.transforms as v2
from PIL import Image
import matplotlib.pyplot as plt

# Define the transformations
trans1 = v2.Compose([
    v2.Resize((185,160)),
    v2.GaussianBlur(kernel_size=3),
    v2.RandomHorizontalFlip(p=1),
    v2.ToTensor(),  # Assuming ToImage() was meant to convert to tensor for PyTorch
    v2.ConvertImageDtype(torch.float32)  # Assuming ToDtype() was meant to convert image dtype
])

trans2 = v2.Compose([
    v2.Resize((185,160)),
    v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    v2.ToTensor(),
    v2.ConvertImageDtype(torch.float32)
])

# Load an image
image_path = 'dataset/test/adults/0.jpg'  # Update this path
image = Image.open(image_path)

# Apply the transformation
transformed_image1 = trans1(image)
transformed_image2 = trans2(image)

# Convert back to PIL Image to display
transformed_image_pil1 = v2.ToPILImage()(transformed_image1)
transformed_image_pil2 = v2.ToPILImage()(transformed_image2)

# Display the images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(transformed_image_pil1)
plt.title('Transformed Image 1')
plt.subplot(1, 2, 2)
plt.imshow(transformed_image_pil2)
plt.title('Transformed Image 2')
plt.show()