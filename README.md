# IMAGE-TRANSFORMATIONS

## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1:
Import necessary libraries (cv2, numpy, matplotlib) and load the source image.


### Step 2:
Create transformation matrices for translation, rotation, scaling, and shearing using functions like cv2.getRotationMatrix2D().


### Step 3:
Apply the geometric transformations using cv2.warpAffine() for affine transformations and cv2.flip() for reflection.


### Step 4:
Crop the image by selecting a specific rectangular region using NumPy array slicing.


### Step 5:
Display the original and all transformed images with appropriate titles using matplotlib.pyplot.


## Program:

## Developed By: YOKESH I
## Register Number: 212224230313
PYTHON
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Load the image
image = cv2.imread('1000077167.jpg')  # Load the image from file
# Display the original image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
plt.title("Original Image")  
plt.axis('off') 
(np.float64(-0.5), np.float64(1279.5), np.float64(812.5), np.float64(-0.5))
```

i)Image Translation
```

tx, ty = 100, 50  # Translation factors (shift by 100 pixels horizontally and 50 vertically)
M_translation = np.float32([[1, 0, tx], [0, 1, ty]])  # Translation matrix: 
# [1, 0, tx] - Horizontal shift by tx
# [0, 1, ty] - Vertical shift by ty
translated_image = cv2.warpAffine(image, M_translation, (image.shape[1], image.shape[0]))  
plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB))  # Display the translated image
plt.title("Translated Image")  
plt.axis('off')
(np.float64(-0.5), np.float64(1279.5), np.float64(812.5), np.float64(-0.5))
```

ii) Image Scaling
```

fx, fy = 5.0, 2.0  # Scaling factors (1.5x scaling for both width and height)
scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
# resize: Resize the image by scaling factors fx, fy
# INTER_LINEAR: Uses bilinear interpolation for resizing
plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))  # Display the scaled image
plt.title("Scaled Image")  # Set title
plt.axis('off')
(np.float64(-0.5), np.float64(6399.5), np.float64(1625.5), np.float64(-0.5))
```

iii)Image shearing
```
shear_matrix = np.float32([[1, 0.5, 0], [0.5, 1, 0]])  # Shearing matrix
# The matrix shears the image by a factor of 0.5 in both x and y directions
# [1, 0.5, 0] - Shear along the x-axis (horizontal)
# [0.5, 1, 0] - Shear along the y-axis (vertical)
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))
plt.imshow(cv2.cvtColor(sheared_image, cv2.COLOR_BGR2RGB))  # Display the sheared image
plt.title("Sheared Image")  # Set title
plt.axis('off')
(np.float64(-0.5), np.float64(1279.5), np.float64(812.5), np.float64(-0.5))
```

iv)Image Reflection
```

reflected_image = cv2.flip(image, 2)  # Flip the image horizontally (1 means horizontal flip)
# flip: 1 means horizontal flip, 0 would be vertical flip, -1 would flip both axes
plt.imshow(cv2.cvtColor(reflected_image, cv2.COLOR_BGR2RGB))  # Display the reflected image
plt.title("Reflected Image")  # Set title
plt.axis('off')
(np.float64(-0.5), np.float64(1279.5), np.float64(812.5), np.float64(-0.5))
```


v)Image Rotation
```

(height, width) = image.shape[:2]  # Get the image height and width
angle = 45  # Rotation angle in degrees (rotate by 45 degrees)
center = (width // 2, height // 2)  # Set the center of rotation to the image center
M_rotation = cv2.getRotationMatrix2D(center, angle, 1)  # Get the rotation matrix
# getRotationMatrix2D: Takes the center of rotation, angle, and scale factor (1 means no scaling)
rotated_image = cv2.warpAffine(image, M_rotation, (width, height))  # Apply rotation
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))  # Display the rotated image
plt.title("Rotated Image")  # Set title
plt.axis('off')
(np.float64(-0.5), np.float64(1279.5), np.float64(812.5), np.float64(-0.5))
```


vi)Image Cropping
```
x, y, w, h = 100, 100, 200, 150  # Define the top-left corner (x, y) and the width (w) and height (h) of the crop
# Cropping the image from coordinates (x, y) to (x+w, y+h)
cropped_image = image[y:y+h, x:x+w]
# The crop is performed by slicing the image array in the y and x directions
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))  # Display the cropped image
plt.title("Cropped Image")  # Set title
plt.axis('off')
(np.float64(-0.5), np.float64(199.5), np.float64(149.5), np.float64(-0.5))
```



## Output:
## ORIGINAL IMAGE

<img width="432" height="570" alt="image" src="https://github.com/user-attachments/assets/22a2e8b6-cb7c-48ee-a285-1f6b313980a0" />


### i)Image Translation
<img width="434" height="577" alt="image" src="https://github.com/user-attachments/assets/80054105-924e-4217-9b3f-5af37ea34201" />



### ii) Image Scaling
<img width="689" height="420" alt="image" src="https://github.com/user-attachments/assets/e8952150-5227-455e-9890-77b653e87f39" />


### iii)Image shearing
<img width="426" height="577" alt="image" src="https://github.com/user-attachments/assets/48568113-535f-4e26-91c6-51ab0bf16555" />



### iv)Image Reflection
<img width="448" height="578" alt="image" src="https://github.com/user-attachments/assets/ad0c9da9-8b86-442c-a756-b808fc780909" />


### v)Image Rotation
<img width="448" height="577" alt="image" src="https://github.com/user-attachments/assets/f0d23a45-e341-4617-9477-c7fbd19a949e" />




### vi)Image Cropping
<img width="334" height="560" alt="image" src="https://github.com/user-attachments/assets/c5d915f2-e62f-4f7b-b1a1-3a0039c17f19" />




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
