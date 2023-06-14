# Image Colorization Service using Deep Learning

The service is capable of colorizing black and white images using a combination of U-Net and conditional Generative Adversarial Network (GAN), which has been trained on the COCO dataset.

The Image Colorization Service offers the following functionalities:

- Colorizing black and white images in JPEG format, including batch processing.
- Supporting the addition of new data to the dataset.
- Running experiments for further improvement.
- Deploying new models.

The model inference process is optimized for efficiency through the use of ONNX format.

## Technologies & Frameworks Used

The Image Colorization API is built using the following technologies and frameworks:

- Python: The programming language used for developing the service.
- FastAPI: A modern, fast (high-performance), web framework for building APIs with Python.
- PyTorch: A deep learning framework used for training and deploying the U-Net and conditional GAN models.
- ONNX: An open format for representing deep learning models that enables efficient inference.
- WandB: A tool for visualizing and tracking machine learning experiments.
- GitHub: The version control system and code hosting platform used for managing the repository.

## Repository Structure

The structure of this repository is organized as follows:

- **app**: Contains the main source code for the Image Colorization API.
  - `main.py`: The main FastAPI application.
  - `colorization.py`: The colorization logic and model inference functions.
- **models**: Stores the trained models and ONNX model files.
- **data**: The dataset used for training the models.
- **scripts**: Additional scripts for data processing, model training, and experimentation.
- **tests**: Unit tests for the functionality of the service.
- **requirements.txt**: A file listing the required Python dependencies for running the service.
- **Dockerfile**: The Dockerfile for containerizing the Image Colorization API.
- **README.md**: The file you are currently reading, providing an overview of the repository and its structure.

Feel free to explore each directory for more details about the specific components and functionalities of the Image Colorization Service.