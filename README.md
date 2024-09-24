# Image Search Application with OpenAI CLIP Model and Faiss Library

This repository contains an Image Search Application that leverages OpenAI's [CLIP (Contrastive Language-Image Pretraining)](https://github.com/openai/CLIP) model and Meta's [Faiss (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) library to enable efficient and accurate similarity search capabilities. The application allows users to perform image searches by inputting natural language queries.

## Demo

![Demo](demo.png)
![Demo](demo.gif)

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Files](#files)
- [License](#license)


## Introduction

### CLIP Model
CLIP (Contrastive Language-Image Pretraining) is a state-of-the-art deep learning model developed by OpenAI, designed to bridge the gap between natural language processing and computer vision. It achieves this by jointly learning to understand images and text through contrastive learning. CLIP takes a pair of an image and a text prompt as inputs and processes them using a CNN and a transformer-based language model, respectively. The model outputs dense embeddings that encode the information of both inputs in a shared space. This enables CLIP to perform various tasks, such as image classification and image search based on natural language queries, without the need for task-specific training.

#### CLIP model usage:
```python
import clip
import torch

model, preprocess = clip.load("ViT-B/32")

images = preprocess(images)
texts = clip.tokenize(texts)

with torch.no_grad():
    image_features = model.encode_image(images)
    text_features = model.encode_text(texts)
```

### Faiss Library
Faiss is an efficient and powerful library developed by Facebook AI Research (FAIR) for similarity search and clustering of dense vectors. It is specifically designed to handle large-scale datasets and high-dimensional vector spaces, making it well-suited for applications in computer vision, natural language processing, and machine learning. Its optimized implementations enable users to perform nearest neighbor searches and clustering operations with outstanding speed and memory efficiency. 

#### Faiss usage:
```python
import faiss

index = faiss.IndexFlatIP(d)  # d is the dimension of the vectors
index.add(vectors)  # indexing

distances, indices = index.search(query_vectors, k)  # k is the number of nearest neighbors to search for
```

### Design
The design of the Image Search Application revolves around the integration of the pretrained CLIP model and Faiss library to enable efficient image search capabilities. Initially, the application processes all images in the search space using the pretrained CLIP model, generating dense embeddings that represent the visual information contained in each image. These embeddings are then indexed and stored using Faiss, allowing for fast and accurate "nearest inner product neighbor" searches.

When a user submits a natural language query, the application takes the input and employs the same CLIP model to generate an embedding that represents the semantic information of the query. It then performs a nearest neighbor search within the Faiss index to find image embeddings that best match the query's embedding. The results are then presented to the user, displaying visually similar images based on their textual description.

## Installation
1. Clone the repository
```commandline
git clone https://github.com/abinthomasonline/clip-faiss.git
```
2. Install [RyzenAI Software packages](https://ryzenai.docs.amd.com/en/latest/inst.html), activate the conda environment where the RyzenAI installed
```
conda activate <env_name>
```
3. Install the required packages
```commandline
conda env update --file env.yaml
pip install ops\cpp --force-reinstall
```
4. Setup envirenment variables for RyzenAI
```
.\setup.bat
```

## Usage
### Prepare models for RyzenAI
#### Quantize Resnet50 & export ONNX model
1. Automatically download clip/RN50, which stores in `.\Models\`
2. Download dataset for quantization to `.\Datasets\`
3. Quantize and convert the RN50 section to ONNX, saved to `.\onnx\`.
```
python prepare_model_clip.py
```
The output contains:
1. `model.onnx`: The entire model
2. `image_model.onnx`: The image encoder
3. `text_model.onnx`: The text encoder
4. `image_model_0.onnx`: The RN50 section of the image encoder
5. `image_model_1.onnx`: The image encoder without RN50
4. `image_model_0_quantized.onnx`: The quantized RN50
#### Quantize and export the text encoder
1. Simpify the onnx model so that more Matmul+add can be fused to GEMM
2. Quantize the model using dynamic quantization
3. Convert the model back to PyTorch model to dynamically offload GEMM operations to NPU.
```
python simpify_onnx_model.py
python convert_onnx_to_torch.py
```
The outputs are in `.\onnx\` as well and contains:
1. `text_model_reshaped.onnx`: The simplified text encoder
2. `text_model_dynamic_quant.pt`: The quantized PyTorch text encoder

### Create index
```commandline
python index.py --image_dir_path <path_to_images>
```
- `image_dir_path`: Path to directory containing images to be indexed

An index file named `index.faiss` will be created in the `static` directory.
A mapping file named `image_paths.json` will be created in the `static` directory.

### Search app in CLI
```commandline
python app.py
```
The application will prompt the user to enter a natural language query. The top result will be displayed in a new window. Enter `exit` to quit the application.

### Search Web App
```commandline
python serve.py
```
The application will be hosted at `http://localhost:5000/`. Enter a natural language query in the search bar and press `Search` to submit the query. The top 5 results will be displayed below the search bar. For security reasons proposed by browers, loading local images may broken.

## Dataset
The demo uses a dataset of 5400 animal images in 90 different categories. The dataset can be downloaded from [here](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals).
Index file and mapping file for this dataset is available in the `static` directory. 

## Files
```commandline
clip-faiss/
├── static/                    # Web app static files                
│   ├── data/                      
│   │   └── images/            # Images
│   ├── app.js 
│   ├── image_paths.json       # mapping file (demo)
│   ├── index.faiss            # index file (demo)
│   └── styles.css       
├── templates/                 # HTML templates
│   └── index.html             
├── app.py                     # Run CLI App
├── index.py                   # Indexing
├── README.md                  # This file
├── requirements.txt           # Requirements file
└── serve.py                   # Run Web App
```

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as per the terms of the license.