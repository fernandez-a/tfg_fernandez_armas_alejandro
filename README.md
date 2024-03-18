# 
The code has been put inside a docker container to help the reproduction of the results.

## Docker

For running the code you will need to ensure that you have free memory on your system and then:

### Build run image

docker build -t <your_image_name> . to build the image

docker docker run --rm <your_image_name> to run the image

### Build visualize image

docker build -t <your_image_name> . to build the image

docker docker run --rm <your_image_name> to run the image



The detr model code has been adapted from https://www.kaggle.com/code/tanulsingh077/end-to-end-object-detection-with-transformers-detr/notebook#Wheat-Detection-Competition-With-DETR to work for the specific use case of wine leaf disease detection.

If you want the images it can be downloaded on https://data.mendeley.com/datasets/j4xs3kh3fd/2 don't forget to cite the authors of the annotated dataset.