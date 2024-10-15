# Replicating Experiment from the Paper: Noise-aware Canny Algorithm for Edge Detection

This project aims to replicate the experiment described in the paper **Noise-aware Canny Algorithm for Edge Detection**. The implementation follows the steps outlined below to perform edge detection on images affected by different types of noise.

## Group Members

- Arquimedes França
- Gil Araújo
- João Vitor Russo
- José Franklin
- Gabriel Sá

## Process Steps

- [ ] **Grayscale Conversion**: Convert the original image to grayscale to simplify the processing.
  
- [ ] **Gaussian Filter**: Apply a Gaussian filter to smooth the image and reduce the effect of noise.
  
- [ ] **Gradient and Direction Calculation**: Compute the gradient magnitude and direction, essential for identifying edges in the images.
  
- [ ] **Non-Maximum Suppression (NMS)**: Reduce the number of points that do not represent edges, preserving only the local maxima.
  
- [ ] **Double Thresholding and Edge Tracking**: Apply two thresholds to classify strong and weak edges, connecting the weak edges to the strong ones for more accurate detection.

## Tests Performed

- [ ] **Salt and Pepper Noise**: Evaluate the algorithm's performance on images affected by salt and pepper noise.
  
- [ ] **Gaussian Noise**: Test the robustness of the algorithm on images with Gaussian noise.
  
- [ ] **Threshold Variations**: Experiment with different thresholds to assess how it impacts edge detection.

## Reference

Yitong Yuan, "Noise-aware Canny Algorithm for Edge Detection," in *2022 IEEE 4th International Conference on Civil Aviation Safety and Information Technology (ICCASIT)*, 2022, pp. 1107-1110. doi: [10.1109/ICCASIT55263.2022.9986653](https://doi.org/10.1109/ICCASIT55263.2022.9986653)
