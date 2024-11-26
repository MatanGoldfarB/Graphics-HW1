#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv; 

// Our implementation of Filter2D from opencv
void customFilter2D(const Mat& src, Mat& dst, const Mat& kernel, Point anchor = Point(-1, -1), double delta = 0) {
    // Kernel properties
    int kRows = kernel.rows;
    int kCols = kernel.cols;
    int kCenterX = kCols / 2;
    int kCenterY = kRows / 2;

    // Anchor point adjustment
    if (anchor == Point(-1, -1)) {
        anchor = Point(kCenterX, kCenterY);
    }

    // Pad the source image
    Mat paddedSrc;
    copyMakeBorder(src, paddedSrc, kCenterY, kCenterY, kCenterX, kCenterX, BORDER_REPLICATE);

    // Create destination matrix
    dst = Mat::zeros(src.size(), src.type());

    // Apply convolution
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            double sum = 0.0;

            for (int ky = 0; ky < kRows; ++ky) {
                for (int kx = 0; kx < kCols; ++kx) {
                    // Compute the padded image indices
                    int srcY = y + ky;
                    int srcX = x + kx;

                    // Perform convolution
                    sum += paddedSrc.at<double>(srcY, srcX) * kernel.at<double>(ky, kx);
                }
            }
            // Write result to the destination matrix
            dst.at<double>(y, x) = sum + delta;
        }
    }
}

// Generates gaussian kernels
Mat generateGaussianKernel(int size, double sigma) {
    Mat kernel(size, size, CV_64F); // Double precision kernel
    double sum = 0.0; // For normalization
    int half = size / 2;

    // Generate kernel
    for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
            double value = (1 / (2 * M_PI * sigma * sigma)) * exp(-(i * i + j * j) / (2 * sigma * sigma));
            kernel.at<double>(i + half, j + half) = value;
            sum += value;
        }
    }
    // Normalize kernel
    kernel /= sum;

    return kernel;
}

// Calculates the gradient of the image
void gradientCalc(const Mat& gausFiltered, Mat& grad, Mat& angle) {
    Mat dX = (Mat_<double>(3, 3) <<
    0, 0, 0,
    -1, 0, 1,
    0, 0, 0);

    Mat dY = (Mat_<double>(3, 3) <<
    0, -1, 0,
    0, 0, 0,
    0, 1, 0);

    Mat gradX, gradY, gaus64F;
    gausFiltered.convertTo(gaus64F, CV_64F);
    customFilter2D(gaus64F, gradX, dX);
    customFilter2D(gaus64F, gradY, dY);

    Mat squaredX, squaredY;
    multiply(gradX, gradX, squaredX);
    multiply(gradY, gradY, squaredY);

    grad = squaredX + squaredY;
    sqrt(grad, grad);
    //Mat angle;
    phase(gradX, gradY, angle, true); // 'true' for angle in degrees; set to 'false' for radians
}

void nonMaxSuppression(const Mat& grad, const Mat& angle, Mat& nms) {
    nms = Mat::zeros(grad.size(), CV_64F); // Output matrix

    int rows = grad.rows, cols = grad.cols;

    for (int i = 1; i < rows - 1; ++i) { // Avoid border pixels
        for (int j = 1; j < cols - 1; ++j) {
            double magnitude = grad.at<double>(i, j);
            double theta = angle.at<double>(i, j);

            // Quantize gradient direction
            theta = fmod(theta + 180, 180.0); // Ensure theta is in [0, 180)
            double neighbor1 = 0, neighbor2 = 0;

            if ((0 <= theta && theta < 22.5) || (157.5 <= theta && theta < 180)) {
                neighbor1 = grad.at<double>(i, j - 1); // Left
                neighbor2 = grad.at<double>(i, j + 1); // Right
            } else if (22.5 <= theta && theta < 67.5) {
                neighbor1 = grad.at<double>(i - 1, j + 1); // Top-right
                neighbor2 = grad.at<double>(i + 1, j - 1); // Bottom-left
            } else if (67.5 <= theta && theta < 112.5) {
                neighbor1 = grad.at<double>(i - 1, j); // Top
                neighbor2 = grad.at<double>(i + 1, j); // Bottom
            } else if (112.5 <= theta && theta < 157.5) {
                neighbor1 = grad.at<double>(i - 1, j - 1); // Top-left
                neighbor2 = grad.at<double>(i + 1, j + 1); // Bottom-right
            }

            // Preserve pixel if it is larger than both neighbors
            if (magnitude >= neighbor1 && magnitude >= neighbor2) {
                nms.at<double>(i, j) = magnitude;
            } else {
                nms.at<double>(i, j) = 0;
            }
        }
    }
}

void doubleThreshold(const Mat& grad, Mat& edges,const int lowThresh,const int highThresh) {
    edges = Mat::zeros(grad.size(), CV_8U); // Initialize output matrix as CV_8U

    for (int i = 0; i < grad.rows; ++i) {
        for (int j = 0; j < grad.cols; ++j) {
            double magnitude = grad.at<double>(i, j);

            if (magnitude >= highThresh) {
                edges.at<uchar>(i, j) = 255; // Strong edge
            } else if (magnitude >= lowThresh) {
                edges.at<uchar>(i, j) = 1; // Weak edge
            } else {
                edges.at<uchar>(i, j) = 0; // Non-edge
            }
        }
    }
}

void hysteresis(const Mat& edges, Mat& finalEdges) {
    // Clone the edges matrix to create the final result
    finalEdges = edges.clone();

    // Loop through all pixels, skipping borders
    for (int i = 1; i < edges.rows - 1; ++i) {
        for (int j = 1; j < edges.cols - 1; ++j) {
            // Process only Weak Edges (value 128)
            if (finalEdges.at<uchar>(i, j) == 1) {
                bool connectedToStrongEdge = false;

                // Check all 8-connected neighbors
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (finalEdges.at<uchar>(i + di, j + dj) == 255) {
                            connectedToStrongEdge = true;
                            break; // No need to check further
                        }
                    }
                    if (connectedToStrongEdge) break;
                }

                // Update the weak edge based on its connectivity
                if (connectedToStrongEdge) {
                    finalEdges.at<uchar>(i, j) = 255; // Promote to Strong Edge
                } else {
                    finalEdges.at<uchar>(i, j) = 0;   // Suppress
                }
            }
        }
    }
}

void choosePattern(const uchar intensity, Mat& pattern) {
    if (intensity >= 0 && intensity < 51) {
        pattern = (Mat_<uchar>(2, 2) <<
        0, 0,
        0, 0);
    } else if (intensity >= 51 && intensity < 102) {
        pattern = (Mat_<uchar>(2, 2) <<
        0, 0,
        255, 0);
    } else if (intensity >= 102 && intensity < 153) {
        pattern = (Mat_<uchar>(2, 2) <<
        0, 255,
        255, 0);
    } else if (intensity >= 153 && intensity < 204) {
        pattern = (Mat_<uchar>(2, 2) <<
        0, 255,
        255, 255);
    } else {
        pattern = (Mat_<uchar>(2, 2) <<
        255, 255,
        255, 255);
    }
}

// Halftone function
void createHalftone(const Mat& input, Mat& output) {
    // Process each pixel
    output = Mat::zeros(input.rows * 2, input.cols * 2, CV_8U);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            // Get intensity of the current pixel
            uchar intensity = input.at<uchar>(i, j);

            // Map intensity to a 2x2 halftone block
            Mat block;
            choosePattern(intensity, block);

            // Place the block in the correct location in the output
            int rowStart = i * 2;
            int colStart = j * 2;
            block.copyTo(output(Rect(colStart, rowStart, 2, 2)));
        }
    }
}

// Function to perform Floyd-Steinberg dithering
void floydSteinbergDithering(const Mat& input, Mat& output) {
    output = input.clone();

    // Determine the step size based on the number of levels
    double step = 255.0 / 15;

    // Process each pixel
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            // Get the original intensity
            uchar oldPixel = output.at<uchar>(y, x);

            // Quantize the pixel to the nearest level
            uchar newPixel = floor(oldPixel / step) * step;

            // Update the pixel value
            output.at<uchar>(y, x) = newPixel;

            // Calculate the quantization error
            double error = oldPixel - newPixel;

            // Error weights for neighbors
            const double weights[4] = {7.0 / 16, 3.0 / 16, 5.0 / 16, 1.0 / 16};

            // Neighbor coordinates: (dx, dy)
            const int neighbors[4][2] = {
                {1, 0},   // Right
                {-1, 1},  // Bottom-left
                {0, 1},   // Bottom
                {1, 1}    // Bottom-right
            };

            // Calculate total weight for valid neighbors
            double totalWeight = 0.0;
            for (int k = 0; k < 4; ++k) {
                int nx = x + neighbors[k][0];
                int ny = y + neighbors[k][1];
                if (nx >= 0 && nx < input.cols && ny >= 0 && ny < input.rows) {
                    totalWeight += weights[k];
                }
            }

            // Distribute error to valid neighbors
            for (int k = 0; k < 4; ++k) {
                int nx = x + neighbors[k][0];
                int ny = y + neighbors[k][1];
                if (nx >= 0 && nx < input.cols && ny >= 0 && ny < input.rows) {
                    double adjustedWeight = weights[k] / totalWeight; // Normalize weight
                    output.at<uchar>(ny, nx) = saturate_cast<uchar>(
                        output.at<uchar>(ny, nx) + error * adjustedWeight);
                }
            }
        }
    }
}

void showFourImages(const Mat& img1, const Mat& img2, const Mat& img3, const Mat& img4) {
    // Ensure all images are the same size
    Size imageSize(256, 256);
    Mat resized1, resized2, resized3, resized4;
    resize(img1, resized1, imageSize);
    resize(img2, resized2, imageSize);
    resize(img3, resized3, imageSize);
    resize(img4, resized4, imageSize);

    // Create a blank canvas for the 2x2 grid
    Mat canvas(512, 512, resized1.type());

    // Place each image in its quadrant
    resized1.copyTo(canvas(Rect(0, 0, 256, 256))); // Top-left
    resized2.copyTo(canvas(Rect(256, 0, 256, 256))); // Top-right
    resized3.copyTo(canvas(Rect(0, 256, 256, 256))); // Bottom-left
    resized4.copyTo(canvas(Rect(256, 256, 256, 256))); // Bottom-right

    // Display the combined image
    imshow("Four Images", canvas);
    waitKey(0);
}

void cannyFilter(const Mat& grayImg, Mat& canny) {
    // First step: Gaussian Filter
    int kernelSize = 5;
    double sigma = 1.4;
    Mat kernel = generateGaussianKernel(kernelSize, sigma);
    Mat gausFiltered, gray64F;
    grayImg.convertTo(gray64F, CV_64F);
    customFilter2D(gray64F, gausFiltered, kernel);

    // Second step: Gradient Calculation
    Mat angle, grad;
    gradientCalc(gausFiltered, grad, angle);
    Mat gradDisplay;
    normalize(grad, gradDisplay, 0, 255, NORM_MINMAX);
    gradDisplay.convertTo(gradDisplay, CV_8U);

    // Third step: Non-Max Suppression
    Mat nms;
    nonMaxSuppression(grad, angle, nms);
    Mat nmsDisplay;
    normalize(nms, nmsDisplay, 0, 255, NORM_MINMAX);
    nmsDisplay.convertTo(nmsDisplay, CV_8U);

    // Fourth step: Double Thresholding
    Mat edges;
    double maxGrad;
    minMaxLoc(nms, nullptr, &maxGrad); // Find the maximum gradient value
    double highThresh = 0.2 * maxGrad; // Set high threshold as 20% of max gradient
    double lowThresh = 0.1 * highThresh; // Set low threshold as 10% of high threshold
    doubleThreshold(nms, edges, lowThresh, highThresh);
    showFourImages(gausFiltered, gradDisplay, nmsDisplay, edges);
    // Fifth step: Hysteresis
    hysteresis(edges, canny);
}

int main(int argc, char** argv) {
    // Check if image path is provided as an argument
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    // Load the image
    Mat image = imread(argv[1]); // argv[1] is the path to the image file

    // Check if the image is loaded successfully
    if (image.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // Task 1: Graycale Image
    Mat grayImg;
    cvtColor(image, grayImg, COLOR_BGR2GRAY);
    imwrite("Grayscale.png", grayImg);
    imshow("Grayscale Display", grayImg);
    waitKey(0);

    // Task 2: Canny 
    Mat canny;
    cannyFilter(grayImg, canny);
    imshow("Canny Result", canny);
    imwrite("Canny.png", canny);
    waitKey(0);

    // Task 3: Halftone
    Mat halftoneImage;
    createHalftone(grayImg, halftoneImage);
    imshow("Halftone Image", halftoneImage);
    imwrite("Halftone.png", halftoneImage);
    waitKey(0);

    // Task 4: FloyedSteinberg 
    Mat ditheredImage;
    floydSteinbergDithering(grayImg, ditheredImage);
    imshow("Floyd-Steinberg Dithered Image", ditheredImage);
    imwrite("FloyedSteinberg.png", ditheredImage);
    waitKey(0);
    return 0;
}