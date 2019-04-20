#include <cv.h>
#include <highgui.h>
#include <iostream>

using namespace cv;
using namespace std;

int T = 100; // Approximation threshold
float f_s = 0.5f; // f sigma, blur factor 
int maxRadius = 8;
float f_c = 1.0f; // curvature filter
float a = 1.0f; // Opacity
float f_g = 1.0f; // Grid Size
int minLength = 4; // Minimum stroke lengths 
int maxLength = 16; // Maximum stroke lengths

void calculateSobelVectors(Mat ref){

}

void Paint(Mat src){
    Mat refImage(src.rows, src.cols, CV_8UC3, Scalar(255,255,255));
    // imwrite("ref.jpg", refImage);
    for(int Ri=maxRadius; Ri>1; Ri/=2){
        GaussianBlur(src, refImage, Size(0,0), Ri*f_s, Ri*f_s);
        // calculateSobelVectors(refImage);
        imwrite("ref.jpg", refImage);
    }
}  

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: painterly_rendering <image_file>" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    Paint(image);

    return 0;
}