#include <cv.h>
#include <highgui.h>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    cout << "OpenCV version : " << CV_VERSION << endl;
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    uint8_t* pixelPtr = (uint8_t*)image.data;
    int cn = image.channels();
    Scalar_<uint8_t> bgrPixel;

    for(int i = 0; i < image.rows; i++)
    {
        for(int j = 0; j < image.cols; j++)
        {
            pixelPtr[i*image.cols*cn + j*cn + 0]=255;
        }
    }

    imwrite("alpha.jpg", image);
    // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    // imshow( "Display window", image );                   // Show our image inside it.

    // waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
