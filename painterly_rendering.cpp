// Compilation: g++ -std=c++11 painterly_rendering.cpp -o painterly_rendering `pkg-config --cflags --libs opencv`

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <cmath> 
#include <vector>
#include <limits>
#include <algorithm>
#include <string> 

using namespace cv;
using namespace std;

/**** SET DEFAULT VALUE(IMPRESSIONIST) ****/
int T = 100; // Approximation threshold
float f_s = 0.5f; // f sigma, blur factor 
int maxRadius = 8; // Maximum Radius
float f_c = 1.0f; // curvature filter
float a = 1.0f; // Opacity
float f_g = 1.0f; // Grid Size
int minLength = 4; // Minimum stroke lengths 
int maxLength = 16; // Maximum stroke lengths
float j_h = 0.0f; // Color jitter factor H
float j_s = 0.0f; // Color jitter factor S
float j_v = 0.0f; // Color jitter factor V
float j_r = 0.0f; // Color jitter factor R
float j_g = 0.0f; // Color jitter factor G
float j_b = 0.0f; // Color jitter factor B

vector<Point> sobel_vec;
int style;
// const float SobelMatrixX[]={
//         1.0f, 0.0f, -1.0f,
//         2.0f, 0.0f , -2.0f,
//         1.0f, 0.0f, -1.0f	
// };

// const float SobelMatrixY[]={
//         1.0f, 2.0f, 1.0f,
//         0.0f, 0.0f, 0.0f,
//         -1.0f, -2.0f, -1.0f	
// };

void setStyle(){
    if(style==0) // type 0 = default (impressionist)
        return;
    if(style==1){ // type 1 = Expressionist
        T = 50;
        f_c = 0.25f;
        a = 0.7f;
        minLength = 16;
        j_v = 0.5f;
    }
    else if(style==2){ // type 2 = Colorist Wash
        T = 200;
        a = 0.5f;
        j_r = 0.3f;
        j_g = 0.3f;
        j_b = 0.3f;
    }
    else if(style==3){ // type 3 = Pointillist
        T = 100;
        maxRadius = 4;
        f_g = 0.5f;
        minLength = 0;
        maxLength = 0;
        j_v = 1.0f;
        j_h = 0.3f;
    }
}

float gradientMag(int x,int y, int width){
	Point G=sobel_vec[y*width+x];
	return (float)sqrt((double)G.x*G.x+G.y*G.y);
}

Point2f gradientDirection(int x,int y, int width){
    Point G=sobel_vec[y*width+x];
    Point2f direction;
    direction.x=G.x/gradientMag(x,y,width);
    direction.y=G.y/gradientMag(x,y,width);
    return direction;
}

int xGradient(Mat image, int x, int y)
{
    return  image.at<uchar>(y-1, x-1) +
            2*image.at<uchar>(y, x-1) +
            image.at<uchar>(y+1, x-1) -
            image.at<uchar>(y-1, x+1) -
            2*image.at<uchar>(y, x+1) -
            image.at<uchar>(y+1, x+1);
}
 
int yGradient(Mat image, int x, int y)
{
    return  image.at<uchar>(y-1, x-1) +
            2*image.at<uchar>(y-1, x) +
            image.at<uchar>(y-1, x+1) -
            image.at<uchar>(y+1, x-1) -
            2*image.at<uchar>(y+1, x) -
            image.at<uchar>(y+1, x+1);
}

void calculateSobelVectors(Mat ref){
    int gx, gy, sum;

    Mat ref_grey;
    cvtColor( ref, ref_grey, CV_BGR2GRAY );

    Point v;
    for(int y = 0; y < ref_grey.rows; y++){
        for(int x = 0; x < ref_grey.cols; x++){
            v.x=0;
            v.y=0;
            sobel_vec[y*ref.cols+x]=v;
        }
    }

    for(int y = 1; y < ref_grey.rows - 1; y++){
        for(int x = 1; x < ref_grey.cols - 1; x++){
            gx = xGradient(ref_grey, x, y);
            v.x=gx;
            gy = yGradient(ref_grey, x, y);
            v.y=gy;
            sobel_vec[y*ref.cols+x]=v;
        }
    }
}

// get the euclidean distance between two color
float colorabsDiff(Vec3b a, Vec3b b){
    uchar diff_b = abs(b.val[0]-a.val[0]);
    uchar diff_g = abs(b.val[1]-a.val[1]);
    uchar diff_r = abs(b.val[2]-a.val[2]);
    return (float)sqrt((double)(diff_b*diff_b+diff_g*diff_g+diff_r*diff_r));
}

// get the euclidean distance at point (i,j)
float pointDiff(int i,int j, Mat Diff, Mat paintArea){
    // check whether this point has been painted or not
    if(paintArea.at<Vec3b>(j,i)==Vec3b(0,0,0)){
        return numeric_limits<float>::max();
    }
    Vec3b diff = Diff.at<Vec3b>(j, i);
    uchar diff_b = diff.val[0];
    uchar diff_g = diff.val[1];
    uchar diff_r = diff.val[2];
    return (float)sqrt((double)(diff_b*diff_b+diff_g*diff_g+diff_r*diff_r));
}

void paintStroke(Mat canvas,Mat paintArea,vector<Point> K,Vec3b strokeColor, int R){
    if(K.size()==1){ // if there is one point in K, then draw a point
        Mat temp_c = canvas.clone();
        circle(temp_c, K[0], cvRound((double)R/2.0), Scalar(Vec3b(strokeColor)), -1);
        addWeighted(temp_c,a,canvas,1-a,0,canvas); // alpha blending because openCV polylines does not support alpha channel
        circle(paintArea, K[0], cvRound((double)R/2.0), Scalar(Vec3b(255,255,255)), -1); // paint white stroke onto paintArea
    }
    else{
        Mat temp_c = canvas.clone();
        polylines(temp_c, K, false, Scalar(Vec3b(strokeColor)), R, CV_AA);
        addWeighted(temp_c,a,canvas,1-a,0,canvas); // alpha blending because openCV polylines does not support alpha channel
        polylines(paintArea, K, false, Scalar(Vec3b(255,255,255)), R, CV_AA); // paint white stroke onto paintArea
    }
}

Vec3b setStrokeColor(Vec3b originalColor){
    Vec3b jColor(originalColor);
    float random;
    if(style == 0)
        return jColor;
    if(style == 1 || style == 3){ // HSV jitter
        Mat3b bgr(jColor);
        Mat3b hsv;
        cvtColor(bgr,hsv,CV_BGR2HSV);
        int h = hsv.at<uchar>(0);
        int s = hsv.at<uchar>(1);
        int v = hsv.at<uchar>(2);
        random = ((float) rand()) / (float) RAND_MAX;
        h += (int)((random-0.5f)*j_h*255);
        h = h<0?0:h;
        h = h>255?255:h;
        random = ((float) rand()) / (float) RAND_MAX;
        s += (int)((random-0.5f)*j_s*255);
        s = s<0?0:s;
        s = s>255?255:s;
        random = ((float) rand()) / (float) RAND_MAX;
        v += (int)((random-0.5f)*j_v*255);
        v = v<0?0:v;
        v = v>255?255:v;
        Mat3b jhsv(Vec3b(h,s,v));
        Mat3b jbgr;
        cvtColor(jhsv,jbgr,CV_HSV2BGR);
        jColor = jbgr.at<Vec3b>(0);
    }
    if(style == 2){ // RGB jitter
        int b=jColor.val[0];
        int g=jColor.val[1];
        int r=jColor.val[2];
        random = ((float) rand()) / (float) RAND_MAX;
        b += (int)((random-0.5f)*j_b*255);
        b = b<0?0:b;
        b = b>255?255:b;
        random = ((float) rand()) / (float) RAND_MAX;
        g += (int)((random-0.5f)*j_g*255);
        g = g<0?0:g;
        g = g>255?255:g;
        random = ((float) rand()) / (float) RAND_MAX;
        r += (int)((random-0.5f)*j_r*255);
        r = r<0?0:r;
        r = r>255?255:r;
        jColor = Vec3b(b,g,r);
    }
    return jColor;
}

void makeSplineStroke(Mat canvas,Point p0,int R, Mat refImage, Mat Diff, Mat paintArea){
    Vec3b strokeColor = setStrokeColor(refImage.at<Vec3b>(p0));
    vector<Point> K; //a new stroke with radius R and color strokeColor
    K.push_back(p0); 
    Point point = p0;
    Point2f lastD(0.0,0.0);
    for(int i=1; i<maxLength; i++){
        if(i>minLength && pointDiff(point.x,point.y,Diff,paintArea)<colorabsDiff(refImage.at<Vec3b>(point),strokeColor))
            break;
        if(gradientMag(point.x,point.y,refImage.cols)==0)
            break;
        Point2f gd = gradientDirection(point.x,point.y,refImage.cols); // get unit vector of gradient
        Point2f delta(-gd.y,gd.x); // compute a normal direction
        // if necessary, reverse direction
        if(lastD.x*delta.x+lastD.y*delta.y<0){
            delta.x = -delta.x;
            delta.y = -delta.y; 
        }
        // filter the stroke direction
        delta.x=f_c*delta.x+(1-f_c)*lastD.x;
        delta.y=f_c*delta.y+(1-f_c)*lastD.y;
        float temp=(float)sqrt((double)delta.x*delta.x+delta.y*delta.y);
        delta.x=delta.x/temp;
        delta.y=delta.y/temp;
        point.x=(int)(point.x+R*delta.x);
        point.y=(int)(point.y+R*delta.y);
        if (point.x<0||point.x>=refImage.cols||point.y<0||point.y>=refImage.rows) {
            break;
        }
        lastD.x=delta.x;
        lastD.y=delta.y;
        K.push_back(point);
    }
    paintStroke(canvas,paintArea,K,strokeColor,R); // paint stroke onto canvas, and record drawn pixel with white color
    // paintStroke(paintArea,K,Vec3b(255,255,255),R); 
}


void paintLayer(Mat canvas,Mat refImage,int brushRadius,Mat paintArea){
    vector<Point> S; // set of Strokes
    int grid=round(brushRadius*f_g); //grid size
    Mat Diff; // difference image
    absdiff(canvas,refImage,Diff);
    for(int x=0;x<canvas.cols;x+=grid){
        for(int y=0;y<canvas.rows;y+=grid){
            float largestError = numeric_limits<float>::min();
            int largestX=x,largestY=y;
            float areaError=0;
            // sum the error near (x,y)
            for(int y2=y-grid/2;y2<y+grid/2;y2++){
                for(int x2=x-grid/2;x2<x+grid/2;x2++){
                    int i=(x2<0)?0:x2;
                    int j=(y2<0)?0:y2;
                    i=(i>=canvas.cols)?canvas.cols-1:i;
                    j=(j>=canvas.rows)?canvas.rows-1:j;
                    float err = pointDiff(i,j,Diff,paintArea);
                    if(err>largestError){
                        largestError=err;
                        largestX=i;
                        largestY=j;
                    }
                    areaError+=err;
                }
            }
            areaError=areaError/(float)(grid*grid);
            if(areaError>T){
                S.push_back(Point(largestX,largestY));
            }
        }
    }
    random_shuffle(S.begin(),S.end()); // shuffle all strokes to random order
    while(S.size()!=0){
        Point p0 = S.back();
        S.pop_back();
        makeSplineStroke(canvas,p0,brushRadius,refImage,Diff,paintArea);
    }
}

// void calculateSobelVectors(Mat ref, Point* sobel_vec){
//     Mat ref_grey;
//     cvtColor( ref, ref_grey, CV_BGR2GRAY );
//     for(int y=0;y<ref_grey.rows;y++){
//         for(int x=0;x<ref_grey.cols;x++){
//             Point v;
//             int value=0;
//             for(int i=-1;i<=1;i++){
//                 for(int j=-1;j<=1;j++){
//                     if(y+j>=0 && y+j<ref_grey.rows && x+i>=0 && x+i<ref_grey.cols){
//                         uchar intensity = ref.at<uchar>(y+j, x+i); 
//                         value+=intensity*SobelMatrixX[(j+1)*3+(i+1)]; 
//                     }  
//                 }
//             }
//             v.x=value;
//             value=0;
//             for(int i=-1;i<=1;i++){
//                 for(int j=-1;j<=1;j++){
//                     if(y+j>=0 && y+j<ref_grey.rows && x+i>=0 && x+i<ref_grey.cols){
//                         uchar intensity = ref.at<uchar>(y+j, x+i); 
//                         value+=intensity*SobelMatrixY[(j+1)*3+(i+1)]; 
//                     }  
//                 }
//             }
//             v.y=value;
//             sobel_vec[y*ref.cols+x]=v;
//         }
//     }
// }

void Paint(Mat src){
    Mat refImage(src.rows, src.cols, CV_8UC3, Scalar(255,255,255));
    Mat canvas(src.rows, src.cols, CV_8UC3, Scalar(255,255,255));
    Mat paintArea(src.rows, src.cols, CV_8UC3, Scalar(0,0,0));  //record whether paint or not just using black/white color
    // Point sobelvector[src.rows*src.cols];
    sobel_vec.resize(src.rows*src.cols);
    for(int Ri=maxRadius; Ri>1; Ri/=2){
        GaussianBlur(src, refImage, Size(0,0), Ri*f_s, Ri*f_s);
        calculateSobelVectors(refImage);
        // Mat sobImage(src.rows, src.cols, CV_8UC1);
        // for(int x=0;x<src.cols;x++){
        //     for(int y=0;y<src.rows;y++){
        //         float mag=gradientMag(x,y,src.cols);
        //         sobImage.at<uchar>(y, x) = min((int)round(mag),255);
        //     }
        // }
        // imwrite("sobtest.jpg", sobImage);
        paintLayer(canvas,refImage,Ri,paintArea);
    }
    imwrite("final_"+to_string(style)+".jpg", canvas);
}  

int main( int argc, char** argv )
{
    if( argc != 3)
    {
     cout <<" Usage: painterly_rendering <image_file> <paint_style:0~3>" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    style = atoi(argv[2]);
    setStyle();
    Paint(image);

    return 0;
}