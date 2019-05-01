// Compilation: g++ -fopenmp -O3 -march=native -std=c++11 painterly_rendering.cpp -o painterly_rendering `pkg-config --cflags --libs opencv`

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <cmath> 
#include <vector>
#include <limits>
#include <algorithm>
#include <string> 
#include "utils.h"
#include <omp.h>

using namespace cv;
using namespace std;

#define M_PI 3.14159265358979323846

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
Mat GKernel;
vector<Point> sobel_vec;
int style, numThreads; 
Timer t;
vector<double> times(5,0);

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

Point Gradient(Mat image, int x, int y) {
    Point v;
    v.x =  image.at<uchar>(y-1, x-1) + 2*image.at<uchar>(y, x-1) + image.at<uchar>(y+1, x-1) 
            - image.at<uchar>(y-1, x+1) - 2*image.at<uchar>(y, x+1) - image.at<uchar>(y+1, x+1); 
    v.y = image.at<uchar>(y-1, x-1) + 2*image.at<uchar>(y-1, x) + image.at<uchar>(y-1, x+1) 
            - image.at<uchar>(y+1, x-1) - 2*image.at<uchar>(y+1, x) - image.at<uchar>(y+1, x+1);
    return v;
}
void calculateSobelVectors(Mat ref){
    int gx, gy, sum;

    Mat ref_grey;
    cvtColor( ref, ref_grey, CV_BGR2GRAY );
#if 0       //disable parallelize since it make slower
    #pragma omp parallel
    {
        Point v;
        #pragma omp for nowait
        for(int x = 0; x < ref_grey.cols; x++){
            v.x=0;
            v.y=0;
            sobel_vec[x]=v;
            sobel_vec[(ref_grey.rows-1)*ref.cols+x]=v;
        }
        #pragma omp for
        for(int y = 0; y < ref_grey.rows; y++){
            v.x=0;
            v.y=0;
            sobel_vec[y*ref.cols]=v;
            sobel_vec[y*ref.cols+(ref_grey.cols-1)]=v;
        }
        #pragma omp for collapse(2)   
        for(int y = 1; y < ref_grey.rows - 1; y++){
            for(int x = 1; x < ref_grey.cols - 1; x++){
                sobel_vec[y*ref.cols+x]=Gradient(ref_grey, x, y);
            }
        }
    }
#endif
    Point v;
    for(int x = 0; x < ref_grey.cols; x++){
        v.x=0;
        v.y=0;
        sobel_vec[x]=v;
        sobel_vec[(ref_grey.rows-1)*ref.cols+x]=v;
    }
    for(int y = 0; y < ref_grey.rows; y++){
        v.x=0;
        v.y=0;
        sobel_vec[y*ref.cols]=v;
        sobel_vec[y*ref.cols+(ref_grey.cols-1)]=v;
    }
    for(int y = 1; y < ref_grey.rows - 1; y++){
        for(int x = 1; x < ref_grey.cols - 1; x++){
            sobel_vec[y*ref.cols+x]=Gradient(ref_grey, x, y);
        }
    }
}
void generateGaussian(double sigma) {
    GKernel = Mat::zeros(5, 5, CV_64F);
    double sum = 0.0, s = 2.0*sigma*sigma;
    //cout<<sigma<<":"<<endl;
    for (int x = -2; x <= 2; x++) { 
        for (int y = -2; y <= 2; y++) {
            double value = (exp(-(x*x+y*y) / s)) / (M_PI * s);
            GKernel.at<double>(x+2,y+2) =  value;
            sum+=value;
        } 
    } 
    for(int x=0; x<5; x++) {
        for(int y=0; y<5; y++) {
            GKernel.at<double>(x,y) = (GKernel.at<double>(x,y)/sum); 
        }
    }
}

void GaussianBlur_parallel(Mat src, Mat dst) {
   #pragma omp parallel for collapse(2) 
    for(int x = 0; x < src.rows; x++) {
        for( int y = 0; y < src.cols; y++) {
            double pix[3] = { 0, 0, 0 };
            for(int i=-2;i<=2;i++){
                for(int j=-2;j<=2;j++){
                    if(x+i<0 || x+i >=src.rows || y+j<0 || y+j>=src.cols)   continue;
                    Vec3b intensity = src.at<Vec3b>(x+i,y+j);
                    pix[0] += ((double)intensity[0])*GKernel.at<double>(i+2,j+2);
                    pix[1] += ((double)intensity[1])*GKernel.at<double>(i+2,j+2);
                    pix[2] += ((double)intensity[2])*GKernel.at<double>(i+2,j+2);
                }
            }
            dst.at<Vec3b>(x,y) = { (uchar)pix[0], (uchar)pix[1], (uchar)pix[2] };
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
        circle(canvas, K[0], cvRound((double)R/2.0), Scalar(Vec3b(strokeColor)), -1);
        circle(paintArea, K[0], cvRound((double)R/2.0), Scalar(Vec3b(255,255,255)), -1); // paint white stroke onto paintArea
    }
    else{
        polylines(canvas, K, false, Scalar(Vec3b(strokeColor)), R);
        polylines(paintArea, K, false, Scalar(Vec3b(255,255,255)), R); // paint white stroke onto paintArea
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
        point.x=(int)round(point.x+R*delta.x);
        point.y=(int)round(point.y+R*delta.y);
        if (point.x<0||point.x>=refImage.cols||point.y<0||point.y>=refImage.rows) {
            break;
        }
        lastD.x=delta.x;
        lastD.y=delta.y;
        K.push_back(point);
    }
    paintStroke(canvas,paintArea,K,strokeColor,R); // paint stroke onto canvas, and record drawn pixel with white color
}
void paintLayer(Mat canvas,Mat refImage,int brushRadius,Mat paintArea){
    vector<Point> S; // set of Strokes
    int grid=round(brushRadius*f_g); //grid size
    Mat Diff; // difference image
    absdiff(canvas,refImage,Diff);
    t.tic();
   // #pragma omp parallel for
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
                #pragma omp critical
                S.push_back(Point(largestX,largestY));
            }
        }
    }
    times[2] += t.toc();
    random_shuffle(S.begin(),S.end()); // shuffle all strokes to random order
//    printf("stroke count %d ", S.size());
    t.tic();
    #pragma omp parallel
    {
        while(S.size()!=0) {
            Point p0(-1,-1);
            #pragma omp critical
            {
                if(!S.empty()) {
                    p0 = S.back();
                    S.pop_back();
                }
            }
            if (p0.x<0||p0.x>=refImage.cols||p0.y<0||p0.y>=refImage.rows)   break;
            makeSplineStroke(canvas,p0,brushRadius,refImage,Diff,paintArea);
        }
    }
    times[3] += t.toc();
}

void Paint(Mat src){
    Mat refImage(src.rows, src.cols, CV_8UC3, Scalar(255,255,255));
    Mat canvas(src.rows, src.cols, CV_8UC3, Scalar(255,255,255));
    Mat paintArea(src.rows, src.cols, CV_8UC3, Scalar(0,0,0));  //record whether paint or not just using black/white color
    sobel_vec.resize(src.rows*src.cols);
    for(int Ri=maxRadius; Ri>1; Ri/=2){
        //GaussianBlur(src, refImage, Size(5,5), Ri*f_s, Ri*f_s);
        generateGaussian(Ri*f_s);
        t.tic();
        GaussianBlur_parallel(src, refImage);
        times[0] += t.toc();
        t.tic();
        calculateSobelVectors(refImage);
        times[1] += t.toc();
        paintLayer(canvas,refImage,Ri,paintArea);
        //printf("radius %d : blurring %10f, sobel %10f, strokpoint %10f, strokedetail %10f, drawstroke %10f\n\n", 
          //              Ri, times[0], times[1], times[2], times[3], times[4]);
        //times.clear();
    }
    printf("blurring %10f, sobel %10f, genstrokpoint %10f, strokedetail+draw %10f\n\n", 
                times[0], times[1], times[2], times[3]);
    imwrite("final_"+to_string(style)+"_"+to_string(numThreads)+".jpg", canvas);
}  

int main( int argc, char** argv )
{
    if( argc != 4)
    {
     cout <<" Usage: painterly_rendering <image_file> <paint_style:0~3> <num_thread>" << endl;
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
    numThreads = atoi(argv[3]); 
    omp_set_num_threads(numThreads);
    setStyle();
    Paint(image);

    return 0;
}
