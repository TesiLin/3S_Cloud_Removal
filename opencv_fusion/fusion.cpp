#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/stitcher.hpp>
using namespace cv;
using namespace std;


bool try_use_gpu = false;
vector<Mat> imgs;
string result_name = "dst1.jpg";
int main(int argc, char* argv[])
{
    Mat img1 = imread("34.jpg");
    Mat img2 = imread("35.jpg");
    Mat img3 = imread("36.jpg");
    Mat img4 = imread("37.jpg");


    if (img1.empty() || img2.empty()|| img3.empty() || img4.empty())
    {
        cout << "Can't read image" << endl;
        return -1;
    }
    imgs.push_back(img1);
    imgs.push_back(img2);
    imgs.push_back(img3);
    imgs.push_back(img4);

    Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
    // 使用stitch函数进行拼接
    Mat pano;
    Stitcher::Status status = stitcher.stitch(imgs, pano);
    if (status != Stitcher::OK)
    {
        cout << "Can't stitch images, error code = " << int(status) << endl;
        return -1;
    }
    imwrite(result_name, pano);
    Mat pano2 = pano.clone();
    // 显示源图像，和结果图像
    imshow("全景图像", pano);
    if (waitKey() == 27)
        return 0;
}