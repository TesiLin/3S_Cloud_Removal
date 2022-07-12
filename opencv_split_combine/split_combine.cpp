#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

/*
图像切分（我是按列进行切分的，按照行也是同样的原理。亦或是按块）
核心代码如下:
*/
//用于存储切分后的小图像
vector<Mat> imgs;
int spCols, spRows;

//src:待切分原图像 splitCols:切分的每个小图像列数 splitRows:切分的每个小图像行数
void imgSplit(Mat src, int splitCols, int splitRows)
{
	spCols = splitCols;
	spRows = splitRows;
	//设置分割后图像存储路径
	string outpath = ".\\split1\\";
	int col = src.cols, row = src.rows;
	//切分后图像数量
	int sum = 0;
	int colnum = 0;//划分后有多少列子图
	int rownum = 0;
	colnum = col / splitCols;
	rownum = row / splitRows;
	//被整除
	if ((col % splitCols == 0)&&(row%splitRows==0))
	{		
		sum = colnum * rownum;
		//迭代器ceil_img存储子图像
		//vector<Mat> ceil_img;
		//迭代器name存储子图像的名字，从0到sum-1
		vector<int> name;
		for (int i = 0; i < sum; i++)
		{
			name.push_back(i);
		}
		Mat image_cut, roi_img, tim_img;
		//存储完整图像
		for (int i = 0; i < rownum; i++)
			for (int j = 0; j < colnum ; j++)
			{
				Rect rect(j * splitCols, i*splitRows, splitCols, splitRows);//左上角点的坐标和矩形的宽和高
				image_cut = Mat(src, rect);//用rect圈出了一块区域
				roi_img = image_cut.clone();//clone深拷贝；ROI, Region of Interest
				imgs.push_back(roi_img);
			}
		//写入到指定文件夹
		for (int i = 0; i < sum; i++)
		{
			imwrite(outpath + to_string(long long((name[i]))) + ".jpg", imgs[i]);
		}
	}
	else if((col % splitCols != 0) && (row % splitRows == 0)) //列不能整除
	{
		colnum += 1;
		sum = colnum * rownum;
		//迭代器ceil_img存储子图像
		//vector<Mat> ceil_img;
		//迭代器name存储子图像的名字，从0到sum-1
		vector<int> name;
		for (int i = 0; i < sum; i++)
		{
			name.push_back(i);
		}
		Mat image_cut, roi_img, tim_img;
		//存储完整图像
		for (int i = 0; i < rownum ; i++)
		{
			for (int j = 0; j < colnum - 1; j++)
			{
				Rect rect(j * splitCols, i*splitRows, splitCols, splitRows);
				image_cut = Mat(src, rect);
				roi_img = image_cut.clone();
				imgs.push_back(roi_img);
			}
			
			//处理每行最后一张图片
			//留余图像(从右往左倒退）
			Rect rect(col - splitCols, i * splitRows, splitCols, splitRows);
			image_cut = Mat(src, rect);
			roi_img = image_cut.clone();
			imgs.push_back(roi_img);
		}
		
		//写入到指定文件夹
		for (int i = 0; i < sum; i++)
		{
			imwrite(outpath + to_string(long long((name[i]))) + ".jpg", imgs[i]);
		}
	}
	else if ((col % splitCols == 0) && (row % splitRows != 0)) //行不能整除
	{
		rownum += 1;
		sum = colnum * rownum;
		//迭代器ceil_img存储子图像
		//vector<Mat> ceil_img;
		//迭代器name存储子图像的名字，从0到sum-1
		vector<int> name;
		for (int i = 0; i < sum; i++)
		{
			name.push_back(i);
		}
		Mat image_cut, roi_img, tim_img;
		//存储完整图像
		for (int i = 0; i < rownum - 1; i++)
		{
			for (int j = 0; j < colnum ; j++)
			{
				Rect rect(j * splitCols, i * splitRows, splitCols, splitRows);
				image_cut = Mat(src, rect);
				roi_img = image_cut.clone();
				imgs.push_back(roi_img);
			}
		}
		//处理最后一行（从下往上倒退）
		for (int t = 0; t < colnum ; t++)
		{
			Rect rect(t * splitCols, row-splitRows, splitCols, splitRows);
			image_cut = Mat(src, rect);
			roi_img = image_cut.clone();
			imgs.push_back(roi_img);
		}

		//写入到指定文件夹
		for (int i = 0; i < sum; i++)
		{
			imwrite(outpath + to_string(long long((name[i]))) + ".jpg", imgs[i]);
		}
	}
	else if ((col % splitCols != 0) && (row % splitRows != 0)) //行列不能整除
	{
		colnum += 1;
		rownum += 1;
		sum = colnum * rownum;
		//迭代器ceil_img存储子图像
		//vector<Mat> ceil_img;
		//迭代器name存储子图像的名字，从0到sum-1
		vector<int> name;
		for (int i = 0; i < sum; i++)
		{
			name.push_back(i);
		}
		Mat image_cut, roi_img, tim_img;
		//存储完整图像
		for (int i = 0; i < rownum - 1; i++)
		{
			for (int j = 0; j < colnum - 1; j++)
			{
				Rect rect(j * splitCols, i * splitRows, splitCols, splitRows);
				image_cut = Mat(src, rect);
				roi_img = image_cut.clone();
				imgs.push_back(roi_img);
			}
			//处理每行最后一张图片
			//留余图像(从右往左倒退）
			Rect rect(col - splitCols, i * splitRows, splitCols, splitRows);
			image_cut = Mat(src, rect);
			roi_img = image_cut.clone();
			imgs.push_back(roi_img);

		}
		//处理最后一行（从下往上倒退）
		for (int t = 0; t < colnum-1 ; t++)
		{
			Rect rect(t * splitCols, row - splitRows, splitCols, splitRows);
			image_cut = Mat(src, rect);
			roi_img = image_cut.clone();
			imgs.push_back(roi_img);
		}
		//处理最后一行最后一列的图
		Rect rect(col - splitCols, row - splitRows, splitCols, splitRows);
		image_cut = Mat(src, rect);
		roi_img = image_cut.clone();
		imgs.push_back(roi_img);

		//写入到指定文件夹
		for (int i = 0; i < sum; i++)
		{
			imwrite(outpath + to_string(long long((name[i]))) + ".jpg", imgs[i]);
		}
	}
}

/*
图像合并
*/
//按列合并两幅图像,默认行一样
Mat mergeCols(Mat src1, Mat src2)
{
	int totalCols = src1.cols + src2.cols;
	Mat dst(src1.rows, totalCols, src1.type());
	Mat submat = dst.colRange(0, src1.cols);
	src1.copyTo(submat);
	submat = dst.colRange(src1.cols, totalCols);
	src2.copyTo(submat);
	return dst;
}

//按行合并两幅图像,默认列一样
Mat mergeRows(Mat src1, Mat src2)
{
	int totalRows = src1.rows + src2.rows;
	Mat dst(totalRows,src1.cols, src1.type());
	Mat submat = dst.rowRange(0, src1.rows);
	src1.copyTo(submat);
	submat = dst.rowRange(src1.rows, totalRows);
	src2.copyTo(submat);
	return dst;
}
//多幅图像合并
void imgMerge(Mat src)
{
	int col = src.cols, row = src.rows;

	int colnum = 0;//划分后有多少列子图
	int rownum = 0;
	colnum = col / spCols;
	rownum = row / spRows;

	Mat dst;
	Mat rowdst;
	Mat image_cut;
	Mat roi_img;

	int left;

	if ((col % spCols == 0) && (row % spRows == 0))
	{
		for (int i = 0; i < rownum; i++)
		{
			dst = imgs[i*colnum];
			for(int j=1;j<colnum;j++)
			{ 
				dst = mergeCols(dst, imgs[j + i * colnum]);
			}
			if (i == 0) rowdst = dst.clone();
			else rowdst = mergeRows(rowdst, dst);
		}
	}
	else if ((col % spCols != 0) && (row % spRows == 0))//列不能整除
	{
		colnum += 1;
		for (int i = 0; i < rownum; i++)
		{
			dst = imgs[i * colnum];
			for (int j = 1; j < colnum-1; j++)
			{
				dst = mergeCols(dst, imgs[j + i * colnum]);
			}
			//处理每行的最后一张
			left = col - (col / spCols) * spCols;//要保留的宽
			Rect rect (spCols-left, 0, left, spRows);//这个rect要和底图对应，坐标原点是底图左上角
			image_cut = Mat(imgs[colnum - 1 + i * colnum], rect);
			roi_img = image_cut.clone();
			dst=mergeCols(dst,roi_img);

			if (i == 0) rowdst = dst.clone();
			else rowdst = mergeRows(rowdst, dst);
		}
	}
	else if ((col % spCols == 0) && (row % spRows != 0))//行不能整除
	{
		rownum += 1;
		for (int i = 0; i < rownum-1; i++)
		{
			dst = imgs[i * colnum];
			for (int j = 1; j < colnum; j++)
			{
				dst = mergeCols(dst, imgs[j + i * colnum]);
			}

			if (i == 0) rowdst = dst.clone();
			else rowdst = mergeRows(rowdst, dst);
		}

		//处理最后一行
		left = row - (row / spRows) * spRows;//要保留的高
		Rect rect(0, spRows - left, spCols, left);
		image_cut = Mat(imgs[colnum * (rownum - 1)], rect);
		dst = image_cut.clone();
		for (int t = 1; t < colnum; t++)
		{
			Rect rect(0, spRows- left, spCols, left);
			image_cut = Mat(imgs[colnum * (rownum-1) + t ], rect);
			roi_img = image_cut.clone();
			dst = mergeCols(dst, roi_img);
		}
		rowdst = mergeRows(rowdst, dst);
	}

	else if ((col % spCols != 0) && (row % spRows != 0))
	{
		colnum += 1;
		rownum += 1;
		for (int i = 0; i < rownum-1; i++)
		{
			dst = imgs[i * colnum];
			for (int j = 1; j < colnum - 1; j++)
			{
				dst = mergeCols(dst, imgs[j + i * colnum]);
			}
			//处理每行的最后一张
			left = col - (col / spCols) * spCols;//要保留的宽
			Rect rect(spCols - left, 0, left, spRows);//这个rect要和底图对应，坐标原点是底图左上角
			image_cut = Mat(imgs[colnum - 1 + i * colnum], rect);
			roi_img = image_cut.clone();
			dst = mergeCols(dst, roi_img);

			if (i == 0) rowdst = dst.clone();
			else rowdst = mergeRows(rowdst, dst);
		}

		//处理最后一行
		int left1 = col - (col / spCols) * spCols;//要保留的宽
		left = row - (row / spRows) * spRows;//要保留的高
		Rect rect(0, spRows - left, spCols, left);
		image_cut = Mat(imgs[colnum * (rownum - 1)], rect);
		dst = image_cut.clone();
		for (int t = 1; t < colnum-1; t++)
		{
			Rect rect(0, spRows - left, spCols, left);
			image_cut = Mat(imgs[colnum * (rownum - 1) + t], rect);
			roi_img = image_cut.clone();
			dst = mergeCols(dst, roi_img);
		}
		//处理最后一行最后一列
		Rect rect1( spCols-left1, spRows - left, left1, left);
		image_cut = Mat(imgs[colnum * rownum -1], rect1);
		roi_img = image_cut.clone();
		dst = mergeCols(dst, roi_img);

		rowdst = mergeRows(rowdst, dst);
	}

	//dst = imgs[0];
	//for (int i = 1; i < colnum * rownum; i++)
	//{
	//	dst = mergeRows(dst, imgs[i]);
	//}
	imwrite("image1_combine.jpg", rowdst);
}

//重叠区域太小不work了。。。
bool try_use_gpu = false;
string result_name = "dst.jpg";
vector<Mat> stitchImg;
int imgMerge2()
{

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



void split_mergeRun()
{
	Mat src = imread("image1.jpg");
	imgSplit(src, 100,100);
	imgMerge(src);
	//imgMerge2();//直接用stitcher类来拼
}


int main()
{
	split_mergeRun();
	return 0;
}