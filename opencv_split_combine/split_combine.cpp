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
ͼ���з֣����ǰ��н����зֵģ�������Ҳ��ͬ����ԭ������ǰ��飩
���Ĵ�������:
*/
//���ڴ洢�зֺ��Сͼ��
vector<Mat> imgs;
int spCols, spRows;

//src:���з�ԭͼ�� splitCols:�зֵ�ÿ��Сͼ������ splitRows:�зֵ�ÿ��Сͼ������
void imgSplit(Mat src, int splitCols, int splitRows)
{
	spCols = splitCols;
	spRows = splitRows;
	//���÷ָ��ͼ��洢·��
	string outpath = ".\\split1\\";
	int col = src.cols, row = src.rows;
	//�зֺ�ͼ������
	int sum = 0;
	int colnum = 0;//���ֺ��ж�������ͼ
	int rownum = 0;
	colnum = col / splitCols;
	rownum = row / splitRows;
	//������
	if ((col % splitCols == 0)&&(row%splitRows==0))
	{		
		sum = colnum * rownum;
		//������ceil_img�洢��ͼ��
		//vector<Mat> ceil_img;
		//������name�洢��ͼ������֣���0��sum-1
		vector<int> name;
		for (int i = 0; i < sum; i++)
		{
			name.push_back(i);
		}
		Mat image_cut, roi_img, tim_img;
		//�洢����ͼ��
		for (int i = 0; i < rownum; i++)
			for (int j = 0; j < colnum ; j++)
			{
				Rect rect(j * splitCols, i*splitRows, splitCols, splitRows);//���Ͻǵ������;��εĿ�͸�
				image_cut = Mat(src, rect);//��rectȦ����һ������
				roi_img = image_cut.clone();//clone�����ROI, Region of Interest
				imgs.push_back(roi_img);
			}
		//д�뵽ָ���ļ���
		for (int i = 0; i < sum; i++)
		{
			imwrite(outpath + to_string(long long((name[i]))) + ".jpg", imgs[i]);
		}
	}
	else if((col % splitCols != 0) && (row % splitRows == 0)) //�в�������
	{
		colnum += 1;
		sum = colnum * rownum;
		//������ceil_img�洢��ͼ��
		//vector<Mat> ceil_img;
		//������name�洢��ͼ������֣���0��sum-1
		vector<int> name;
		for (int i = 0; i < sum; i++)
		{
			name.push_back(i);
		}
		Mat image_cut, roi_img, tim_img;
		//�洢����ͼ��
		for (int i = 0; i < rownum ; i++)
		{
			for (int j = 0; j < colnum - 1; j++)
			{
				Rect rect(j * splitCols, i*splitRows, splitCols, splitRows);
				image_cut = Mat(src, rect);
				roi_img = image_cut.clone();
				imgs.push_back(roi_img);
			}
			
			//����ÿ�����һ��ͼƬ
			//����ͼ��(���������ˣ�
			Rect rect(col - splitCols, i * splitRows, splitCols, splitRows);
			image_cut = Mat(src, rect);
			roi_img = image_cut.clone();
			imgs.push_back(roi_img);
		}
		
		//д�뵽ָ���ļ���
		for (int i = 0; i < sum; i++)
		{
			imwrite(outpath + to_string(long long((name[i]))) + ".jpg", imgs[i]);
		}
	}
	else if ((col % splitCols == 0) && (row % splitRows != 0)) //�в�������
	{
		rownum += 1;
		sum = colnum * rownum;
		//������ceil_img�洢��ͼ��
		//vector<Mat> ceil_img;
		//������name�洢��ͼ������֣���0��sum-1
		vector<int> name;
		for (int i = 0; i < sum; i++)
		{
			name.push_back(i);
		}
		Mat image_cut, roi_img, tim_img;
		//�洢����ͼ��
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
		//�������һ�У��������ϵ��ˣ�
		for (int t = 0; t < colnum ; t++)
		{
			Rect rect(t * splitCols, row-splitRows, splitCols, splitRows);
			image_cut = Mat(src, rect);
			roi_img = image_cut.clone();
			imgs.push_back(roi_img);
		}

		//д�뵽ָ���ļ���
		for (int i = 0; i < sum; i++)
		{
			imwrite(outpath + to_string(long long((name[i]))) + ".jpg", imgs[i]);
		}
	}
	else if ((col % splitCols != 0) && (row % splitRows != 0)) //���в�������
	{
		colnum += 1;
		rownum += 1;
		sum = colnum * rownum;
		//������ceil_img�洢��ͼ��
		//vector<Mat> ceil_img;
		//������name�洢��ͼ������֣���0��sum-1
		vector<int> name;
		for (int i = 0; i < sum; i++)
		{
			name.push_back(i);
		}
		Mat image_cut, roi_img, tim_img;
		//�洢����ͼ��
		for (int i = 0; i < rownum - 1; i++)
		{
			for (int j = 0; j < colnum - 1; j++)
			{
				Rect rect(j * splitCols, i * splitRows, splitCols, splitRows);
				image_cut = Mat(src, rect);
				roi_img = image_cut.clone();
				imgs.push_back(roi_img);
			}
			//����ÿ�����һ��ͼƬ
			//����ͼ��(���������ˣ�
			Rect rect(col - splitCols, i * splitRows, splitCols, splitRows);
			image_cut = Mat(src, rect);
			roi_img = image_cut.clone();
			imgs.push_back(roi_img);

		}
		//�������һ�У��������ϵ��ˣ�
		for (int t = 0; t < colnum-1 ; t++)
		{
			Rect rect(t * splitCols, row - splitRows, splitCols, splitRows);
			image_cut = Mat(src, rect);
			roi_img = image_cut.clone();
			imgs.push_back(roi_img);
		}
		//�������һ�����һ�е�ͼ
		Rect rect(col - splitCols, row - splitRows, splitCols, splitRows);
		image_cut = Mat(src, rect);
		roi_img = image_cut.clone();
		imgs.push_back(roi_img);

		//д�뵽ָ���ļ���
		for (int i = 0; i < sum; i++)
		{
			imwrite(outpath + to_string(long long((name[i]))) + ".jpg", imgs[i]);
		}
	}
}

/*
ͼ��ϲ�
*/
//���кϲ�����ͼ��,Ĭ����һ��
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

//���кϲ�����ͼ��,Ĭ����һ��
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
//���ͼ��ϲ�
void imgMerge(Mat src)
{
	int col = src.cols, row = src.rows;

	int colnum = 0;//���ֺ��ж�������ͼ
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
	else if ((col % spCols != 0) && (row % spRows == 0))//�в�������
	{
		colnum += 1;
		for (int i = 0; i < rownum; i++)
		{
			dst = imgs[i * colnum];
			for (int j = 1; j < colnum-1; j++)
			{
				dst = mergeCols(dst, imgs[j + i * colnum]);
			}
			//����ÿ�е����һ��
			left = col - (col / spCols) * spCols;//Ҫ�����Ŀ�
			Rect rect (spCols-left, 0, left, spRows);//���rectҪ�͵�ͼ��Ӧ������ԭ���ǵ�ͼ���Ͻ�
			image_cut = Mat(imgs[colnum - 1 + i * colnum], rect);
			roi_img = image_cut.clone();
			dst=mergeCols(dst,roi_img);

			if (i == 0) rowdst = dst.clone();
			else rowdst = mergeRows(rowdst, dst);
		}
	}
	else if ((col % spCols == 0) && (row % spRows != 0))//�в�������
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

		//�������һ��
		left = row - (row / spRows) * spRows;//Ҫ�����ĸ�
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
			//����ÿ�е����һ��
			left = col - (col / spCols) * spCols;//Ҫ�����Ŀ�
			Rect rect(spCols - left, 0, left, spRows);//���rectҪ�͵�ͼ��Ӧ������ԭ���ǵ�ͼ���Ͻ�
			image_cut = Mat(imgs[colnum - 1 + i * colnum], rect);
			roi_img = image_cut.clone();
			dst = mergeCols(dst, roi_img);

			if (i == 0) rowdst = dst.clone();
			else rowdst = mergeRows(rowdst, dst);
		}

		//�������һ��
		int left1 = col - (col / spCols) * spCols;//Ҫ�����Ŀ�
		left = row - (row / spRows) * spRows;//Ҫ�����ĸ�
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
		//�������һ�����һ��
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

//�ص�����̫С��work�ˡ�����
bool try_use_gpu = false;
string result_name = "dst.jpg";
vector<Mat> stitchImg;
int imgMerge2()
{

	Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
	// ʹ��stitch��������ƴ��
	Mat pano;
	Stitcher::Status status = stitcher.stitch(imgs, pano);
	if (status != Stitcher::OK)
	{
		cout << "Can't stitch images, error code = " << int(status) << endl;
		return -1;
	}
	imwrite(result_name, pano);
	Mat pano2 = pano.clone();
	// ��ʾԴͼ�񣬺ͽ��ͼ��
	imshow("ȫ��ͼ��", pano);
	if (waitKey() == 27)
		return 0;
}



void split_mergeRun()
{
	Mat src = imread("image1.jpg");
	imgSplit(src, 100,100);
	imgMerge(src);
	//imgMerge2();//ֱ����stitcher����ƴ
}


int main()
{
	split_mergeRun();
	return 0;
}