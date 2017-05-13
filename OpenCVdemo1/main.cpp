#define M_PI 3.14159265358979323846
//���Ҿ�ȷ��
#define cosPrecision  0.685
//С����������������
#define maxReduceArea 2500
//��ͼת����ͼƬ��С
#define imgThreshold  550
//�����εľ���
#define squarePrecision 5
#include <opencv2/opencv.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include<opencv2/imgproc/imgproc.hpp>  
#include <stdlib.h>
#include <stdio.h>
#include<string>
#include<sstream> 
#include<math.h>
#include<iostream>
using namespace cv;
using namespace std;
int cou = 6;//��ʼ�ļ�
int imgNumber = 7;//���ļ��ĸ���
string imgRootPath = "D:/����/workspace/ʶ���ά��/��ά��/";
string imgRootWritePath = "D:/����/workspace/opencv/�ز�/��ά�����/���/";
string imgRootSWritePath = "D:/����/workspace/opencv/�ز�/��ά�����/С����/";
string imgPath,writePath,writeSmallPath;
Mat src;
Mat  src_gray;
string str;//������ת����char
double fScale = 1;      //���ű���  
double fScale1 = 1;      //���ű���
vector<vector<Point> > smallRectCons;
vector<Point2f> point_all;
vector<vector<Point> > contours;
vector<vector<Point> > contours2;
vector<Vec4i> hierarchy;
void init();
string itos(int i);  // ��int ת����string 
void getRectHier5(Mat img);//��δ���5�ľ��ο�
int getArea(RotatedRect rect);//��þ������
void showSmallRect(Mat img);//��ʾ���е�С����
bool isCurrentSmallRect(RotatedRect rect1, RotatedRect rect2, RotatedRect rect3);//�ж��Ƿ�Ϊ��Ҫ��С����
RotatedRect getMaxLargeRect();
double getCosine(Point2f p1, Point2f p2, Point2f p3);//����p1����p2,p3������ֵ
void removeRect();//ȥ�����ž���
void text()
{
	Mat image(200, 200, CV_8UC3, Scalar(0));
	RotatedRect rRect = RotatedRect(Point2f(100, 100), Size2f(100, 100), 45);
	Point2f vertices[4];
	rRect.points(vertices);
	for (int i = 0; i < 4; i++)
		line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0));
	Rect brect = rRect.boundingRect();
	rectangle(image, brect, Scalar(255, 0, 0));
	imshow("rectangles", image);
	waitKey(0);
}
int main()
{

	CvMemStorage * storage = cvCreateMemStorage(0);
	
	while (cou++<imgNumber)
	{
		init();
		if (cou < 10)
		{
			writeSmallPath = imgRootSWritePath + "DSC_000" + itos(cou) + ".jpg";
			imgPath = imgRootPath + "DSC_000"+itos(cou)+".JPG";
			writePath = imgRootWritePath + "DSC_000" + itos(cou) + ".jpg";
		}
		else if (cou < 100)
		{
			imgPath = imgRootPath + "DSC_00" + itos(cou) + ".JPG";
			writeSmallPath = imgRootSWritePath + "DSC_00" + itos(cou) + ".jpg";
			writePath = imgRootWritePath + "DSC_00" + itos(cou) + ".jpg";
		}
		
		src = imread(imgPath);
		if (!src.data)
			return -1;
		if (src.rows > 900 || src.cols > 900)
		{
			int minNum = src.rows < src.cols ? src.rows : src.cols;
			double modNum = minNum / imgThreshold;
			fScale = 1 / modNum;
		}
		resize(src, src, Size(src.cols*fScale, src.rows*fScale));//����

		//imshow("��ԭʼͼ��Canny��Ե���", src);//��ʾԭʼͼ
									  //��4��ת��Ϊ�Ҷ�ͼ  
		cvtColor(src, src_gray, CV_RGB2GRAY);
		//��3��ʹ�ø�˹�˲���������  
		GaussianBlur(src_gray, src_gray, Size(3, 3), 0);
		//blur(src_gray, src_gray, Size(3, 3));
		Canny(src_gray, src_gray, 100, 250);
		//imshow("��˹ȥ��canny��Ե�㷨", src_gray);
		findContours(src_gray, contours, hierarchy, CV_RETR_TREE, CHAIN_APPROX_SIMPLE);
		getRectHier5(src);
		showSmallRect(src);
		if (contours2.size()>3)
			removeRect();
		getMaxLargeRect();
		cout << "*******************��ǰͼƬΪ: " << cou << "****************" << endl;
	}
	
	return 0;
}
void getRectHier5(Mat img)
{
	Mat imgCopy;
	img.copyTo(imgCopy);
	int k = 0, ic = 0, counter = 0;
	int parentIdx = -1;
	for (int i = 0; i < contours.size(); i++)
	{
		if (hierarchy[i][2] != -1 && ic == 0)
		{
			parentIdx = i;
			ic++;
		}
		else if (hierarchy[i][2] != -1)
		{
			ic++;
		}
		else if (hierarchy[i][2] == -1)
		{
			ic = 0;
			parentIdx = -1;
		}

		if (ic >= 3&&parentIdx!=-1)
		{
			contours2.push_back(contours[parentIdx]);
			//cout << parentIdx << " " << endl;
			/*
			drawContours(imgCopy, contours, parentIdx, (0, 0, 255), 3);
			imshow("С����", imgCopy);
			waitKey();
			*/
			//parentIdx = -1;
			counter++;
		}

	}
}
void showSmallRect(Mat img)
{
	Mat imgCopy;
	img.copyTo(imgCopy);
	for (int i = 0; i<contours2.size(); i++)
		drawContours(imgCopy, contours2, i, (0, 0, 255), 3);
	//imshow("С����", imgCopy);
	imwrite(writeSmallPath, imgCopy);
	//waitKey();
}
RotatedRect getMaxLargeRect()
{
	Mat imcopy;
	src.copyTo(imcopy);
	//��С��Χ��
	RotatedRect rect, rectResult;
	Point2f vertices[4];
	for (int i = 0; i < smallRectCons.size(); i++)
	{
		rect = minAreaRect(smallRectCons[i]);
		rect.points(vertices);
		for (int j = 0; j<4; j++)
			point_all.push_back(vertices[j]);
	}
	if (smallRectCons.size() >= 2)
	{
		rectResult = minAreaRect(point_all);
		rectResult.points(vertices);
		for (int i = 0; i < 4; i++)
			line(imcopy, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 3);
		imshow("", imcopy);
		waitKey();
		imwrite(writePath, imcopy);
	}
	return rectResult;
}
bool isCurrentSmallRect(RotatedRect rect1, RotatedRect rect2, RotatedRect rect3)
{
	double cosValue1, cosValue2, cosValue = getCosine(rect1.center, rect2.center, rect3.center);
	
	if (abs(rect1.size.height - rect1.size.width) > squarePrecision||
		abs(rect2.size.height - rect2.size.width) > squarePrecision|| 
		abs(rect3.size.height - rect3.size.width) > squarePrecision)
		return false;//�����һ���������������˳�
	else if (abs(getArea(rect1) - getArea(rect2))>maxReduceArea || abs(getArea(rect1) - getArea(rect3))>maxReduceArea)
		return false;//����������������󣬲�����С����
	else if (abs(abs(cosValue) - cos(M_PI / 2)) < cosPrecision)
	{
		cosValue1 = getCosine(rect2.center, rect1.center, rect3.center);
		if (abs(abs(cosValue1) - cos(M_PI / 4)) <  cosPrecision)
			return true;
	}
	else if (abs(abs(cosValue) - cos(M_PI / 4)) <  cosPrecision)
	{
		cosValue1 = getCosine(rect2.center, rect1.center, rect3.center);
		cosValue2 = getCosine(rect3.center, rect1.center, rect2.center);
		if (abs(abs(cosValue1) - cos(M_PI / 4)) <  cosPrecision || abs(abs(cosValue2) - cos(M_PI / 2)) < cosPrecision)
			return true;
	}
	return false;
}
void removeRect()
{
	RotatedRect rect1, rect2, rect3;
	int flag;
	//��������,ÿ���������������������
	for (int i = 0; i < contours2.size(); i++)
	{
		flag = 0;
		for (int j = i + 1; j < contours2.size() + i; j = j + 1)
		{
			for (int z = j + 1; z < contours2.size() + i; z++)
			{
				rect1 = minAreaRect(contours2[i]);
				rect2 = minAreaRect(contours2[j%contours2.size()]);
				rect3 = minAreaRect(contours2[z%contours2.size()]);
				if (isCurrentSmallRect(rect1, rect2, rect3))
				{
					smallRectCons.push_back(contours2[i]);
					smallRectCons.push_back(contours2[j%contours2.size()]);
					smallRectCons.push_back(contours2[z%contours2.size()]);
					contours2.erase(contours2.begin() + i);
					contours2.erase(contours2.begin() + j%contours2.size());
					contours2.erase(contours2.begin() + z%contours2.size());
					flag = 1;
					break;
				}
			}
			if (flag)
				break;
		}
	}
	if (contours2.size() > 0&&smallRectCons.size()>1)
	{
		for (int i = 0; i < contours2.size(); i++)
		{
			rect1 = minAreaRect(contours2[i]);
			rect2 = minAreaRect(smallRectCons[0]);
			rect3 = minAreaRect(smallRectCons[1]);
			if (isCurrentSmallRect(rect1, rect2, rect3))
			{
				smallRectCons.push_back(contours2[i]);
				//contours2.erase(contours2.begin() + i);
			}
		}
	}
}
double getCosine(Point2f p1, Point2f p2, Point2f p3)
{
	double x1, x2, y1, y2;
	//��ȡ�����ߵ��������꣬��������Ϊp1
	x1 = p2.x - p1.x;
	x2 = p3.x - p1.x;
	y1 = p2.y - p1.y;
	y2 = p3.y - p1.y;
	//����������
	double cosine = (x1*x2 + y1*y2) / (sqrt(x1*x1 + y1*y1) * sqrt(x2*x2 + y2*y2) + 1e-10);
	return (x1*x2 + y1*y2) / (sqrt(x1*x1 + y1*y1) * sqrt(x2*x2 + y2*y2) + 1e-10);
}
int getArea(RotatedRect rect)
{
	return rect.size.height*rect.size.width;
}
void init()
{
	smallRectCons.clear();
	point_all.clear();
	contours.clear();
	contours2.clear();
	hierarchy.clear();
}
string itos(int i) // ��int ת����string 
{
	stringstream s;
	s.clear();
	s << i;
	return s.str();
}
