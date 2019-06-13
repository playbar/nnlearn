#include <iostream>
#include <string>

#include "simd/SimdSse2.h"
#include "simd/SimdBase.h"
#include <opencv2/opencv.hpp>

using namespace std;

void BgraToGrayTest()
{
	string strImageName = "./testdata/cat.jpg";
	int iImageWidth = 10000;
	int iImageHeight = 10000;

	cv::Mat matSrc = cv::imread(strImageName, 1);
	cv::cvtColor(matSrc, matSrc, cv::COLOR_BGR2BGRA);
	cv::resize(matSrc, matSrc, cv::Size(iImageWidth, iImageHeight), 0, 0, 1);

	cv::Mat matDst1, matDst2;
	matDst1 = cv::Mat::zeros(iImageHeight, iImageWidth, CV_8UC1);
	matDst2 = cv::Mat::zeros(iImageHeight, iImageWidth, CV_8UC1);

	int iRemainder = iImageWidth & 0x03;
	int iGrayStride =  iRemainder ? iImageWidth + 4 - iRemainder : iImageWidth;
	CV_Assert(iRemainder == 0);

	double dTimeC = cv::getTickCount();
	Simd::Base::BgraToGray(matSrc.data, iImageWidth, iImageHeight, iImageWidth * 4, matDst1.data, iGrayStride);
	dTimeC = ((double)cv::getTickCount() - dTimeC) / cv::getTickFrequency();

	double dTimeSimd = cv::getTickCount();
	Simd::Sse2::BgraToGray(matSrc.data, iImageWidth, iImageHeight, iImageWidth * 4, matDst2.data, iGrayStride);
	dTimeSimd = ((double)cv::getTickCount() - dTimeSimd) / cv::getTickFrequency();

	cout<<"C run time : "<<dTimeC<<endl;
	cout<<"Simd run time : "<<dTimeSimd<<endl;

	int iDiffCount = 0;

	for (int i = 0; i < iImageHeight; i++) {
		uchar* p1 = matDst1.ptr<uchar>(i);
		uchar* p2 = matDst2.ptr<uchar>(i);

		for (int j = 0; j < iImageWidth; j++) {
			if (p1[j] != p2[j])
				iDiffCount ++;
		}	
	}

	cout<<"the different count: "<<iDiffCount<<endl;
}

int main(int argc, char* argv[])
{
	BgraToGrayTest();

	cout<<"ok!"<<endl;
	return 0;
}

