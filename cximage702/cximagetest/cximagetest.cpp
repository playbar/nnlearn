// cximagetest.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include "funset.h"
#include "ximage.h"

using namespace std;

void test_gif();

int main(int argc, char* argv[])
{
	test_gif();

	cout<<"ok!!!"<<endl;

	return 0;
}

void test_gif()
{
	string strGifName = "../../Data/fire.gif";
	string strSavaPath = "../../Data/";

	decoding_gif(strGifName, strSavaPath);

	string strImgPath = "../../Data/*.png";
	strGifName = "../../Data/tmp.gif";

	encoding_gif(strImgPath, strGifName);
}