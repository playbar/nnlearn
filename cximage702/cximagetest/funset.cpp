#include "funset.h"
#include <iostream>
#include "ximagif.h"
#include <io.h>

using namespace std;

void decoding_gif(string strGifName, string strSavePath)
{
	CxImage img;

	img.Load(strGifName.c_str(), CXIMAGE_FORMAT_GIF);

	int iNumFrames = img.GetNumFrames();
	cout<<"frames num = "<<iNumFrames<<endl;

	CxImage* newImage = new CxImage();

	for (int i = 0; i < iNumFrames; i++) {
		newImage->SetFrame(i);
		newImage->Load(strGifName.c_str(), CXIMAGE_FORMAT_GIF);

		char tmp[64];
		sprintf(tmp, "%d", i);

		string tmp1;
		tmp1 = tmp1.insert(0, tmp);

		tmp1 = strSavePath + tmp1 + ".png";

		newImage->Save(tmp1.c_str(), CXIMAGE_FORMAT_PNG);
	}

	if (newImage) delete newImage;
}

int TraverseFolder(const string strFilePath, string strImageNameSets[])
{
	int iImageCount=0;

	_finddata_t fileInfo;

	long handle = _findfirst(strFilePath.c_str(), &fileInfo);

	if (handle == -1L) {
		cerr << "failed to transfer files" << endl;
		return -1;
	}

	do {
		//cout << fileInfo.name <<endl;
		strImageNameSets[iImageCount] = (string)fileInfo.name;

		iImageCount ++;

	} while (_findnext(handle, &fileInfo) == 0);

	return iImageCount;
}

void encoding_gif(string strImgPath, string strGifName)
{
	string strImgSets[100] = {};

	int iImgCount = TraverseFolder(strImgPath, strImgSets);

	string strTmp = strImgPath.substr(0, strImgPath.find_last_of("/") +1);

	CxImage** img = new CxImage*[iImgCount];
	if (img == NULL) {
		cout<<"new Cximage error!"<<endl;
		return;
	}

	for (int i = 0; i < iImgCount; i++) {
		string tmp1;
		tmp1 = strTmp + strImgSets[i];
		img[i] = new CxImage;
		img[i]->Load(tmp1.c_str(), CXIMAGE_FORMAT_PNG);
	}

	CxIOFile hFile;
	hFile.Open(strGifName.c_str(), "wb");

	CxImageGIF multiimage;

	multiimage.SetLoops(3);
	multiimage.SetDisposalMethod(2);
	multiimage.Encode(&hFile, img, iImgCount, false, false);

	hFile.Close();

	delete [] img;
}