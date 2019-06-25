#include <iostream>
#include "CImg.h"
using namespace cimg_library;

void test()
{
    CImg<unsigned char> image("lena.jpg"), visu(500,400,1,3,0);
    const unsigned char red[] = { 255,0,0 }, green[] = { 0,255,0 }, blue[] = { 0,0,255 };
    image.blur(2.5);
    CImgDisplay main_disp(image,"Click a point"), draw_disp(visu,"Intensity profile");
    while (!main_disp.is_closed && !draw_disp.is_closed)
    {
      main_disp.wait();
      if (main_disp.button && main_disp.mouse_y>=0)
       {
        const int y = main_disp.mouse_y;
        visu.fill(0).draw_graph(image.get_crop(0,y,0,0,image.dimx()-1,y,0,0),red,1,255,0);
        visu.draw_graph(image.get_crop(0,y,0,1,image.dimx()-1,y,0,1),green,1,255,0);
        visu.draw_graph(image.get_crop(0,y,0,2,image.dimx()-1,y,0,2),blue,1,255,0).display(draw_disp);
        }
      }
}

void test1()
{
	// 定义一个每个颜色 8 位(bit)的 640x400 的彩色图像
	CImg<unsigned char> img(640,400,1,3);
	//将像素值设为 0（黑色）
	img.fill(0);
	// 定义一个紫色
	unsigned char purple[] = { 255,0,255 };

	// 在坐标(100, 100)处画一个紫色的“Hello world”
	img.draw_text(100,100,"Hello World",purple);
	// 在一个标题为“My first CImg code”的窗口中显示这幅图像
	img.display("My first CImg code");

	std::cout<<"ok!"<<std::endl;
}

int main(int argc, char* argv[])
{
    test();
	return 0;
}

