//
//

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;

int flann_point_matching()
{
//Mat img_1=imread("image0.jpg");
//Mat img_2=imread("image1.jpg");

//Mat img_1=imread("box.png");
//Mat img_2=imread("box_in_scene.png");

    Mat img_1=imread("m1.png");
    Mat img_2=imread("test1.png");

    if( !img_1.data || !img_2.data )
    {
        std::cout<< " --(!) Error reading images " << std::endl;
        return -1;
    }

//-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    SiftFeatureDetector detector( minHessian );
//    SurfFeatureDetector detector( minHessian );

    std::vector<KeyPoint> keypoints_1, keypoints_2;

    detector.detect( img_1, keypoints_1 );
    detector.detect( img_2, keypoints_2 );

//-- Step 2: Calculate descriptors (feature vectors)
    SiftDescriptorExtractor extractor;
//    SurfDescriptorExtractor extractor;

    Mat descriptors_1, descriptors_2;

    extractor.compute( img_1, keypoints_1, descriptors_1 );
    extractor.compute( img_2, keypoints_2, descriptors_2 );

//-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;

//-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist )
            min_dist = dist;
        if( dist > max_dist )
            max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
//-- PS.- radiusMatch can also be used here.
    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        if( matches[i].distance < 2*min_dist )
        {
            good_matches.push_back( matches[i]);
        }
    }

//-- Draw only "good" matches
    Mat img_matches;
    vector<char> matchesMask;
    drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 matchesMask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

//-- Show detected matches
    imshow( "Good Matches", img_matches );
//imwrite("Lena_match_surf.jpg",img_matches);
//imwrite("Lena_match_sift.jpg",img_matches);
    for( int i = 0; i < good_matches.size(); i++ )
//{
// printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i,  good_matches[i].queryIdx,good_matches[i].trainIdx );
//}
    {
//good_matches[i].queryIdx保存着第一张图片匹配点的序号，keypoints_1[good_matches[i].queryIdx].pt.x 为该序号对应的点的x坐标。y坐标同理
//good_matches[i].trainIdx保存着第二张图片匹配点的序号，keypoints_2[good_matches[i].trainIdx].pt.x 为为该序号对应的点的x坐标。y坐标同理
        printf( "-- Good Match [%d] Keypoint 1(%f,%f): %d  -- Keypoint 2(%f,%f): %d  \n", i,
                keypoints_1[good_matches[i].queryIdx].pt.x,keypoints_1[good_matches[i].queryIdx].pt.y, good_matches[i].queryIdx,
                keypoints_2[good_matches[i].trainIdx].pt.x,keypoints_2[good_matches[i].trainIdx].pt.y,good_matches[i].trainIdx );
    }
    waitKey(0);
    return 0;
}


int bfmatch_point_matching()
{
    Mat srcImage1 = imread("m1.png",IMREAD_GRAYSCALE);
    Mat srcImage2 = imread("test1.png",IMREAD_GRAYSCALE);

    //判断文件是否读取成功
    if (srcImage1.empty() || srcImage2.empty())
    {
        std::cout << "图像加载失败!";
        return -1;
    }
    else
        std::cout << "图像加载成功..." << std::endl;

    //检测两幅图像中的特征点
    int minHessian = 2000;      //定义Hessian矩阵阈值

    SurfFeatureDetector detector(minHessian);       //定义Surf检测器
    vector<KeyPoint>keypoint1, keypoint2;           //定义两个KeyPoint类型矢量存储检测到的特征点
    detector.detect(srcImage1, keypoint1);
    detector.detect(srcImage2, keypoint2);

    //计算特征向量的描述子
    SurfDescriptorExtractor descriptorExtractor;
    Mat descriptors1, descriptors2;

    descriptorExtractor.compute(srcImage1, keypoint1, descriptors1);
    descriptorExtractor.compute(srcImage2, keypoint2, descriptors2);

    //使用BruteForceMatcher进行描述符匹配
    BFMatcher matcher(NORM_L2);
    vector<DMatch>matches;
    matcher.match(descriptors1, descriptors2, matches);

    //绘制匹配特征点
    Mat matchImage;
    drawMatches(srcImage1, keypoint1, srcImage2, keypoint2, matches, matchImage);

    //显示匹配的图像
    namedWindow("Match", WINDOW_AUTOSIZE);
    imshow("Match", matchImage);

    waitKey(0);

    return 0;
}

