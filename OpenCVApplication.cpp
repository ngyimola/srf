// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <fstream>
#include <stdio.h>
#include "stdafx.h"
#include "common.h"
#include "time.h"
#include <random>
#include <algorithm> 

using namespace cv;


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

// Lab 1
std::vector<Point2f> readPointsFromFile()
{
	int width = 500;
	int height = 500;
	int dimension = 2;
	Mat pointMatrix = Mat(height, width, CV_8UC3);

	FILE* f = fopen("Images/lab1/points0.txt", "r");
	int nrOfPoints;
	fscanf(f, "%d", &nrOfPoints);

	std::vector<Point2f> points;

	for (int i = 0; i < nrOfPoints; i++)
	{
		float x, y;
		fscanf(f, "%f%f", &x, &y);
		points.push_back(Point2f(x, y));

		if (x <= width && y <= height)
		{
			circle(pointMatrix, Point2d(x, y), 3, Scalar(255, 0, 0), -1);
		}
	}

	fclose(f);
	return points;
}

Mat pointsToImage(std::vector<Point2f> points)
{
	int n = points.size();
	int width = 500;
	int height = 500;
	Mat img = Mat(height, width, CV_8UC3);

	for (int i = 0; i < n; i++)
	{

		if (points.at(i).x <= height && points.at(i).y <= width)
		{
			circle(img, Point2d(points.at(i).x, points.at(i).y), 3, Scalar(255, 0, 0), -1);
		}
	}

	return img;
}

//method 1 b closed

void calculateTeta()
{
	std::vector<Point2f> points = readPointsFromFile();
	int n = points.size();
	Mat teta = Mat(2, 1, CV_32F);
	Mat A = Mat(n, 2, CV_32F);
	Mat b = Mat(n, 1, CV_32F);

	for (int i = 0; i < n; i++)
	{
		A.at<float>(i, 1) = points.at(i).x;
		A.at<float>(i, 0) = 1;
		b.at<float>(i, 0) = points.at(i).y;
	}

	teta = (A.t()*A).inv()*A.t()*b;
	printf("teta0 = %f, teta1 = %f\n", teta.at<float>(0, 0), teta.at<float>(1, 0));

	Mat img = pointsToImage(points);
	line(img, Point2d(0, teta.at<float>(0, 0)), Point2d(500, teta.at<float>(0, 0) + 500 * teta.at<float>(1, 0)), Scalar(0, 0, 0));

	imshow("points", img);
	waitKey(0);
}

// method 1 a closed
void calculateTeta1()
{
	std::vector<Point2f> points = readPointsFromFile();
	int n = points.size();
	float sumXY = 0;
	float sumX = 0;
	float sumY = 0;
	float sumX2 = 0;

	for (int i = 0; i < n; i++)
	{
		sumXY += points.at(i).x * points.at(i).y;
		sumX += points.at(i).x;
		sumY += points.at(i).y;
		sumX2 += pow(points.at(i).x, 2);
	}

	float teta1 = (n * sumXY - sumX * sumY) / (n * sumX2 - pow(sumX, 2));
	float teta0 = (sumY - teta1 * sumX) / n;

	Mat img = pointsToImage(points);
	line(img, Point2d(0, teta0), Point2d(500, teta0 + 500 * teta1), Scalar(0, 0, 0));

	imshow("points", img);
	waitKey(0);

}

// method 1 open

void optimiseTeta()
{
	std::vector<Point2f> points = readPointsFromFile();
	int n = points.size();
	Mat teta = Mat(2, 1, CV_32F);
	Mat tetaNew = Mat(2, 1, CV_32F);

	// init teta with random values
	teta.at<float>(0, 0) = rand() % n;
	teta.at<float>(1, 0) = rand() % n;

	float alpha = 0.00000001; // learning rate
	Mat img = pointsToImage(points);

	float error = 100000;

	while(true)
	{
		// calculate gradient
		Mat gradient = Mat(2, 1, CV_32F);
		gradient.at<float>(0, 0) = 0;
		gradient.at<float>(1, 0) = 0;

		for (int i = 0; i < n; i++)
		{
			gradient.at<float>(0, 0) += teta.at<float>(0,0) + teta.at<float>(1,0) * points.at(i).x - points.at(i).y;
			gradient.at<float>(1, 0) += (teta.at<float>(0, 0) + teta.at<float>(1, 0) * points.at(i).x - points.at(i).y) * points.at(i).x;
		}

		teta = teta - alpha * gradient;
		line(img, Point2d(0, teta.at<float>(0, 0)), Point2d(500, teta.at<float>(0, 0) + 500 * teta.at<float>(1, 0)), Scalar(0, 0, 0));

		imshow("points", img);
		waitKey(100);
	}
	
}

// method 2 closed
void calculateTeta2()
{
	std::vector<Point2f> points = readPointsFromFile();
	int n = points.size();
	float sumXY = 0;
	float sumX = 0;
	float sumY = 0;
	float sumX2Y2 = 0;

	for (int i = 0; i < n; i++)
	{
		sumXY += points.at(i).x * points.at(i).y;
		sumX += points.at(i).x;
		sumY += points.at(i).y;
		sumX2Y2 += pow(points.at(i).y, 2) - pow(points.at(i).x, 2);
	}

	float beta = -0.5 * atan2(2 * sumX2Y2 - (2 * sumX * sumY)/(float) n, sumX2Y2 + (pow(sumX, 2)/(float)n - pow(sumY, 2)/(float)n));
	float ro = (cos(beta) * sumX + sin(beta) * sumY)/(float)n;

	Mat img = pointsToImage(points);
	line(img, Point(0, ro / sin(beta)), Point(img.cols, (ro - img.cols * cos(beta)) / sin(beta)), Scalar(0, 0, 255));

	imshow("points", img);
	waitKey(0);
}

//Lab2
std::vector<Point> readPoints()
{
	Mat img = imread("Images/lab2/points1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	
	Mat inv;
	cv::bitwise_not(img, inv);
	std::vector<Point> points;
	cv::findNonZero(inv, points);
	return points;
}

void ransac()
{
	Mat img = imread("Images/lab2/points1.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	Mat inv;
	cv::bitwise_not(img, inv);
	std::vector<Point> pts;
	cv::findNonZero(inv, pts);
	
	float t = 10;
	float p = 0.99;
	float q = 0.3;
	float s = 2;

	int n = pts.size();
	float T = n*q;
	float N = log(1 - p) / log(1 - pow(q, s));
	
	int inliner = 0;
	Point3d optimalLine = Point3d(0, 0, 0);

	srand(time(0));
	for (int j = 0; (j < N) && (inliner < T); j++) 
	{
		
		int p1 = rand() % n;
		int p2 = rand() % n;

		Point3f linie;
		linie.x = pts.at(p1).y - pts.at(p2).y;
		linie.y = pts.at(p2).x - pts.at(p1).x;
		linie.z = pts.at(p1).x * pts.at(p2).y - pts.at(p2).x * pts.at(p1).y;

		float dist = 0;

		int i = 0;
		for each (Point p in pts)
		{
			dist = abs(linie.x * p.x + linie.y * p.y + linie.z) / sqrt(pow(linie.x, 2) + sqrt(pow(linie.y, 2)));
			if (dist <= t)
				i++;
		}
		if (i > inliner)
		{
			inliner = i;
			optimalLine = linie;
		}
	}

	cv::line(img, Point2d(0, -optimalLine.z / optimalLine.y), Point2d(500, (-optimalLine.z - optimalLine.x * 500)/ optimalLine.y), Scalar(0, 0, 0));

	cv::imshow("img", img);
	cv::waitKey();
}

// Lab 3
struct peak {
	int theta, ro, hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};

void houghTransform()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat colorImg = imread(fname, CV_LOAD_IMAGE_COLOR);
		int D = sqrt(pow(img.rows, 2) + pow(img.cols, 2));
		int roMax = D + 1;
		int thetaMax = 360;
		Mat Hough = Mat(thetaMax, roMax, CV_32SC1);

		for (int i = 0; i < thetaMax; i++)
		{
			for (int j = 0; j < roMax; j++)
			{
				Hough.at<int>(i, j) = 0;
			}
		}

		int EDGE = 255;
		int BCKGROUND = 0;

		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				if (img.at<uchar>(i, j) == EDGE)
				{
					for (int theta = 0; theta < thetaMax; theta++)
					{
						float ro = j * cos(2 * PI * (theta / 360.0)) + i * sin(2 * PI * (theta / 360.0));
						if (ro >= 0 && ro < roMax)
						{
							Hough.at<int>(theta, ro)++;
						}
					}
				}
			}
		}

		int maxHough = 0;
		for (int i = 0; i < Hough.rows; i++)
		{
			for (int j = 0; j < Hough.cols; j++)
			{
				if (Hough.at<int>(i, j) > maxHough)
				{
					maxHough = Hough.at<int>(i, j);
				}

				
			}
		}	
		
		Mat houghImg;
		Hough.convertTo(houghImg, CV_8UC1, 255.f / maxHough);
		imshow("Hough", houghImg);

		// extract local maximums
		 
		int boxSize = 15;
		std::vector<peak> peaks;

		for (int i = 0; i < Hough.rows - boxSize; i += boxSize)
		{
			for (int j = 0; j < Hough.cols - boxSize; j += boxSize)
			{
				peak localMax;
				localMax.theta = 0;
				localMax.ro = 0;
				localMax.hval = 0;

				for (int ii = 0; ii < boxSize; ii++)
				{
					for (int jj = 0; jj < boxSize; jj++)
					{
						if (Hough.at<int>(i + ii, j + jj) > localMax.hval) 
						{
							localMax.theta = i + ii;
							localMax.ro = j + jj;
							localMax.hval = Hough.at<int>(i + ii, j + jj);
						}

					}
				}
				if (localMax.hval != 0)
				{
					peaks.push_back(localMax);
				}
			}
		}

		std::sort(peaks.begin(), peaks.end());
		for(int i = 0; i < 15; i ++)
		{
			line(colorImg, Point2d(0, peaks.at(i).ro / sin(2 * PI * (peaks.at(i).theta / 360.0))), Point2d(img.cols, (peaks.at(i).ro - img.cols * cos(2 * PI * (peaks.at(i).theta / 360.0))) / sin(2 * PI * (peaks.at(i).theta / 360.0))), Scalar(255, 255, 0));
		}

		imshow("img", colorImg);

		waitKey(0);
	}

}

// lab4

Mat translateImg(Mat img, int offsetx, int offsety) {
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);
	warpAffine(img, img, trans_mat, img.size());
	return img;
}

void distanceTransform()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		// Mat dt = Mat(img.size(), CV_8UC1);
		// init DT matrix
		Mat dt = img.clone();

		// upper mask
		int ui[5] = {-1, -1, -1, 0, 0};
		int uj[5] = {-1, 0, 1, -1, 0};

		//lower mask
		int li[5] = {0, 0, 1, 1, 1};
		int lj[5] = {0, 1, -1, 0, 1};

		//weight
		int wHV = 2;
		int wD = 3;

		for (int i = 1; i < dt.rows - 1; i++)
		{
			for (int j = 1; j < dt.cols - 1; j++)
			{
				uchar minU = dt.at<uchar>(i, j);

				for (int k = 0; k < 5; k++)
				{
					int w;

					// upper
					if (ui[k] * uj[k] == 0) { w = wHV; }
					else { w = wD; }

					if (dt.at<uchar>(i + ui[k], j + uj[k]) + w < minU)
					{
						minU = dt.at<uchar>(i + ui[k], j + uj[k]) + w;
					}
				}
				dt.at<uchar>(i, j) = minU;
			}
		}

		for (int i = dt.rows - 2; i > 0; i--)
		{
			for (int j = dt.cols - 2; j > 0; j--)
			{
				uchar minL = dt.at<uchar>(i, j);

				for (int k = 0; k < 5; k++)
				{
					int w;

					// lower
					if (li[k] * lj[k] == 0) { w = wHV; }
					else { w = wD; }

					if (dt.at<uchar>(i + li[k], j + lj[k]) + w < minL)
					{
						minL = dt.at<uchar>(i + li[k], j + lj[k]) + w;
					}
				}
				dt.at<uchar>(i, j) = minL;
			}
		}

		imshow("dt", dt);
		waitKey(0);

		//Centru de masa a dt-ului
		int nDT = 0;
		int centerXdt = 0;
		int centerYdt = 0;

		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				if (img.at<uchar>(i, j) == 0)
				{
					nDT++;
					centerXdt += i;
					centerYdt += j;
				}
			}
		}

		Mat src = imread("E:/mappa/ut/an4/SRF/lab/OpenCVApplication-VS2015_OCV31_basic/Images/lab4/PatternMatching/unknown_object1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
		int n = 0;
		int centerX = 0;
		int centerY = 0;

		float score = 0;

		// Centru de masa a sablonului
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					n++;
					centerX += i;
					centerY += j;
					
				}
			}
		}

		
		

		int deltaX = centerXdt / nDT - centerX / n;
		int deltaY = centerYdt / nDT - centerY / n;

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if((src.at<uchar>(i,j) == 0) && (i + deltaX >= 0) && (i + deltaX < dt.rows) && (j + deltaY >= 0) && (j + deltaY < dt.cols))
					score += dt.at<uchar>(i + deltaX, j + deltaY);
			}
		}
		score /= (float)n;
		printf("Score is: %f", score);
		waitKey(0);

	}
}

//lab5
void statisticalData()
{
	char folder[256] = "E:/mappa/ut/an4/SRF/lab/OpenCVApplication-VS2015_OCV31_basic/Images/lab5";
	char fname[256];
	Mat faces = Mat(400, 361, CV_8UC1);

	//read faces and flatten
	for (int i = 1; i <= 400; i++) 
	{
		sprintf(fname, "%s/face%05d.bmp", folder, i);
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		for (int ii = 0; ii < 19; ii++)
		{
			for (int jj = 0; jj < 19; jj++)
			{
				faces.at<uchar>(i-1, ii *19 + jj) = img.at<uchar>(ii, jj);
			}
		}
	}

	// calculate mean features
	Mat I = Mat(1, 361, CV_32FC1);

	for (int i = 0; i < 361; i++) 
	{
		I.at<float>(0, i) = 0;
	}

	for (int j = 0; j < 361; j++)
	{
		for (int i = 0; i < 400; i++)
		{
			I.at<float>(0, j) += faces.at<uchar>(i, j);
		}
		I.at<float>(0, j) /= 400;
	}
	
	FILE *f1 = fopen("meanVal.csv", "w");
	if (f1 == NULL) printf("There was a problem at opening\n");
	for (int j = 0; j < 361; j++)
	{
		fprintf(f1, "%f,", I.at<float>(0, j));
	}

	fclose(f1);

	// covariance
	Mat cov = Mat(361, 361, CV_32FC1);
	for (int i = 0; i < 361; i++)
	{
		for (int j = 0; j < 361; j++)
		{
			cov.at<float>(i, j) = 0;
		}

	}

	for (int i = 0; i < 361; i++)
	{
		for (int j = 0; j < 361; j++)
		{
			for (int k = 0; k < 400; k++)
			{
				cov.at<float>(i, j) += (faces.at<uchar>(k, i) - I.at<float>(0, i)) * (faces.at<uchar>(k, j) - I.at<float>(0, j));
			}
			cov.at<float>(i, j) /= 400;
		}
	}

	FILE *f2 = fopen("covariance.csv", "w");
	if (f2 == NULL) printf("There was a problem at opening\n");
	for (int i = 0; i < 361; i++)
	{
		for (int j = 0; j < 361; j++)
		{
			fprintf(f2, "%f,", cov.at<float>(i, j));
		}
		fprintf(f2, "\n");
	}
	fclose(f2);

	//standard deviation
	Mat dev = Mat(1, 361, CV_32FC1);

	for (int i = 0; i < 361; i++)
	{
		dev.at<float>(0, i) = 0;
	}

	for (int i = 0; i < 361; i++)
	{
		for (int j = 0; j < 400; j++)
		{
			dev.at<float>(0, i) += pow((faces.at<uchar>(j, i) - I.at<float>(0, i)), 2);
		}
		dev.at<float>(0, i) = sqrt(dev.at<float>(0, i)/400);
	}

	FILE *f4 = fopen("dev.csv", "w");
	if (f4 == NULL) printf("There was a problem at opening\n");
	for (int i = 0; i < 361; i++)
	{
		fprintf(f4, "%f,", dev.at<float>(0, i));
	}
	fclose(f4);

	// correlation coeficients
	Mat ro = Mat(361, 361, CV_32FC1);
	for (int i = 0; i < 361; i++)
	{
		for (int j = 0; j < 361; j++)
		{
			ro.at<float>(i, j) = 0;
		}
	}

	for (int i = 0; i < 361; i++)
	{
		for (int j = 0; j < 361; j++)
		{
			ro.at<float>(i, j) = cov.at<float>(i, j) / (dev.at<float>(0, i) * dev.at<float>(0, j));			
		}
	}

	FILE *f3 = fopen("corelCoef.csv", "w");
	if (f3 == NULL) printf("There was a problem at opening\n");
	for (int i = 0; i < 361; i++)
	{
		for (int j = 0; j < 361; j++)
		{
			fprintf(f3, "%f,", ro.at<float>(i, j));
		}
		fprintf(f3, "\n");
	}
	fclose(f3);

	// correlation img
	printf("Enter values for corelation img");
	int xi;
	scanf("%d", &xi);
	int xj;
	scanf("%d", &xj);
	int yi; 
	scanf("%d", &yi);
	int yj;
	scanf("%d", &yj);
	Mat cimg = Mat(256, 256, CV_8UC1);
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			cimg.at<uchar>(i, j) = 255;
		}
	}
	for (int k = 0; k < 400; k++)
	{
		cimg.at<uchar>(faces.at<uchar>(k, xi), faces.at<uchar>(k, xj)) = 0;
		cimg.at<uchar>(faces.at<uchar>(k, yi), faces.at<uchar>(k, yj)) = 0;
	}

	imshow("faces", cimg);

	waitKey(0);
}


// Lab 6

void principalComponent() {
	FILE* f = fopen("Images/lab7/pca3d.txt", "r");
	int n, d;
	fscanf(f, "%d", &n);
	fscanf(f, "%d", &d);
	Mat X = Mat(n, d, CV_64FC1);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < d; j++)
		{
			fscanf(f, "%lf", &X.at<double>(i, j));
		}
	}
	fclose(f);
	
	Mat mean = Mat(1, d, CV_64FC1);
	for (int i = 0; i < d; i++)
	{
		mean.at<double>(0, i) = 0;
	}


	// Mean
	for (int i = 0; i < d; i++)
	{
		for (int j = 0; j < i; j++)
		{
			mean.at<double>(0, i) += X.at<double>(j, i);
		}
		mean.at<double>(0, i) /= n;
	}

	// X normalized

	Mat Xnorm = X.clone();

	for (int i = 0; i < n; i++)
	{
		Xnorm.row(i) = Xnorm.row(i) - mean;
	}

	// covariance
	Mat C = Mat(d, d, CV_64FC1);
	C = Xnorm.t() * Xnorm / (n-1);


	//Eigen
	Mat Lambda, Q;
	cv::eigen(C, Lambda, Q);
	Q = Q.t();

	for (int i = 0; i < Lambda.rows; i++)
	{
		for (int j = 0; j < Lambda.cols; j++)
		{
			printf("%lf ", Lambda.at<double>(i, j));
		}
		printf("\n");
	}

	Mat Xcoef = Mat(n, d, CV_64FC1);
	Mat Xaprox = Mat(n, d, CV_64FC1);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < d; j++)
		{
			Xaprox.at<double>(i, j) = 0;
		}
	}

	for (int k = 0; k < d; k++)
	{
		Xcoef.col(k) = Xnorm * Q.col(k);
		Xaprox += Xnorm*Q.col(k) * Q.col(k).t();
	}

	Mat dif = Mat(n, d, CV_64FC1);
	dif = abs(Xaprox - X);

	Mat img = Mat(500, 500, CV_8UC1);
	for (int i = 0; i < 500; i++)
	{
		for (int j = 0; j < 500; j++)
		{
			img.at<uchar>(i, j) = 255;
		}
	}

	double minX = Xcoef.at<double>(0, 0);
	double maxX = Xcoef.at<double>(0, 0);
	double minY = Xcoef.at<double>(0, 1);
	double maxY = Xcoef.at<double>(0, 1);
	double minZ = Xcoef.at<double>(0, 2);
	double maxZ = Xcoef.at<double>(0, 2);

	for (int i = 0; i < n; i++)
	{
		if (minX > Xcoef.at<double>(i, 0))
		{
			minX = Xcoef.at<double>(i, 0);
		}
		else if (maxX < Xcoef.at<double>(i, 0))
		{
			maxX = Xcoef.at<double>(i, 0);
		}

		if (minY > Xcoef.at<double>(i, 1))
		{
			minY = Xcoef.at<double>(i, 1);
		}
		else if (maxY < Xcoef.at<double>(i, 1))
		{
			maxY = Xcoef.at<double>(i, 1);
		}

		if (minZ > Xcoef.at<double>(i, 2))
		{
			minZ = Xcoef.at<double>(i, 2);
		}
		else if (maxZ < Xcoef.at<double>(i, 2))
		{
			maxZ = Xcoef.at<double>(i, 2);
		}
	}

	for (int i = 0; i < n; i++)
	{
		Xcoef.at<double>(i, 0) = ((Xcoef.at<double>(i, 0) - minX) / (maxX-minX) ) * 499;
		Xcoef.at<double>(i, 1) = ((Xcoef.at<double>(i, 1) - minY) / (maxY - minY)) * 499;

		int x = Xcoef.at<double>(i, 0); 
		int y = Xcoef.at<double>(i, 1);// < 0 ? 0 : (Xaprox.at<double>(i, 1) > 499 ? 499 : Xaprox.at<double>(i, 1));
		img.at<uchar>(x, y) = (uchar)((Xcoef.at<double>(i, 2) - minZ) / (maxZ - minZ)) * 255;
	}


		
	imshow("img", img);
	waitKey(0);
}

// Lab 7

struct ImgPoint
{
	std::vector<int> features;
	int cluster;
};

std::vector<ImgPoint> kmeans(int k, std::vector<ImgPoint> points, int d)
{
	std::vector<ImgPoint> means;

	int n = points.size();

	std::default_random_engine generator;
	generator.seed(time(NULL));
	std::uniform_int_distribution<int> distribution(0, n - 1);

	// Initialization
	// generate random means and assign a separate clusters
	for (size_t i = 0; i < k; i++)
	{
		int randint = distribution(generator);
		ImgPoint point = points.at(randint);
		point.cluster = i;
		means.push_back(point);
	}

	bool isChanged = true;
	int it = 0;
	while (isChanged&&it<1000)
	{
		it++;
		// Assignment
		for (size_t i = 0; i < n; i++)
		{
			float minDist = INT_MAX;
			for (size_t j = 0; j < k; j++)
			{
				float dist = 0;
				// euclidean dist in d-dimensional space
				for (size_t di = 0; di < d; di++)
				{
					dist += pow(means.at(j).features.at(di) - points.at(i).features.at(di), 2);
				}
				dist = sqrt(dist);

				if (dist < minDist) 
				{
					minDist = dist;
					points.at(i).cluster = means.at(j).cluster;
				}
			}
		}
		// update
		std::vector<int> nForReg;
		std::vector< std::vector<int> > sumForReg;

		sumForReg.resize(k);
		for (int i = 0; i < k; ++i)
		{
			//Grow Columns by n
			sumForReg[i].resize(d);
		}

		for (size_t i = 0; i < k; i++)
		{
			nForReg.push_back(0);
		}

		for (size_t i = 0; i < n; i++)
		{
			nForReg.at(points.at(i).cluster)++;
			for (size_t j = 0; j < d; j++)
			{
				sumForReg[points.at(i).cluster][j] += points.at(i).features.at(j);
			}
		}

		for (size_t i = 0; i < k; i++)
		{
			for (size_t j = 0; j < d; j++)
			{
				int newFeature = nForReg.at(i)!=0 ? sumForReg[i][j] / nForReg.at(i) : 0;
				if (means.at(i).features.at(j) != newFeature) {
					means.at(i).features.at(j) = newFeature;
					isChanged = true;
				}
				else
				{
					isChanged = false;
				}
			}
		}
	}
	return points;
}

std::vector<ImgPoint> kmeansReturnMeans(int k, std::vector<ImgPoint> points, int d)
{
	std::vector<ImgPoint> means;

	int n = points.size();

	std::default_random_engine generator;
	generator.seed(time(NULL));
	std::uniform_int_distribution<int> distribution(0, n - 1);

	// Initialization
	// generate random means and assign a separate clusters
	for (size_t i = 0; i < k; i++)
	{
		int randint = distribution(generator);
		ImgPoint point = points.at(randint);
		point.cluster = i;
		means.push_back(point);
	}

	bool isChanged = true;
	int it = 0;
	while (isChanged&&it<1000)
	{
		it++;
		// Assignment
		for (size_t i = 0; i < n; i++)
		{
			float minDist = INT_MAX;
			for (size_t j = 0; j < k; j++)
			{
				float dist = 0;
				// euclidean dist in d-dimensional space
				for (size_t di = 0; di < d; di++)
				{
					dist += pow(means.at(j).features.at(di) - points.at(i).features.at(di), 2);
				}
				dist = sqrt(dist);

				if (dist < minDist)
				{
					minDist = dist;
					points.at(i).cluster = means.at(j).cluster;
				}
			}
		}
		// update
		std::vector<int> nForReg;
		std::vector< std::vector<int> > sumForReg;

		sumForReg.resize(k);
		for (int i = 0; i < k; ++i)
		{
			//Grow Columns by n
			sumForReg[i].resize(d);
		}

		for (size_t i = 0; i < k; i++)
		{
			nForReg.push_back(0);
		}

		for (size_t i = 0; i < n; i++)
		{
			nForReg.at(points.at(i).cluster)++;
			for (size_t j = 0; j < d; j++)
			{
				sumForReg[points.at(i).cluster][j] += points.at(i).features.at(j);
			}
		}

		for (size_t i = 0; i < k; i++)
		{
			for (size_t j = 0; j < d; j++)
			{
				int newFeature = nForReg.at(i) != 0 ? sumForReg[i][j] / nForReg.at(i) : 0;
				if (means.at(i).features.at(j) != newFeature) {
					means.at(i).features.at(j) = newFeature;
					isChanged = true;
				}
				else
				{
					isChanged = false;
				}
			}
		}
	}
	return means;
}


std::vector<ImgPoint> read2dPoints(Mat img)
{
	std::vector<ImgPoint> points;
	for (size_t i = 0; i < img.rows; i++)
	{
		for (size_t j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) == 0)
			{
				std::vector<int> coord;
				coord.push_back(i);
				coord.push_back(j);
				ImgPoint imgPoint = { coord, 0 };
				points.push_back(imgPoint);
			}
		}
	}
	return points;
}

std::vector<Vec3b> generateRandomColors(int k)
{
	std::default_random_engine gen;
	std::uniform_int_distribution<int> dist_img(0, 255);
	std::vector<Vec3b> colors;
	colors.resize(k);
	for (uchar i = 0; i < k; i++)
	{
		colors[i] = { (uchar)dist_img(gen), (uchar)dist_img(gen), (uchar)dist_img(gen) };
	}
		
	return colors;
}

void pointsSimple(Mat img, int k)
{
	std::vector<ImgPoint> points = read2dPoints(img);
	std::vector<ImgPoint> processed = kmeans(k, points, 2);
	std::vector<Vec3b> colors = generateRandomColors(k);

	Mat dst = Mat(img.size(), CV_8UC3);
	for (size_t i = 0; i < dst.rows; i++)
	{
		for (size_t j = 0; j < dst.cols; j++)
		{
			dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
		}
	}

	for (size_t i = 0; i < processed.size(); i++)
	{
		dst.at<Vec3b>(processed.at(i).features.at(0), processed.at(i).features.at(1)) = colors[processed.at(i).cluster];
	}

	cv::imshow("points", dst);
	cv::waitKey(0);

}

void voronoi(Mat img, int k)
{
	std::vector<ImgPoint> points = read2dPoints(img);
	std::vector<ImgPoint> means = kmeansReturnMeans(k, points, 2);
	std::vector<Vec3b> colors = generateRandomColors(k);

	Mat dst = Mat(img.rows, img.cols, CV_8UC3);
	for (size_t i = 0; i < dst.rows; i++)
	{
		for (size_t j = 0; j < dst.cols; j++)
		{
			float minDist = INT_MAX;
			for (int ki = 0; ki < k; ki++)
			{
				float dist = sqrt(pow(std::abs((double)i - means.at(ki).features.at(0)), 2) + pow(std::abs((double)j - means.at(ki).features.at(1)), 2));
				if (dist < minDist)
				{
					minDist = dist;
					dst.at<Vec3b>(i, j) = colors[means.at(ki).cluster];
				}
			}	
		}
	}
	imshow("img", img);
	cv::imshow("voronoi", dst);
	cv::waitKey(0);
}

void pointsColor(Mat img, int k)
{
	std::vector<ImgPoint> points;
	for (size_t i = 0; i < img.rows; i++)
	{
		for (size_t j = 0; j < img.cols; j++)
		{
			std::vector<int> c;
			c.push_back(img.at<Vec3b>(i, j).val[0]);
			c.push_back(img.at<Vec3b>(i, j).val[1]);
			c.push_back(img.at<Vec3b>(i, j).val[2]);

			ImgPoint imgColor = { c, 0 };
			points.push_back(imgColor);
		}
	}

	std::vector<ImgPoint> processed = kmeans(k, points, 3);
	std::vector<Vec3b> colors = generateRandomColors(k);

	Mat dst_color = Mat(img.size(), CV_8UC3);
	for (size_t i = 0; i < dst_color.rows; i++)
	{
		for (size_t j = 0; j < dst_color.cols; j++)
		{
			dst_color.at<Vec3b>(i, j) = colors[processed.at(i * dst_color.cols + j).cluster];
		}
	}
	imshow("img", img);
	cv::imshow("color", dst_color);
	cv::waitKey(0);
}

void pointsGray(Mat img, int k)
{
	std::vector<ImgPoint> points;
	for (size_t i = 0; i < img.rows; i++)
	{
		for (size_t j = 0; j < img.cols; j++)
		{
			std::vector<int> intensity;
			intensity.push_back(img.at<uchar>(i, j));
			ImgPoint imgIntens = { intensity, 0 };
			points.push_back(imgIntens);
		}
	}
	std::vector<ImgPoint> processed = kmeans(k, points, 1);
	std::vector<Vec3b> colors = generateRandomColors(k);
	Mat dst_gray = Mat(img.size(), CV_8UC3);
	for (size_t i = 0; i < dst_gray.rows; i++)
	{
		for (size_t j = 0; j < dst_gray.cols; j++)
		{
			dst_gray.at<Vec3b>(i, j) = colors[processed.at(i * dst_gray.cols + j).cluster];
		}
	}
	cv::imshow("gray", dst_gray);
	cv::waitKey();
}

void km()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat colorImg = imread(fname, CV_LOAD_IMAGE_COLOR);
		Mat img;
		cv::cvtColor(colorImg, img, cv::COLOR_RGB2GRAY);

		// Define the number of clusters
		printf("K = ");
		int k;
		scanf("%d", &k);

		//Choose option
		int option;
		do
		{
			printf("Choose k-means:\n");
			printf(" 1 - Points simple\n");
			printf(" 2 - Points Voronoi\n");
			printf(" 3 - Points color\n");
			printf(" 4 - Points gary\n");
			printf(" 0 - Exit\n\n");
			printf("Option: ");
			scanf("%d", &option);
			switch (option)
			{
				case 1:
					pointsSimple(img, k);
					break;
				case 2:
					voronoi(img, k);
					break;
				case 3:
					pointsColor(colorImg, k);
					break;
				case 4:
					pointsGray(img, k);
					break;			
			}
		} while (option != 0);
	}
}

// Lab 8

#define NR_OF_BINS 8
#define NR_OF_IMAGES 672
#define NR_OF_CLASSES 6
#define NR_OF_NEIGHBOURS 7
#define NR_OF_TEST_IMAGES 85

const int nrDim = NR_OF_BINS * 3;

void calcHist(Mat img, int*hist) {
	Vec3i color_hist[256];

	// init vector
	for (size_t i = 0; i < 256; i++)
	{
		color_hist[i] = Vec3i(0, 0, 0);
	}

	// calculate histogram
	for (size_t i = 0; i < img.rows; i++)
	{
		for (size_t j = 0; j < img.cols; j++)
		{
			color_hist[img.at<Vec3b>(i, j)[0]][0] ++;
			color_hist[img.at<Vec3b>(i, j)[1]][1] ++;
			color_hist[img.at<Vec3b>(i, j)[2]][2] ++;
		}
	}


	Vec3i compr_color_hist[NR_OF_BINS];
	for (size_t i = 0; i < NR_OF_BINS; i++) {
		compr_color_hist[i] = Vec3i(0, 0, 0);
	}

	float chunk_size = 256.f / NR_OF_BINS;

	for (size_t i = 0; i < NR_OF_BINS; i++) {
		for (size_t j = 0; j < 256; j++) {
			if (j >= (i*chunk_size) && j < ((i + 1)*chunk_size)) {
				compr_color_hist[i] += color_hist[j];
			}
		}
	}

	//might store it as |  B  |  G  |  R  |  
	for (size_t i = 0; i < 3; i++) {
		for (size_t j = 0; j < NR_OF_BINS; j++) {
			hist[i*NR_OF_BINS + j] = compr_color_hist[j][i];
		}
	}
}

struct DistStruct {
	int dist;
	int label;
};

bool compFunc(DistStruct a, DistStruct b)
{
	return (a.dist < b.dist);
}

void knearest()
{
	char classes[NR_OF_CLASSES][10] = { "beach", "city", "desert", "forest", "landscape", "snow" };

	int fileNr = 0, c = 0;
	int rowX = 0;
	char fname[1000];

	Mat X(NR_OF_IMAGES, nrDim, CV_32FC1);
	int Y[NR_OF_IMAGES];
	int hist[nrDim];

	for (c = 0; c < NR_OF_CLASSES; c++) {
		fileNr = 0;
		rowX = 0;
		while (1) {
			sprintf(fname, "Images/lab8/train/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
			if (img.cols == 0) break;

			//calculate the histogram in hist
			calcHist(img, hist);

			for (int d = 0; d < nrDim; d++)
				X.at<float>(rowX, d) = hist[d];
			Y[rowX] = c;
			rowX++;
		}
	}

	Mat C(NR_OF_CLASSES, NR_OF_CLASSES, CV_32FC1);

	for (size_t i = 0; i < C.rows; i++)
	{
		for (size_t j = 0; j < C.cols; j++)
		{
			C.at<float>(i, j) = 0;
		}

	}

	int src_hist[nrDim];

	for (c = 0; c < NR_OF_CLASSES; c++) {
		fileNr = 0;
		rowX = 0;
		while (1) {
			sprintf(fname, "Images/lab8/test/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
			if (img.cols == 0) break;

 			calcHist(img, src_hist);

			std::vector<DistStruct> distances;
			distances.resize(NR_OF_IMAGES);
			for (size_t i = 0; i < NR_OF_IMAGES; i++)
			{
				distances.at(i).dist = 0;
				distances.at(i).label = Y[i];
			}

			for (size_t i = 0; i < NR_OF_IMAGES; i++)
			{
				for (size_t j = 0; j < nrDim; j++)
				{
					distances[i].dist += abs(X.at<float>(i, j) - src_hist[j]);
				}
			}

			// sort the distances
			std::sort(distances.begin(), distances.end(), compFunc);

			//vote hist
			std::vector<int> vote;
			vote.resize(NR_OF_CLASSES, 0);

			for (size_t i = 0; i < NR_OF_NEIGHBOURS; i++)
			{
				vote[distances[i].label] ++;
			}

			int maxVote = 0;
			int maxClass = 0;
			for (size_t i = 0; i < NR_OF_CLASSES; i++)
			{
				if (maxVote < vote[i])
				{
					maxVote = vote[i];
					maxClass = i;
				}
			}

			C.at<float>(c, maxClass)++;

			rowX++;
		}
	}

	// accuratete
	float acc = 0;
	float correct = 0;
	float all = 0;

	for (size_t i = 0; i < NR_OF_CLASSES; i++)
	{
		correct += C.at<float>(i, i);
		for (size_t j = 0; j < NR_OF_CLASSES; j++)
		{
			all += C.at<float>(i, j);
		}
	}

	acc = correct / all;
	printf("Acc: %f", acc);

	/*
	while (openFileDlg(fname))
	{

		// Read test img
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int src_hist[nrDim];
		calcHist(src, src_hist);

		std::vector<DistStruct> distances;
		distances.resize(NR_OF_IMAGES);
		for (size_t i = 0; i < NR_OF_IMAGES; i++)
		{
			distances.at(i).dist = 0;
			distances.at(i).label = Y[i];
		}

		for (size_t i = 0; i < NR_OF_IMAGES; i++)
		{
			for (size_t j = 0; j < nrDim; j++)
			{
				distances[i].dist += abs(X.at<float>(i, j) - src_hist[j]);
			}
		}

		// sort the distances
		std::sort(distances.begin(), distances.end(), compFunc);

		//vote hist
		std::vector<int> vote;
		vote.resize(NR_OF_CLASSES, 0);

		//init
		for (size_t i = 0; i < NR_OF_CLASSES; i++)
		{
			vote[i] = 0;
		}

		for (size_t i = 0; i < NR_OF_NEIGHBOURS; i++)
		{
			vote[distances[i].label] ++;
		}

		int maxVote = 0;
		int maxClass = 0;
		for (size_t i = 0; i < NR_OF_CLASSES; i++)
		{
			if (maxVote < vote[i])
			{
				maxVote = vote[i];
				maxClass = i;
			}
		}

		printf("It is callsified as: %d", maxClass);
	}
	*/


}

#define NR_CLASSES 10
#define IMG_SIZE 28

int classifyBayes(Mat img, std::vector<double> priors, Mat likelihood) {
	double class_logs[NR_CLASSES];

	for (int c = 0; c < NR_CLASSES; c++) {
		class_logs[c] = log(priors[c]);
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img.at<uchar>(i, j) == 0) {
					class_logs[c] += log(1 - likelihood.at<double>(c, i*IMG_SIZE + j));
				}
				else {
					class_logs[c] += log(likelihood.at<double>(c, i*IMG_SIZE + j));
				}
			}
		}
	}

	double max = class_logs[0];
	double maxc = 0;

	for (int c = 1; c < NR_CLASSES; c++) {
		if (class_logs[c] > max) {
			max = class_logs[c];
			maxc = c;
		}
	}

	return maxc;
}

// Lab 9
void bayes()
{
	// nr of images from the training set
	int n = 60000;
	int d = IMG_SIZE*IMG_SIZE;
	//feature vector
	Mat X = Mat(n, d, CV_8UC1);
	std::vector<int> y;
	y.resize(n);

	int ni = 0;

	for (int c = 0; c < NR_CLASSES; c++) {
		char fname[256];
		int index = 0;
		while (1) {
			sprintf(fname, "Images/lab9/train/%d/%06d.png", c, index);
			Mat img = imread(fname, 0);
			if (img.cols == 0) break;
			//process img
			for (int i = 0; i < img.rows; i++) {
				for (int j = 0; j < img.cols; j++) {
					if (img.at<uchar>(i, j) < 128)
						X.at<uchar>(ni, i * IMG_SIZE + j) = 0;
					else {
						X.at<uchar>(ni, i * IMG_SIZE + j) = 255;
					}
				}
			}
			y[ni] = c;
			index++;
			ni++;
		}	}

	// calculate priors
	std::vector<double> priors;
	priors.resize(NR_CLASSES);

	for (int i = 0; i < NR_CLASSES; i++) {
		priors[i] = 0;
	}

	for (int i = 0; i < n; i++) {
		priors[y[i]] ++;
	}

	for (size_t i = 0; i < NR_CLASSES; i++)
	{
		priors[i] /= (double)n;
	}

	// likelihood
	Mat likelihood = Mat(NR_CLASSES, d, CV_64FC1);

	//init with 0
	for (size_t i = 0; i < likelihood.rows; i++)
	{
		for (size_t j = 0; j < likelihood.cols; j++)
		{
			likelihood = 0;
		}
	}

	for (size_t i = 0; i < n; i++)
	{
		// total nr of pixels for 
		for (size_t k = 0; k < d; k++)
		{
			if (X.at<uchar>(i, k) == 255) {
				likelihood.at<double>(y[i], k)++;
			}
		}
	}

	for (size_t i = 0; i < NR_CLASSES; i++)
	{
		for (size_t j = 0; j < d; j++)
		{
			likelihood.at<double>(i, j) = (likelihood.at<double>(i, j) + 1) / ((priors[i] * n) + NR_CLASSES);
		}
	}

	// read test images
	int n_test = 10000;
	Mat confusion = Mat(NR_CLASSES, NR_CLASSES, CV_32FC1);

	for (size_t i = 0; i < confusion.rows; i++)
	{
		for (size_t j = 0; j < confusion.cols; j++)
		{
			confusion.at<float>(i, j) = 0;
		}
	}

	for (int c = 0; c < NR_CLASSES; c++) {
		char fname[256];
		int index = 0;
		while (1) {
			sprintf(fname, "Images/lab9/test/%d/%06d.png", c, index);
			Mat img = imread(fname, 0);
			if (img.cols == 0) break;
			//process img
			for (int i = 0; i < img.rows; i++) {
				for (int j = 0; j < img.cols; j++) {
					if (img.at<uchar>(i, j) < 128)
						img.at<uchar>(i, j) = 0;
					else {
						img.at<uchar>(i, j) = 255;
					}
				}
			}
			confusion.at<float>(c, classifyBayes(img, priors, likelihood)) ++;
			index++;
		}	}

	// accuratete
	float acc = 0;
	float correct = 0;
	float all = 0;

	for (size_t i = 0; i < NR_CLASSES; i++)
	{
		correct += confusion.at<float>(i, i);
		for (size_t j = 0; j < NR_CLASSES; j++)
		{
			all += confusion.at<float>(i, j);
		}
	}


	acc = correct / all;
	printf("Accuracy is: %f\n", acc);

}




int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Lab1\n");
		printf(" 11 - Lab1 open\n");
		printf(" 12 - Lab2\n");
		printf(" 13 - Hough\n");
		printf(" 14 - Distance transform\n");
		printf(" 15 - Statistical data\n");
		printf(" 16 - Principal componant analasys\n");
		printf(" 17 - K-means\n");
		printf(" 18 - K-nearest\n");
		printf(" 19 - Bayes\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				calculateTeta2();
				break;
			case 11:
				optimiseTeta();
				break;
			case 12:
				ransac();
				break;
			case 13:
				houghTransform();
				break;
			case 14: 
				distanceTransform();
				break;
			case 15:
				statisticalData();
				break;
			case 16:
				principalComponent();
				break;
			case 17:
				km();
				break;
			case 18:
				knearest();
				break;
			case 19:
				bayes();
				break;
		}
	}
	while (op!=0);
	return 0;
}