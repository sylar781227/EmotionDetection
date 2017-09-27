
	#include <stdio.h>
	#include <stdlib.h>
	#include "opencv/highgui.h"
	#include "stasm_lib.h"
	#include <iostream>
	#include <fstream>
	#include <opencv2/ml/ml.hpp>
	#include <opencv2/core/core.hpp>
	#include <conio.h>
	#define numberOfPoints 11

	using namespace std;
	using namespace cv;

	//TODO: ADD ALL FUNCTION DEFINITIONS HERE
	void returnPoints(const char* , int );
	void readImages(float landmarks[][numberOfPoints] , float labels[]);
	void returnPointsForTest(const char* ipath);
	float testFrame(const char* ipath);

	//TODO: CHANGE THE CONSTANT TO THE NUMBER OF TRAINING IMAGES IN DATASET
	const int n=112;
	float landmarkTrainArray[n][numberOfPoints];
	float labels[n];
	float landmarkTestArray[numberOfPoints];
	CvSVM svm;

	void main()
	{
		float classPrediction=0;
		String emotionPrediction="";

		cout  << " Training Images Now ................" << endl;
		readImages(landmarkTrainArray,labels);

		Mat trainData = Mat(n,numberOfPoints,DataType<float>::type, landmarkTrainArray);
		Mat trainLabels = Mat(n,1,DataType<float>::type,labels);

		CvSVMParams params;
		params.svm_type    = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

		//CvParamGrid CvParamGrid_C(pow(2.0,-5), pow(2.0,15), pow(2.0,2));
		//CvParamGrid CvParamGrid_gamma(pow(2.0,-15), pow(2.0,3), pow(2.0,2));
		//if (!CvParamGrid_C.check() || !CvParamGrid_gamma.check())
		//	cout<<"The grid is NOT VALID."<<endl;
		//CvSVMParams paramz;
		//paramz.kernel_type = CvSVM::RBF;
		//paramz.svm_type = CvSVM::C_SVC;
		//paramz.term_crit = cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001);
		//svm.train_auto(trainData,trainLabels,Mat(),Mat(),paramz,10,CvParamGrid_C,CvParamGrid_gamma,CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU),
		//	CvSVM::get_default_grid(CvSVM::COEF),CvSVM::get_default_grid(CvSVM::DEGREE),true);

		//dbvtraining
		svm.train(trainData, trainLabels, Mat(), Mat(), params);
		
		string filename = "../data/emo1.avi";
		//VideoCapture capture(filename);
		VideoCapture capture;
		//open capture object at location zero (default location for webcam)
		capture.open(0);
		
		
		Mat frame;

		if( !capture.isOpened() )
			throw "Error when reading steam_avi";

		namedWindow( "w", 1);
		for( int counter=0; ; counter++)
		{
			capture >> frame;
			Mat clonedFrame = frame.clone();
			if(frame.empty())
				break;
			if(counter%10==0) {
				
				waitKey(20); // waits to display frame
				imwrite("../data/test/test.jpg", clonedFrame);
				classPrediction=testFrame("../data/test/test.jpg");
				
				if(classPrediction==0)
					{
						emotionPrediction="Happy";
					}
				if(classPrediction==1)
					{
						emotionPrediction="Angry";

					}
				if(classPrediction==2)
					{
						emotionPrediction="Sad";
					}
				putText(frame,emotionPrediction, Point(100,100), FONT_HERSHEY_PLAIN, 0.8, CV_RGB(0, 255, 0), 0.5, false);
				imshow("w", frame);
				cout<<"TEST IMAGE IS predicted AS "<<classPrediction<<endl;

			}
		}
		waitKey(0); // key press to close window
		// releases and window destroy are automatic in C++ interface

		getch();
	}


	float testFrame(const char* ipath) 
	{
		returnPointsForTest(ipath);
		string emotionPrediction="Normal";
		Mat testData = Mat(1, numberOfPoints, DataType<float>::type, landmarkTestArray);
		float classValue = svm.predict(testData);
		cout<<"TEST IMAGE IS CLASSIFIED AS "<<classValue<<endl;

		return classValue;
	}

	void readImages(float landmarks[][numberOfPoints] , float labels[])
	{
		char* path = "../data/imagelist3.txt";
		string line;
		string prefix = "../data/";
		ifstream myfile (path);
		int i=0;

		if (myfile.is_open())
		{
			while ( getline (myfile,line) )
			{
				string imagepath = line.substr(0,line.find(";"));
				unsigned l = line.length() - 1;
				char label = line.at(l);
				imagepath = prefix + imagepath;
				const char* ipath;
				ipath = imagepath.c_str();
				cout << ipath << endl;
				returnPoints(ipath,i);
				float la = (float) (label - '0');
				labels[i] = la;
				i++;
			}
			cout << " Totally Trained " << i << " Imagess " << endl;
			myfile.close();
		}

	}

	void returnPoints(const char* ipath, int i)
	{
		cv::Mat_<unsigned char> img(cv::imread(ipath, CV_LOAD_IMAGE_GRAYSCALE));
		if (!img.data)
		{
			printf("Cannot load %s\n", ipath);
			exit(1);
		}

		int foundface;
		//TODO: CHANGE THE VALUE HERE IF YOU SHORTEN THE ARRAY
		float landmark[2 * stasm_NLANDMARKS]; // x,y coords (note the 2)
		float tdata[stasm_NLANDMARKS][2];
		float label[1] = {1};
		Mat labelMat(1,1,CV_32FC1,label);

		if (!stasm_search_single(&foundface, landmark,
			(const char*)img.data, img.cols, img.rows, ipath, "../data"))
		{
			printf("Error in stasm_search_single: %s\n", stasm_lasterr());
			exit(1);
		}

		if (!foundface) {
			printf("No face found in %s\n", ipath);
		}
		else
		{
			printf("FACE FOUND %s\n", ipath);
		}




		//for(int counter=0; counter<(2*stasm_NLANDMARKS); counter++) {
		//	landmarkTrainArray[i][counter] = landmark[counter];
		//}
		int counterX=0, counterY=0;
		float landmarkPtsX[stasm_NLANDMARKS];
		float landmarkPtsY[stasm_NLANDMARKS];
		for(int arraycounter=0;arraycounter<2 * stasm_NLANDMARKS; arraycounter++)
		{
			if(arraycounter%2==0)
				landmarkPtsX[counterX++]=landmark[arraycounter];
			else
				landmarkPtsY[counterY++]=landmark[arraycounter];
		}

		landmarkTrainArray[i][0] = sqrt(pow((landmarkPtsY[59] - landmarkPtsY[65]), 2) + pow((landmarkPtsX[59] - landmarkPtsX[65]), 2));
		landmarkTrainArray[i][1] = sqrt(pow((landmarkPtsY[59] - landmarkPtsY[30]), 2) + pow((landmarkPtsX[59] - landmarkPtsX[30]), 2));
		landmarkTrainArray[i][2] = sqrt(pow((landmarkPtsY[40] - landmarkPtsY[65]), 2) + pow((landmarkPtsX[40] - landmarkPtsX[65]), 2));
		landmarkTrainArray[i][3] = sqrt(pow((landmarkPtsY[68] - landmarkPtsY[74]), 2) + pow((landmarkPtsX[68] - landmarkPtsX[74]), 2));
		landmarkTrainArray[i][4] = sqrt(pow((landmarkPtsY[68] - landmarkPtsY[56]), 2) + pow((landmarkPtsX[68] - landmarkPtsX[56]), 2));
		landmarkTrainArray[i][5] = sqrt(pow((landmarkPtsY[74] - landmarkPtsY[56]), 2) + pow((landmarkPtsX[74] - landmarkPtsX[56]), 2));
		landmarkTrainArray[i][6] = sqrt(pow((landmarkPtsY[18] - landmarkPtsY[22]), 2) + pow((landmarkPtsX[18] - landmarkPtsX[22]), 2));
		landmarkTrainArray[i][7] = sqrt(pow((landmarkPtsY[18] - landmarkPtsY[30]), 2) + pow((landmarkPtsX[18] - landmarkPtsX[30]), 2));
		landmarkTrainArray[i][8] = sqrt(pow((landmarkPtsY[22] - landmarkPtsY[40]), 2) + pow((landmarkPtsX[22] - landmarkPtsX[40]), 2));
		landmarkTrainArray[i][9] = sqrt(pow((landmarkPtsY[22] - landmarkPtsY[30]), 2) + pow((landmarkPtsX[22] - landmarkPtsX[30]), 2));
		landmarkTrainArray[i][10] = sqrt(pow((landmarkPtsY[24] - landmarkPtsY[40]), 2) + pow((landmarkPtsX[24] - landmarkPtsX[40]), 2));


	}

	void returnPointsForTest(const char* ipath)
	{
		cv::Mat_<unsigned char> img(cv::imread(ipath, CV_LOAD_IMAGE_GRAYSCALE));
		if (!img.data)
		{
			printf("Cannot load %s\n", ipath);
			exit(1);
		}

		int foundface;
		//TODO: CHANGE THE VALUE HERE IF YOU SHORTEN THE ARRAY
		float landmark[2 * stasm_NLANDMARKS]; // x,y coords (note the 2)
		float tdata[stasm_NLANDMARKS][2];
		float label[1] = {1};
		Mat labelMat(1,1,CV_32FC1,label);

		if (!stasm_search_single(&foundface, landmark,
			(const char*)img.data, img.cols, img.rows, ipath, "../data"))
		{
			printf("Error in stasm_search_single: %s\n", stasm_lasterr());
			exit(1);
		}

		if (!foundface) {
			printf("No face found in %s\n", ipath);
		}
		else
		{
			printf("FACE FOUND %s\n", ipath);
		}

		int counterX=0, counterY=0;
		float landmarkPtsX[stasm_NLANDMARKS];
		float landmarkPtsY[stasm_NLANDMARKS];
		for(int arraycounter=0;arraycounter<2 * stasm_NLANDMARKS; arraycounter++)
		{
			if(arraycounter%2==0)
				landmarkPtsX[counterX++]=landmark[arraycounter];
			else
				landmarkPtsY[counterY++]=landmark[arraycounter];
		}

		landmarkTestArray[0] = sqrt(pow((landmarkPtsY[59] - landmarkPtsY[65]), 2) + pow((landmarkPtsX[59] - landmarkPtsX[65]), 2));
		landmarkTestArray[1] = sqrt(pow((landmarkPtsY[59] - landmarkPtsY[30]), 2) + pow((landmarkPtsX[59] - landmarkPtsX[30]), 2));
		landmarkTestArray[2] = sqrt(pow((landmarkPtsY[40] - landmarkPtsY[65]), 2) + pow((landmarkPtsX[40] - landmarkPtsX[65]), 2));
		landmarkTestArray[3] = sqrt(pow((landmarkPtsY[68] - landmarkPtsY[74]), 2) + pow((landmarkPtsX[68] - landmarkPtsX[74]), 2));
		landmarkTestArray[4] = sqrt(pow((landmarkPtsY[68] - landmarkPtsY[56]), 2) + pow((landmarkPtsX[68] - landmarkPtsX[56]), 2));
		landmarkTestArray[5] = sqrt(pow((landmarkPtsY[74] - landmarkPtsY[56]), 2) + pow((landmarkPtsX[74] - landmarkPtsX[56]), 2));
		landmarkTestArray[6] = sqrt(pow((landmarkPtsY[18] - landmarkPtsY[22]), 2) + pow((landmarkPtsX[18] - landmarkPtsX[22]), 2));
		landmarkTestArray[7] = sqrt(pow((landmarkPtsY[18] - landmarkPtsY[30]), 2) + pow((landmarkPtsX[18] - landmarkPtsX[30]), 2));
		landmarkTestArray[8] = sqrt(pow((landmarkPtsY[22] - landmarkPtsY[40]), 2) + pow((landmarkPtsX[22] - landmarkPtsX[40]), 2));
		landmarkTestArray[9] = sqrt(pow((landmarkPtsY[22] - landmarkPtsY[30]), 2) + pow((landmarkPtsX[22] - landmarkPtsX[30]), 2));
		landmarkTestArray[10] = sqrt(pow((landmarkPtsY[24] - landmarkPtsY[40]), 2) + pow((landmarkPtsX[24] - landmarkPtsX[40]), 2));

	}