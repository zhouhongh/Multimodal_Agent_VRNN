SBU-Kinect-Interaction dataset v2.0
================================================

This package contains the SBU-Kinect-Interaction dataset version 2.0. It comprises of RGB-D video sequences of humans performing interaction activities that are recording using the Microsoft Kinect sensor. This dataset was originally recorded for a class project, and it must be used only for the purposes of research. If you use this dataset in your work, please cite the following paper.

	Kiwon Yun, Jean Honorio, Debaleena Chattopadhyay, Tamara L. Berg, and Dimitris Samaras, The 2nd International Workshop on Human Activity Understanding from 3D Data at Conference on Computer Vision and Pattern Recognition (HAU3D-CVPRW), CVPR 2012

For questions and comments, please email Kiwon Yun <kyun@cs.stonybrook.edu>
Data written: v0.1, Dec 23, 2013	
Revision: v2.0, Apr 16, 2015 - noisy data for the MILBoost experiment was added.

================================================

Overview

	All videos are recorded in the same laboratory environment. Seven participants (s01-s07) performed activities and the dataset is composed 21 sets, where each set contains videos of a pair of different persons performing all eight interactions. Note that in most interactions, one person is acting and the other person is reacting. Each set contains one or two sequences per action category. The entire dataset has a total of 300 interactions approximately.

	The noisy version is also included in the dataset. The noisy version of the data was used to evaluate MILBoost experiments (Set2 in Table 2 of the paper). We segment the original recorded sequence by starting from five frame earlier than the original start frame and ending five frame later than the original final frame. Noisy data contains more irrelevant actions since participants randomly moved between action categories when we collected data. 
	
	For the general purpose of evaluation, we recommend you to download 'clean version'.
	
Download links
	
	1) Clean version (original data)
	
	Set01: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s02.zip
	Set02: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s03.zip
	Set03: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s07.zip
	Set04: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s01.zip
	Set05: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s03.zip
	Set06: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s06.zip
	Set07: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s07.zip
	Set08: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s02.zip
	Set09: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s04.zip
	Set10: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s05.zip
	Set11: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s06.zip
	Set12: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s02.zip
	Set13: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s03.zip
	Set14: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s06.zip
	Set15: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s05s02.zip
	Set16: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s05s03.zip
	Set17: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s02.zip
	Set18: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s03.zip
	Set19: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s04.zip
	Set20: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s07s01.zip
	Set21: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s07s03.zip

	2) Noisy data (Set 2 in Table 2)
	
	The download links are as similar as original sets, but with '_noisy' as postfix for each file. For example, the link of the noisy data for Set01 is:
	http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s02_noisy.zip
	
Contents
	
	Each .zip file contains one directory per activity. Each activity directory has one or two directories per sequence.
	Each sequence has the following files.
	
	1) depth_xxxxxx.png: depth data file corresponding to frame number xxxxxx of activity.
		Depth image are 640 × 480 pixels.
		
	2) rgb_xxxxxx.png: RGB data file corresponding to frame number xxxxxx of activity.
		Color image are 640 × 480 pixels.
		
	3) skeleton_pos.txt: Skeleton position data
	
		Skeleton data consists of 15 joints per person. Each row follows the following format.
	
			Frame#,PA(1),PA(2),...,PA(15),PB(1),PB(2),...,PB(15)

			PA(i)   => position of ith joint (x,y,z) for the subject A located at left
			PB(i)   => position of ith joint (x,y,z) for the subject B located at right

    		x and y are normalized as [0,1] while z is normalized as [0,7.8125]

			Joint number -> Joint name
				1 -> HEAD
				2 -> NECK
				3 -> TORSO
				4 -> LEFT_SHOULDER
				5 -> LEFT_ELBOW
				6 -> LEFT_HAND
				7 -> RIGHT_SHOULDER
				8 -> RIGHT_ELBOW
				9 -> RIGHT_HAND
				10 -> LEFT_HIP
				11 -> LEFT_KNEE
				12 -> LEFT_FOOT
				13 -> RIGHT_HIP
				14 -> RIGHT_KNEE
				15 -> RIGHT_FOOT
				
Details		
		
	To classify eight action categories, we train SVMs in a one-vs-all fashion, and evaluation is done by 5-fold cross validation, and we randomly split the dataset into 5 folds of 4-5 two-actor sets each. The following shows which sets we selected for each fold:
	
		Fold 1 - Set# 01, 09, 15, 19 (s01s02, s03s04, s05s02, s06s04)
		Fold 2 - Set# 05, 07, 10, 16 (s02s03, s02s07, s03s05, s05s03)
		Fold 3 - Set# 02, 03, 20, 21 (s01s03, s01s07, s07s01, s07s03)
		Fold 4 - Set# 04, 06, 08, 11 (s02s01, s02s06, s03s02, s03s06)
		Fold 5 - Set# 12, 13, 14, 17, 18 (s04s02, s04s03, s04s06, s06s02, s06s03)
		
	The skeleton data was normalized. In order to extract the original position of the joints, the following equations are needed:
	
	original_X = 1280 - (normalized_X .* 2560);
	original_Y = 960 - (normalized_Y .* 1920);
	original_Z = normalized_Z .* 10000 ./ 7.8125;
	
	You are able to find the pixel locations of the joints in the depth or the RGB images using built-in Kinect functions such as NuiTransform, SkeletonToDepthImage or NuiTransformSkeletonToDepthImage.