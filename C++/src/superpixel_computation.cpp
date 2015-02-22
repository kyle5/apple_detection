#include "include_in_all.h"
#include "superpixel_computation.h"

using namespace cv;
using namespace std;

Mat compute_superpixels( Mat mat ) {
  Mat mat_copy;
  int resize_factor = 2;
  int org_rows = mat.rows;
  int org_cols = mat.cols;
  int smaller_rows = org_rows / resize_factor;
  int smaller_cols = org_cols / resize_factor;
  if ( org_rows > 1000 || org_cols > 1000 ) {
    resize( mat, mat_copy, Size( smaller_cols, smaller_rows ), 0, 0, INTER_NEAREST );
  } else {
    mat_copy = mat.clone();
  }
  float* image = new float[mat_copy.rows*mat_copy.cols*mat_copy.channels()];
  for (int i = 0; i < mat_copy.rows; ++i) {
		for (int j = 0; j < mat_copy.cols; ++j) {
			image[j + mat_copy.cols*i + mat_copy.cols*mat_copy.rows*0] = mat_copy.at<cv::Vec3b>(i, j)[0];
			image[j + mat_copy.cols*i + mat_copy.cols*mat_copy.rows*1] = mat_copy.at<cv::Vec3b>(i, j)[1];
			image[j + mat_copy.cols*i + mat_copy.cols*mat_copy.rows*2] = mat_copy.at<cv::Vec3b>(i, j)[2];
		}
  }
  vl_uint32* segmentation = new vl_uint32[mat_copy.rows*mat_copy.cols];
  vl_size height = mat_copy.rows;
  vl_size width = mat_copy.cols;
  vl_size channels = mat_copy.channels();
  vl_size region = 30 / resize_factor;
  float regularization = 1000.;
  vl_size minRegion = 10 / resize_factor;
  vl_slic_segment( segmentation, image, width, height, channels, region, regularization, minRegion );
	// convert the 1D segmentation array into a Mat to be used by the following routines 
 	//Mat classifications = Mat::zeros( mat.rows, mat.cols, CV_16U );
	Mat segmentations_cv_16U = create_mat_from_array( mat_copy, segmentation );
  if ( org_rows > 1000 || org_cols > 1000 ) {
    resize( segmentations_cv_16U, segmentations_cv_16U, Size( org_cols, org_rows ), 0, 0, INTER_NEAREST );
  }
	return segmentations_cv_16U;
}
