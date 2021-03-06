#include "classify_images.h"
#include "feature_computation.h"
#include "utility.h"

int orchard_class_to_int( orchard_classification cur_class ) {
  int ret = 0;
  switch ( cur_class ) {
    case APPLE : ret=1;
    case LEAF : ret=2;
    case TREE_TRUNK : ret=3;
    case TOO_DARK : ret=4;
    case VERY_TOP : ret=5;
    case TOP : ret=6;
    case BOTTOM : ret=7;
    case VERY_BOTTOM : ret=8;
  }
  return ret;
}

void filter_features_by_valid_superpixels(Mat &texture_features, Mat &color_features, vector<vector<long> > &grid_indices, const vector<Mat> &superpixels_to_check_vec) {
  if ( ( (int) superpixels_to_check_vec.size() ) == 0 ) return;
  verify_type( texture_features.type(), CV_64F );
  verify_type( color_features.type(), CV_64F );
  vector<int> superpixels_to_keep( (int) grid_indices.size(), 1 );
  for ( int i = 0; i < (int) superpixels_to_check_vec.size(); i++ ) { // for each superpixel designation
    Mat superpixels_to_check = superpixels_to_check_vec[i];
    if ( superpixels_to_check.rows == 0 ) continue; // There is no filtering; The superpixels are all valid...
    if ( superpixels_to_check.rows != (int) grid_indices.size() ) throw runtime_error( "( superpixels_to_check.rows != (int) grid_indices.size() )" );
    verify_type( superpixels_to_check.type(), CV_8U );
    for ( int j = 0; j < (int) superpixels_to_check.rows; j++ ) { // for each superpixel
      uchar check_superpixel_b = superpixels_to_check.at<uchar>(j, 0);
      if ( !( (int) check_superpixel_b ) ) {
        superpixels_to_keep[j] = 0;
      }
    }
  }
  vector<int> indices_of_matrices_to_keep;
  int cur_start = 0, cur_end = -1;
  for ( int i = 0; i < (int) grid_indices.size(); i++ ) {
    cur_start = cur_end + 1;
    cur_end = cur_start + (int) grid_indices[i].size() - 1;
    if ( superpixels_to_keep[i] ) {
      for ( int k = cur_start; k <= cur_end; k++ ) {
        indices_of_matrices_to_keep.push_back( k );
      }
    } else {
      grid_indices[i] = vector<long>();
    }
  }
  sort( indices_of_matrices_to_keep.begin(), indices_of_matrices_to_keep.end() );
  indices_of_matrices_to_keep.erase( unique( indices_of_matrices_to_keep.begin(), indices_of_matrices_to_keep.end() ), indices_of_matrices_to_keep.end() );
  Mat texture_features_new = Mat::zeros( (int) indices_of_matrices_to_keep.size(), texture_features.cols, CV_64F );
  Mat color_features_new = Mat::zeros( (int) indices_of_matrices_to_keep.size(), color_features.cols, CV_64F );
  for ( int i = 0; i < indices_of_matrices_to_keep.size(); i++ ) {
    int cur_idx = indices_of_matrices_to_keep[i];
    for ( int j = 0; j < texture_features.cols; j++ ) texture_features_new.at<double>(i, j) = texture_features.at<double>(cur_idx, j);
    for ( int j = 0; j < color_features.cols; j++ ) color_features_new.at<double>(i, j) = color_features.at<double>(cur_idx, j);
  }
  texture_features = texture_features_new;
  color_features = color_features_new;
}

struct classification_result classify_image( struct classifier_features cf_struct, string raw_img_path, struct features_full_images features_apple_images, string feature_type, vector<Mat> &superpixels_to_check_vec, int idx_loo ) {

  struct classification_result ret;
  Mat superpixel_image, texture_features, color_features, valid_superpixels;
  vector< vector<long> > grid_indices;
  float resize_factor = 1;
  if ( idx_loo == -1 ) {
    Mat input_img_rgb_uchar_train, input_img_gray_uchar_train_mask;
    input_img_rgb_uchar_train = read_image_and_flip_90( raw_img_path, resize_factor );
    struct features_computed features_cur_img = compute_features_single_image( input_img_rgb_uchar_train, feature_type );
    ret.superpixel_image = features_cur_img.superpixel_image;
    texture_features = features_cur_img.texture_features;
    color_features = features_cur_img.color_features;
    grid_indices = features_cur_img.grid_indices;
    // valid_superpixels = features_cur_img.superpixels_value_is_valid_8U;
  } else {
    grid_indices = features_apple_images.indices[idx_loo];
    color_features = features_apple_images.all_color_features[idx_loo].clone();
    texture_features = features_apple_images.all_texture_features[idx_loo].clone();
    ret.superpixel_image = features_apple_images.superpixel_imgs[idx_loo].clone();
    // valid_superpixels = features_apple_images.valid_superpixels[idx_loo].clone();
  }
  
  texture_features.convertTo( texture_features, CV_64F );
  color_features.convertTo( color_features, CV_64F );

  // filter by superpixels_to_check_vec
  filter_features_by_valid_superpixels( texture_features, color_features, grid_indices, superpixels_to_check_vec );
  
  Mat superpixel_image2 = ret.superpixel_image.clone();
  Mat visible_superpixels = draw_grid_superpixels<long>( superpixel_image2, grid_indices );
  ostringstream temp;
  temp << "green/sup_visible_" << idx_loo << "_" << (int) superpixels_to_check_vec.size() << ".pgm";
  string temp_str = temp.str();
  imwrite( temp_str.c_str(), visible_superpixels );
  
  // normalize and pca using the normalization and pca previously computed
  Mat pca_descriptors_process_64F, normalized_combined_texture_and_color_descriptors_process_64F, normalized_texture_descriptors_process, normalized_color_descriptors_process, normalized_color_descriptors_process_64F;
  normalized_texture_descriptors_process = normalize_mat( texture_features, cf_struct.norm_factors_texture );
  normalized_color_descriptors_process = normalize_mat( color_features, cf_struct.norm_factors_color );
  check_for_nan( color_features );
  vector<Mat> color_features_mean_and_std = compute_mean_and_std( color_features );
  cerr << " color_features_mean_and_std[0]: " << color_features_mean_and_std[0] << endl;
  cerr << " color_features_mean_and_std[1]: " << color_features_mean_and_std[1] << endl;
  
  // use PCA to project the points into adjusted space
  cf_struct.pca_transform.project( normalized_texture_descriptors_process, pca_descriptors_process_64F );
  normalized_color_descriptors_process.convertTo( normalized_color_descriptors_process_64F, CV_64F );
  normalized_combined_texture_and_color_descriptors_process_64F = pca_descriptors_process_64F;
  combine_features_horizontally_ref( normalized_combined_texture_and_color_descriptors_process_64F, normalized_color_descriptors_process_64F );
  
  // TODO : ex : setup classification function with classification method (random forests, feature space, etc.) as input
  vector<vector<uchar> > all_superpixel_classifications_combined = query_superpixel_features( normalized_combined_texture_and_color_descriptors_process_64F, cf_struct, grid_indices );
  
  Mat visible_superpixels_classified = draw_grid_superpixels<uchar>( superpixel_image2, all_superpixel_classifications_combined );
  ostringstream temp2;
  temp2 << "green/sup_classified_" << idx_loo << ".pgm";
  string temp2_str = temp2.str();
  imwrite( temp2_str.c_str(), visible_superpixels_classified );
  
  ret.classifications = all_superpixel_classifications_combined;
  return ret;
}

Mat calculate_percentage_by_type( const Mat superpixel_image, vector<vector<uchar> > classifications ) {
  // return : Mat : rows == # superpixels : cols == # of types
  int max_value = compute_max_value( superpixel_image );
  if (max_value != (int) classifications.size() ) {
    fprintf( stderr, "mv: %d\nclass_size: %d\n", max_value, (int) classifications.size() );
    throw runtime_error( "superpixels not equal to classifications length!?!?!?" );
  }
  // TODO : ex : somehow need to find the number of classifications
  // // should have a number of classifications, a pairing between strings and classifications, and a color for each classification
  int total_orchard_classifications = 8;
  Mat labels_assigned = Mat( max_value, total_orchard_classifications, CV_32F, Scalar(0) );
  // get the percentages for each type of value
  int i, j;
  // for each superpixel
  for (i = 0; i < (int) classifications.size(); i++ ) {
    const vector<uchar> classifications_sp = classifications[i];
    float total_classifications = (float) classifications_sp.size();
    // for each possible value : get the posible classifications
    for ( j = 0; j < (int) classifications_sp.size(); j++ ) {
      uchar val_uchar = classifications_sp[j];
      int val_int = (int) val_uchar;
      int idx_offset_0 = val_int - 1;
      labels_assigned.at<float>( i, idx_offset_0 )++;
    }
    for ( j = 0; j < (int) labels_assigned.cols; j++ )
      if (total_classifications != 0) {
        labels_assigned.at<float>(i, j) /= total_classifications;
      } // else labels_assigned is  zeros already
  }
  return labels_assigned;
}

int get_total_indices( vector< vector< long > > grid_indices ) {
  int total_count = 0;
  for (int i = 0; i < (int) grid_indices.size(); i++) {
    total_count += (int) grid_indices[i].size();
  }
  return total_count;
}

vector<vector<uchar> > query_superpixel_features( Mat query_features, classifier_features cf_struct, const vector< vector<long> > &grid_indices ) {
  int total_count = get_total_indices( grid_indices );
  if ( total_count != query_features.rows ) {
    cout << "total_count:  " << total_count << endl;
    cout << "query_features.rows:  " << query_features.rows << endl;
    throw runtime_error( "Need to account for keypoints that are near to edge of image!!!" );
  }
  vector<vector<uchar> > ret( (int) grid_indices.size(), vector<uchar>() );
  int type_of_classification = 1;
  if ( type_of_classification == 1 ) {
    // BFMatcher;
    Mat cur_features, combined_descriptors;
    query_features.convertTo( cur_features, CV_32F );
    cf_struct.feature_space.convertTo( combined_descriptors, CV_32F );
    int n_knn = 5;
    Mat c_cur = Mat::zeros( n_knn, 1, CV_8U );
    Mat distances;
    BFMatcher matcher( NORM_L2, false );
    vector< vector< DMatch > > matches = vector< vector< DMatch > >();
    Mat combined_descriptors_64F, cur_features_64F;
    combined_descriptors.convertTo( combined_descriptors_64F, CV_64F );
    cur_features.convertTo( cur_features_64F, CV_64F );

    check_for_nan( combined_descriptors_64F );
    check_for_nan( cur_features_64F );
    
    matcher.knnMatch( cur_features, combined_descriptors, matches, n_knn );
    if ( (int) matches.size() != cur_features.rows || cur_features.rows != get_total_indices( grid_indices ) ) {
      throw runtime_error("outer features don't match");
    }
    for ( int i = 0; i < (int) matches.size(); i++ ) {
      if ( matches[i].size() != n_knn ) {
        cerr << "matches[i].size(): " << matches[i].size() << endl;
        cerr << "n_knn: " << n_knn << endl;
        throw runtime_error("inner features don't match");
      }
    }
    int cur_idx = -1;
    for ( int i = 0; i < (int) grid_indices.size(); i++ )
    for ( int j = 0; j < (int) grid_indices[i].size(); j++)
    {
      cur_idx++;
      uchar max_value = 0;
      uchar max_type = 0;
	    for ( int k = 0; k < n_knn; k++) {
		    int idx = matches[cur_idx][k].trainIdx;
        
        if ( idx > cf_struct.combined_labels.rows ) {
          cerr << "matches:   " << matches.size() << endl;
          cerr << "cf_struct.combined_labels.rows:   " << cf_struct.combined_labels.rows << endl;
          cerr << "idx:   " << idx << endl;
        }
		    uchar cur_type = cf_struct.combined_labels.at<uchar>( idx, 0 );
        uchar cur_value = ++c_cur.at<uchar>( cur_type, 0 );
        if ( cur_value > max_value ) {
          max_value = cur_value;
          max_type = cur_type;
        }
	    }
      for ( int k = 0; k < n_knn; k++) c_cur.at<uchar>( k, 0 ) = 0;
      ret[i].push_back( max_type );
    }
  } else {
    // Random Forest
  }
  return ret;
}
