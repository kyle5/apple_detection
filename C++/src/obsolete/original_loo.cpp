
	// for loop over loo
	for( int i = 0; i < (int) all_valid_texture_features.size(); i++ ) {
		string input_img_rgb_uchar_path = fpaths.raw_paths_match_mask[i];
		Mat input_img_rgb_uchar = imread( input_img_rgb_uchar_path );
		transpose( input_img_rgb_uchar, input_img_rgb_uchar );
		flip(input_img_rgb_uchar, input_img_rgb_uchar, 0);
		int img_number_real = fpaths.mask_paths_image_numbers[i];
		std::ostringstream stringStream;
    stringStream << setw(6) << setfill('0') << img_number_real;
    string img_number = stringStream.str();
    string apple_type_and_img_number = apple_type + string("/") + img_number;
		
		Mat src_gray;
		cvtColor(input_img_rgb_uchar, src_gray, CV_BGR2GRAY);
		Mat pixels_over_25 = Mat::zeros(src_gray.rows, src_gray.cols, CV_8U), pixels_over_50 = Mat::zeros(src_gray.rows, src_gray.cols, CV_8U);
		threshold( src_gray, pixels_over_25, 25, 255, THRESH_BINARY );
		threshold( src_gray, pixels_over_50, 50, 255, THRESH_BINARY );
		Mat superpixel_img = superpixel_imgs[i];
		if (superpixel_img.rows * superpixel_img.cols == 0) throw runtime_error( "no superpixel image? the superpixel image size is 0?" );
    
    // BEGIN : TODO : LOO feature set creation
    // Before classification needed: struct with :
    //  // feature space, combined_labels, pca for texture transformation, perhaps the features for the current image
    
    // inputs: vector<vector> Mat color, vector<vector> Mat texture, vector<vector> of classifications for groups
    // outputs: normalized/pca'ed feature space, pca transformation for texture features
    vector<vector<Mat> > complete_texture_features;
    vector<vector<Mat> > complete_color_features;
    vector< vector<orchard_classification> > complete_labels;
    complete_texture_features.push_back(all_valid_texture_features); complete_texture_features.push_back(all_invalid_texture_features);
    complete_color_features.push_back(all_valid_color_features); complete_color_features.push_back(all_invalid_color_features);
    
    struct classifier_features cd_apple_segmentation = setup_classifier_features( complete_texture_features, complete_color_features, complete_labels );
    create_svm( classifier_features );
    create_random_forest( classifier_features );
    
		Mat combined_loo_train_all_valid_texture_features( 0, all_valid_texture_features[0].cols, all_valid_texture_features[0].type() );
		Mat combined_loo_train_all_invalid_texture_features( 0, all_invalid_texture_features[0].cols, all_invalid_texture_features[0].type() );
		Mat combined_loo_train_all_valid_color_features( 0, all_valid_color_features[0].cols, all_valid_color_features[0].type() );
		Mat combined_loo_train_all_invalid_color_features( 0, all_invalid_color_features[0].cols, all_invalid_color_features[0].type() );
    // END : TODO : LOO feature set creation
    
    /*
    int max_pca_dimensions = 6;
    Mat feature_set = create_feature_set_excluding_index(  );
    Mat normalizing_factors_texture = create_normalizing_factors( feature_set );
    normalized_texture_
		feature_set = apply_normalizing_factors( feature_set, normalizing_factors );
    pca_transformation = create_pca_transformation( normalized_feature_set );
    apply_pca_transformation( normalized_feature_set, pca_transformation, max_pca_dimensions );
    create_classifier(  );
    query_classifier(  );
    */
    
		vector<Mat> loo_train_all_invalid_color_features;
		for( int j = 0; j < (int) all_valid_texture_features.size(); j++ ) {
			if( i == j ) continue;
			if (combined_loo_train_all_valid_texture_features.rows == 0 || combined_loo_train_all_valid_texture_features.cols == 0) {
				combined_loo_train_all_valid_texture_features = all_valid_texture_features[j];
			} else {
				vconcat( combined_loo_train_all_valid_texture_features, all_valid_texture_features[j], combined_loo_train_all_valid_texture_features );
			}
			if (combined_loo_train_all_invalid_texture_features.rows == 0 || combined_loo_train_all_invalid_texture_features.cols == 0) {
				combined_loo_train_all_invalid_texture_features = all_invalid_texture_features[j];
			} else {
				vconcat( combined_loo_train_all_invalid_texture_features, all_invalid_texture_features[j], combined_loo_train_all_invalid_texture_features );
			}
			if (combined_loo_train_all_valid_color_features.rows == 0 || combined_loo_train_all_valid_color_features.cols == 0) {
				combined_loo_train_all_valid_color_features = all_valid_color_features[j];
			} else {
				vconcat( combined_loo_train_all_valid_color_features, all_valid_color_features[j], combined_loo_train_all_valid_color_features );
			}
			if (combined_loo_train_all_invalid_color_features.rows == 0 || combined_loo_train_all_invalid_color_features.cols == 0) {
				combined_loo_train_all_invalid_color_features = all_invalid_color_features[j];
			} else {
				vconcat( combined_loo_train_all_invalid_color_features, all_invalid_color_features[j], combined_loo_train_all_invalid_color_features );
			}
		}

		vector<vector<long> > grid_keypoints_processing = all_superpixel_grid_keypoints_processing[i];
		Mat combined_texture_features, combined_color_features;
		vconcat( combined_loo_train_all_valid_texture_features, combined_loo_train_all_invalid_texture_features, combined_texture_features );
		vconcat( combined_loo_train_all_valid_color_features, combined_loo_train_all_invalid_color_features, combined_color_features );

		vector<orchard_classification> valid(combined_loo_train_all_valid_texture_features.rows, APPLE);
		vector<orchard_classification> invalid(combined_loo_train_all_invalid_texture_features.rows, LEAF);
		vector<orchard_classification> combined_labels;
		combined_labels.insert( combined_labels.end(), valid.begin(), valid.end() );
		combined_labels.insert( combined_labels.end(), invalid.begin(), invalid.end() );

		double total_valid = (double) valid.size();
		double total_invalid = (double) invalid.size();
		double total = total_valid + total_invalid;
		double percent_apple = total_valid / total;
		double percent_non_apple = total_invalid / total;

		// compute pca
		// do pca on combined
		Mat empty, normalized_combined_texture_and_color_descriptors_train_64F, pca_descriptors_train;
		vector<Mat> norm_factors_texture = compute_mean_and_std( combined_texture_features );
		vector<Mat> norm_factors_color = compute_mean_and_std( combined_color_features );
		fprintf( stderr, "texture norm training\n" );
		Mat normalized_texture_descriptors_train = normalize_mat( combined_texture_features, norm_factors_texture );
		Mat normalized_color_descriptors_train = normalize_mat( combined_color_features, norm_factors_color );
		fprintf( stderr, "before pca\n" );
		PCA pca( normalized_texture_descriptors_train, empty, CV_PCA_DATA_AS_ROW, 6 );
		pca.project( normalized_texture_descriptors_train, pca_descriptors_train );
    fprintf(stderr, "ccca");
    printf( "pca_descriptors_train: r: %d c: %d\nnormalized_color_descriptors_train: r: %d: c: %d", pca_descriptors_train.rows, pca_descriptors_train.cols, normalized_color_descriptors_train.rows, normalized_color_descriptors_train.cols );
		hconcat( pca_descriptors_train, normalized_color_descriptors_train, normalized_combined_texture_and_color_descriptors_train_64F ); // TODO : fix error on this line
    fprintf(stderr, "cccb");

		Mat pca_descriptors_process, normalized_combined_texture_and_color_descriptors_process_64F;
		Mat all_superpixel_features_combined = all_superpixel_features_vec[i];
		Mat normalized_texture_descriptors_process = normalize_mat( all_superpixel_features_combined, norm_factors_texture );
		Mat all_superpixel_color_features_combined = all_superpixel_color_features_vec[i];
		Mat normalized_color_descriptors_process = normalize_mat( all_superpixel_color_features_combined, norm_factors_color );
		pca.project( normalized_texture_descriptors_process, pca_descriptors_process );
    fprintf(stderr, "bbba");
		hconcat( pca_descriptors_process, normalized_color_descriptors_process, normalized_combined_texture_and_color_descriptors_process_64F );
    fprintf(stderr, "bbbb");

		fprintf(stderr, "before matching features\n");

		vector<orchard_classification> all_supperpixel_classifications_combined = query_superpixel_features( combined_labels, normalized_combined_texture_and_color_descriptors_process_64F, normalized_combined_texture_and_color_descriptors_train_64F );

		if ( ((int)all_supperpixel_classifications_combined.size()) != all_superpixel_features_combined.rows ) throw runtime_error("features are not equal!?");
		fprintf(stderr, "after matching features\n");
		
		fprintf( stderr, "all_superpixel_features_combined: rows: %d: cols: %d\n", all_superpixel_features_combined.rows, all_superpixel_features_combined.cols );
		vector< Mat > all_superpixel_features = split_features( all_superpixel_features_combined, grid_keypoints_processing );
		fprintf(stderr, "aaaaaabb\n");

		vector<vector<orchard_classification>> all_superpixel_classifications( (int) all_superpixel_features.size(), vector<orchard_classification>() );
		vector<bool> valid_points( (int) all_superpixel_features.size(), false );
		fprintf(stderr, "aaaaa\n");
		
    vector<double> superpixel_dominant_angle( (int) all_superpixel_features.size(), 0 );    

    Mat src_gray_blurred;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    int c;
    GaussianBlur( src_gray, src_gray_blurred, Size(3,3), 0, 0, BORDER_DEFAULT );
    Mat grad_x, grad_y;
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

		Mat cur_valid_img = valid_imgs[i];
		int cur_start = 0;
		for ( int k = 0; k < (int) all_superpixel_features.size(); k++ ) {
			Mat cur_superpixel_features = all_superpixel_features[k];
			int cur_end = cur_start + cur_superpixel_features.rows - 1;
			int step = 0;
			if ( cur_superpixel_features.rows != 0 && cur_superpixel_features.cols != 0 ) {
				step = cur_superpixel_features.rows;
			}
			int count_valid_superpixel_grid = 0;
      double total_x_cur = 0, total_y_cur = 0;
			for( int j = 0; j < step; j++ ) {
				int cur_pt_idx = grid_keypoints_processing[k][j];
				int cur_y = cur_pt_idx % pixels_over_25.rows;
				int cur_x = cur_pt_idx / pixels_over_25.rows;
				int cur_valid = cur_valid_img.at<uchar>( cur_y, cur_x );
				orchard_classification cur_classification;
				bool pixel_above_25 = pixels_over_25.at<uchar>( cur_y, cur_x ) == 255;
				if ( pixel_above_25 ) {
					cur_classification = all_supperpixel_classifications_combined[cur_start + j];
				} else {
					cur_classification = TOO_DARK;
				}
				all_superpixel_classifications[k].push_back( cur_classification );
				if (cur_valid > 0) {
					count_valid_superpixel_grid++;
				}
				short x_grad_cur = grad_x.at<short>( cur_y, cur_x );
				short y_grad_cur = grad_y.at<short>( cur_y, cur_x );
        total_x_cur += (int) x_grad_cur;
        total_y_cur += (int) y_grad_cur;
			}
      double angle_cur_superpixel = atan2( total_y_cur, total_x_cur );
      superpixel_dominant_angle[k] = angle_cur_superpixel;

			if ( count_valid_superpixel_grid > ( ( (double) step) / 2 ) ) {
				valid_points[k] = true;
			}
			cur_start += step;
		}
		fprintf(stderr, "aaaab\n");
		fprintf( stderr, "percent\napple: %.2f\nnonapple: %.2f\n", percent_apple, percent_non_apple );
		// for each level, for each superpixel, compute the number of valid superpixels
			// save the resulting image classifications to directories
		// later ... compute the recall and precision rates for each image
		fprintf(stderr, "aaabb\n");
		
		int total_levels = 10;
		vector<double> levels;
		for(int j = 1; j < total_levels; j++) {
			double cur_level = ( (double) 0.25 ) / ((double)total_levels) * ((double)j) + 0.75;
			levels.push_back( cur_level );
		}
		// get the "correct" classification for the superpixel : majority inside the superpixel are valid

		Mat tp_fp_tn_fn = Mat::zeros( (int) levels.size(), 5, CV_32S );
		for (int j = 0; j < (int) levels.size(); j++) {
			double cur_level = levels[j];
			double multiplier_apple = 1 - cur_level;
			double multiplier_everything_else = cur_level;
			
			char * temp_1 = (char*) malloc( sizeof(char)*300 );
			sprintf( temp_1, "%s/input_img_rgb_uchar_path.ppm", apple_type_and_img_number.c_str() );
			fprintf(stderr, "aaabc\n");
      
      Mat superpixel_dominant_angle_mat = Mat::zeros( input_img_rgb_uchar.rows, input_img_rgb_uchar.cols, CV_8UC3 );

			int cur_tp = 0, cur_fp = 0, cur_tn = 0, cur_fn = 0;
			vector<orchard_classification> cur_superpixel_classifications((int) all_superpixel_classifications.size(), TREE_TRUNK);
			for( int k = 0; k < all_superpixel_classifications.size(); k++ ) { // for superpixel
				orchard_classification cur_classification = TREE_TRUNK;
				double apple_count = 0, leaf_count = 0, bark_count = 0, dark_count = 0;
				for( int a = 0; a < all_superpixel_classifications[k].size(); a++ ) { // for grid point
					orchard_classification ct = all_superpixel_classifications[k][a];
					if( ct == APPLE ) {apple_count++;} if( ct == LEAF ) {leaf_count++; } else if( ct == TREE_TRUNK) {bark_count++; } else { dark_count++; }
				}
				if ( ( ( multiplier_apple * apple_count / percent_apple ) >= ( multiplier_everything_else * leaf_count / percent_non_apple ) && (multiplier_apple * apple_count) >= (multiplier_everything_else * bark_count ) ) && apple_count > 0 ) {
					cur_superpixel_classifications[k] = APPLE;

				} else if ( leaf_count >= apple_count && leaf_count >= bark_count && leaf_count > 0 ) {
					cur_superpixel_classifications[k] = LEAF;
				} else if( bark_count > dark_count ) {
					cur_superpixel_classifications[k] = TREE_TRUNK;
				} else {
					cur_superpixel_classifications[k] = TOO_DARK;
				}
				if ( cur_superpixel_classifications[k] == APPLE ) {
					if( valid_points[k] ) {
						cur_tp++;
					} else {
						cur_fp++;
					}
				} else {
					if( valid_points[k] ) {
						cur_fn++;
					} else {
						cur_tn++;
					}
				}
			}
			tp_fp_tn_fn.at<int>( j, 0 ) = (int) (cur_level*100);
			tp_fp_tn_fn.at<int>( j, 1 ) = cur_tp;
			tp_fp_tn_fn.at<int>( j, 2 ) = cur_fp;
			tp_fp_tn_fn.at<int>( j, 3 ) = cur_tn;
			tp_fp_tn_fn.at<int>( j, 4 ) = cur_fn;
			// display the classifications

      char gradient_img_name_cur[512];
			sprintf(gradient_img_name_cur, "%s/gradient_classifications_level_%.2f.ppm", apple_type_and_img_number.c_str(), levels[j] );
			Mat superpixel_classifications = display_superpixel_classifications( cur_superpixel_classifications, input_img_rgb_uchar, superpixel_img, superpixel_dominant_angle, gradient_img_name_cur );
			char buf[512];
			// Kyle : TODO write out the probability of the superpixel classification along with the segmentation
			sprintf(buf, "%s/superpixel_classifications_level_%.2f.ppm", apple_type_and_img_number.c_str(), levels[j] );
      imwrite( buf, superpixel_classifications );
			fprintf(stderr, "aaabx\n");
		}
		char create_image_results_folder[300], pre_recall_filepath[300];
		sprintf( create_image_results_folder, "mkdir %s", apple_type_and_img_number.c_str() );
		system( create_image_results_folder );
		sprintf( pre_recall_filepath, "%s/precision_recall_data.yml", apple_type_and_img_number.c_str() );
		cv::FileStorage storage_pre_recall( pre_recall_filepath, cv::FileStorage::WRITE );
		storage_pre_recall << "tp_fp_tn_fn" << tp_fp_tn_fn;
		storage_pre_recall.release();
	}

