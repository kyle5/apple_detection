#ifndef __SETUP_CORRECT_APPLE_PATHS__
#define __SETUP_CORRECT_APPLE_PATHS__

#include <string>
#include <vector>
#include <utility>
#include "include_in_all.h"

using namespace std;
using namespace cv;

struct filepaths_apple_project {
  vector<string> raw_training_paths;
  vector<string> mask_paths;
  vector<string> mos_paths;
  vector<int> all_types_image_numbers;
  vector<int> mask_image_numbers;
  vector<int> mos_image_numbers;
  vector<string> testing_image_paths;
  vector<int> testing_image_numbers;
};

vector< pair<int, int> > sort_vector_with_indices( vector<int> vector_to_sort );
vector<string> create_paths( string path_str, vector<string> fnames );
string get_filename( string filename );
vector<int> get_image_numbers( vector<string> image_paths, string image_name_root );
vector<string> setup_filepaths( string filepath_root, string filepath_ext, vector<int> corresponding_numbers );
vector<string> correlate_image_paths( vector<int> match_numbers, vector<int> to_be_matched_numbers, vector<string> to_be_matched_paths );
struct filepaths_apple_project setup_correct_apple_paths( string dataset_path );
void sort_images_and_image_numbers( vector< string > &testing_paths_preliminary, vector< int > &numbers_testing_images );
void prune_image_paths_and_image_numbers( vector< string > &testing_paths_preliminary, vector< int > &numbers_testing_images );
#endif
