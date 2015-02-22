count_pgm=`ls -1 build/*.pgm 2>/dev/null | wc -l`
if [ $count_pgm != 0 ]; then
  rm build/*.pgm;
fi
count_exe=`ls -1 build/*.exe 2>/dev/null | wc -l`
if [ $count_exe != 0 ]; then
  rm build/*.exe;
fi
count_ppm=`ls -1 build/*.ppm 2>/dev/null | wc -l`
if [ $count_ppm != 0 ]; then
  rm build/*.ppm;
fi
vl_feat_dir=/home/kyle/Downloads/vlfeat-0.9.19
g++ -g -std=c++0x -o build/routine_superpixel_texture_and_color_majority.exe -Wl,-rpath=/usr/local/lib,-rpath=${vl_feat_dir}/bin/glnxa64/ -I./include -I/home/kyle/Downloads/vlfeat-0.9.19/  -I/usr/local/include/ src/*.cpp -L/usr/local/lib -l opencv_core -l opencv_imgproc -l opencv_highgui -l opencv_features2d -l opencv_nonfree -l opencv_flann -L ${vl_feat_dir}/bin/glnxa64/ -l vl
