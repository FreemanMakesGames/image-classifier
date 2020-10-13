import argparse

from data_extraction import get_data_from_images


argparser = argparse.ArgumentParser( description = "Extract data from images." )
argparser.add_argument( "--img-dir", "-id" )
argparser.add_argument( "--save-path", "-sp" )
args = argparser.parse_args()

img_dir = args.img_dir
save_path = args.save_path

get_data_from_images( img_dir, save_path )

