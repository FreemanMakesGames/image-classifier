import sys
import argparse

from images_extraction import get_images_from_game_dir

argparser = argparse.ArgumentParser( description = "Extract images from videos." )
argparser.add_argument( "--raw-data-dir", "-rdd" )
argparser.add_argument( "--save-dir", "-sd" )
args = argparser.parse_args()

raw_data_dir = args.raw_data_dir
save_dir = args.save_dir

# Iterate all genre directories.
for genre_entry in os.scandir( raw_data_dir ):

    assert genre_entry.is_dir()

    for game_entry in os.scandir( genre_entry.path ):

        # Skip info.json etc.
        if not game_entry.is_dir():
            continue

        get_images_from_game_dir( game_entry.path, img_per_game )

