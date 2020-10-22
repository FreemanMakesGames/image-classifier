import os
import sys
import shutil
import argparse

from images_extraction import get_images_from_game_dir


# Parse args.
argparser = argparse.ArgumentParser( description = "Extract images from videos." )
argparser.add_argument( "--raw-data-dir", "-rdd" )
argparser.add_argument( "--save-dir", "-sd" )
argparser.add_argument( "--img-per-game", "-ipg", type = int )
argparser.add_argument( "--initial-skip-sec", "-iss", type = int )
argparser.add_argument( "--gap-sec", "-gs", type = int )
args = argparser.parse_args()
raw_data_dir = args.raw_data_dir
save_dir = args.save_dir
img_per_game = args.img_per_game
initial_skip_sec = args.initial_skip_sec
gap_sec = args.gap_sec


# Nuke saving dir.
if os.path.isdir( save_dir ):
    shutil.rmtree( save_dir )

# Iterate all genre directories.
for genre_entry in os.scandir( raw_data_dir ):

    if not genre_entry.is_dir():
        print( "Warning: A non-dir among genre directories?" )
        continue

########Skip third-person shooters for now#################
    if genre_entry.name == "third-person-shooter":
        continue
###########################################################

    for game_entry in os.scandir( genre_entry.path ):

        # Skip info.json etc.
        if not game_entry.is_dir():
            continue

        get_images_from_game_dir( game_entry.path, game_entry.path.replace( raw_data_dir, save_dir, 1 ), img_per_game, initial_skip_sec, gap_sec )

