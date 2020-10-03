import os
import shutil
import pathlib
import cv2 as cv

project_dir = "/home/insight/Documents/Projects/image-classifier"
raw_data_dir = os.path.join( project_dir, "raw-data" )
processed_data_dir = os.path.join( project_dir, "processed-data" )

initial_skip_sec = 60  # How much to skip the start of each video.
gap_sec = 20  # Delay in seconds between screenshots.
img_per_game = 50

def process_images_from_dir( dir_path ):

    # The directory with images to process has to be under raw_data_dir
    assert dir_path.find( raw_data_dir ) == 0

    for entry in os.scandir( dir_path ):

        entry_path = os.path.join( dir_path, entry.name )

        if entry.is_dir():
            process_images_from_dir( entry_path )
            continue

        name, ext = os.path.splitext( entry.name )

        if ext != ".mp4":
            continue

        save_dir = processed_data_dir
        for s in dir_path.split( '/' )[ 1: ]:
            save_dir = os.path.join( save_dir, s )

        print( save_dir )
        continue

def get_images_from_game_dir( dir_path, required_count ):

    save_dir = dir_path.replace( raw_data_dir, processed_data_dir, 1 )

    if os.path.isdir( save_dir ):
        shutil.rmtree( save_dir )
    pathlib.Path( save_dir ).mkdir( parents = True )

    collected_count = 0
    for entry in os.scandir( dir_path ):

        name, ext = os.path.splitext( entry.path )

        assert ext == ".mp4"

        collected_count += get_images_from_video( entry.path, required_count - collected_count, save_dir )

        if collected_count == required_count:
            return

    if collected_count < required_count:
        print( f"Only {collected_count} images are collected from {dir_path}" )

def get_images_from_video( vid_path, max_count, save_dir ):

    vid_cap = cv.VideoCapture( vid_path )

    fps = vid_cap.get( cv.CAP_PROP_FPS )

    initial_skip_frames = initial_skip_sec * fps
    gap_frames = gap_sec * fps

    counter = 0
    while vid_cap.isOpened():

        vid_cap.set( cv.CAP_PROP_POS_FRAMES, gap_frames * counter + initial_skip_frames )

        ret, frame = vid_cap.read()

        if not ret:
            break

        frame = cv.cvtColor( frame, cv.COLOR_BGR2GRAY )

        frame = cv.resize( frame, ( 400, 225 ), interpolation = cv.INTER_AREA )

        #cv.imshow( "frame", frame )
        #cv.waitKey( 0 )

        cv.imwrite( os.path.join( save_dir, f"{vid_path.split( '/' )[ -1 ]}_{counter}.jpg" ), frame )

        counter += 1

        if counter >= max_count:
            break

    vid_cap.release()
    cv.destroyAllWindows()

    return counter


# Iterate all genre directories.
for genre_entry in os.scandir( raw_data_dir ):

    assert genre_entry.is_dir()

    for game_entry in os.scandir( genre_entry.path ):

        # Skip info.json etc.
        if not game_entry.is_dir():
            continue

        get_images_from_game_dir( game_entry.path, img_per_game )

