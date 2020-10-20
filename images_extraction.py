import os
import shutil
import pathlib
import cv2 as cv


def get_images_from_game_dir( dir_path, save_dir, required_count, initial_skip_sec, gap_sec ):

    if os.path.isdir( save_dir ):
        shutil.rmtree( save_dir )
    pathlib.Path( save_dir ).mkdir( parents = True )

    collected_count = 0
    for entry in os.scandir( dir_path ):

        name, ext = os.path.splitext( entry.path )

        # Skip vidlist.txt and so on.
        if ext != ".mp4":
            continue

        collected_count += get_images_from_video( entry.path, save_dir, required_count - collected_count, initial_skip_sec, gap_sec )

        if collected_count == required_count:
            return

    if collected_count < required_count:
        print( f"Only {collected_count} images are collected from {dir_path}" )

def get_images_from_video( vid_path, save_dir, max_count, initial_skip_sec = 60, gap_sec = 20 ):

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

        cv.imwrite( os.path.join( save_dir, f"{vid_path.split( '/' )[ -1 ]}-{counter}.jpg" ), frame )

        counter += 1

        if counter >= max_count:
            break

    vid_cap.release()
    cv.destroyAllWindows()

    return counter

