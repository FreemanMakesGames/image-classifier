import os
import cv2 as cv

raw_data_dir = "raw-data"
processed_data_dir = "processed-data"

gap_sec = 20  # Delay in seconds between screenshots.

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


        # Get images from video.

        vid_cap = cv.VideoCapture( os.path.join( raw_data_dir ), "first-person-shooter/csgo/csgo.mp4" )

        gap_frames = gap_sec * vid_cap.get( cv.CAP_PROP_FPS )

        counter = 0
        while vid_cap.isOpened():

            vid_cap.set( cv.CAP_PROP_POS_FRAMES, gap_frames * counter )

            ret, frame = vid_cap.read()

            if not ret:
                break

            counter += 1

            frame = cv.cvtColor( frame, cv.COLOR_BGR2GRAY )

            frame = cv.resize( frame, ( 400, 225 ), interpolation = cv.INTER_AREA )

            cv.imshow( "frame", frame )
            cv.waitKey( 0 )

            #cv.imwrite( "processed-data/{0}.jpg".format( counter ), frame )

        vid_cap.release()
        cv.destroyAllWindows()


process_images_from_dir( raw_data_dir )

