import cv2 as cv

gap_sec = 20  # Delay in seconds between screenshots.

vid_cap = cv.VideoCapture( "raw-data/first-person-shooter/csgo/csgo.mp4" )

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

