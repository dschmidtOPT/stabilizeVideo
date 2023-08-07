""" Code snippets copied from spmallick learnopencv repo.  Code herein does not represent original work or production intent.
    Please reference https://github.com/spmallick/learnopencv/blob/master/VideoStabilization/video_stabilization.py for more information """

import os
import cv2
import argparse
import numpy as np

### Constants ###





def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size)/window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed

def smooth(trajectory, SMOOTHING_RADIUS):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

class Logger:
    '''Writes to output log depending upon verbosity level'''
    def __init__(self, args):
        fname = os.path.join(args.inPath , "VideoStabilizerLog.txt")
        import datetime
        now = datetime.datetime.now()
        with open(fname, "w") as f:
            f.write("videoStabilizer main() ran {now.year}/{now.month}/{now.day} - {now.hour}:{now.minute}:{now.second} \n")
            f.write(f"Running OpenCV version {cv2.__version__}\n")

class Video:
    '''Holds video capture data structure and video metadata
    Available colorspaces for playback and edit:
        cv2.COLOR_BGR2YCrCb
        cv2,COLOR_BGR2HSV
        cv2.COLOR_BGR2LAB
        cv2.COLOR_BGR2GRAY
    '''
    # The larger the smoothing radius, the more stable the video, but less reactive to sudden panning
    SMOOTHING_RADIUS=250
    thermal = False
    n_frames = -1
    fps = -1
    w = -1
    h = -1 
    fullpath = ""
    def __init__( self, args ):
        vname = args.inPath
        if not os.path.exists(os.path.abspath(vname)):
            print("\tWARNING - provided video file path does not exist.  Attempting file open")
        self.cvopen( vname )
        self.stabilize()
        # Release video
        self.cap.release()

    def cvplayback( self, vname ):
        while self.cap.isOpened():
             ret, frame = self.cap.read()
             # if frame is read correctly ret is True
             if not ret:
                 print("Can't receive frame (stream end?). Exiting ...")
                 break
             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             cv2.imshow('frame', gray)
             if cv2.waitKey(1) == ord('q'):
                 break
        self.cap.release()
         
    def cvopen( self, vname ):
        cap = cv2.VideoCapture( vname )
        self.n_frames = int( cap.get( cv2.CAP_PROP_FRAME_COUNT ) )
        self.fps = cap.get( cv2.CAP_PROP_FPS )
        self.w = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ))
        self.h = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outname = os.path.join( args.cwd,
                                os.path.basename(vname).split(".m")[0] + "_stabilized.mp4")                                          
        print(f"{outname}:\nH={self.h}pixels;W={self.w}pixels,fps={self.fps}")
        if (self.w == 640 and self.h == 480):
            print("Thermal Video Detected - adjusting stabilization parameters")
            self.SMOOTHING_RADIUS=50
            self.thermal = True
        self.out = cv2.VideoWriter( outname, fourcc, self.fps, ( 2*self.w, self.h ), True )
        self.fullpath = os.path.abspath( outname )
        self.cap = cap

    def stabilize( self ):
        cap = self.cap
        _, prev = cap.read()
        prev_gray = cv2.cvtColor( prev, cv2.COLOR_BGR2GRAY )
        transforms = np.zeros( ( self.n_frames-1, 3 ), np.float32 )
        if self.n_frames > 5000:
            everyFrame = 0
        else:
            everyFrame = 1
        
        for i in range(self.n_frames-2):
            #Detect feature points in previous frame
            if self.thermal:
                prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                                  maxCorners=200,
                                                  qualityLevel = 0.001,
                                                  minDistance=20,
                                                  blockSize=3)
            else:
                prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                                  maxCorners=250,
                                                  qualityLevel = 0.005,
                                                  minDistance=40,
                                                  blockSize=3)
            success, curr = cap.read()
            if not success:
                break

            # Convert to grayscale
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow (i.e. track feature points)
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

            # Sanity check
            assert prev_pts.shape == curr_pts.shape

            # Filter only valid points
            idx = np.where(status==1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            #Find transformation matrix
            #m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less
            #dx = m[0,2]
            #dy = m[1,2]

            m, mask = cv2.estimateAffine2D(prev_pts, curr_pts) 
            #import pdb;pdb.set_trace()
            # Extract translation

            # Store transformation:
            if isinstance(m,np.ndarray):
                dx = m[0,2]
                dy = m[1,2]
                # Extract rotation angle
                da = np.arctan2(m[1,0], m[0,0])                
            else:  # Something unexpected caused a failure, m is "None"
                dx = dy = da = 0
            # Store transformation
            transforms[i] = [dx,dy,da]
                

            # Move to next frame
            prev_gray = curr_gray

            if i % ( int((self.n_frames**0.5) / 3) ) == 0:
                print(f"Frame: {i}/{self.n_frames} - Tracked points: {len(prev_pts)}")

        trajectory = np.cumsum(transforms, axis=0)

        smoothed_trajectory = smooth(trajectory, self.SMOOTHING_RADIUS)

        difference = smoothed_trajectory - trajectory

        transforms_smooth = transforms + difference

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for i in range(self.n_frames-2):
            success, frame = cap.read()
            if not success:
                break
            # Extract transformations from the new transformation array
            dx = transforms_smooth[i,0]
            dy = transforms_smooth[i,1]
            da = transforms_smooth[i,2]

            # Reconstruct transformation matrix accordingly to new values
            m = np.zeros((2,3), np.float32)
            m[0,0] = np.cos(da)
            m[0,1] = -np.sin(da)
            m[1,0] = np.sin(da)
            m[1,1] = np.cos(da)
            m[0,2] = dx
            m[1,2] = dy

            # Apply affine wrapping to the given frame
            frame_stabilized = cv2.warpAffine(frame, m, ( self.w, self.h))

            # Fix border artifacts
            frame_stabilized = fixBorder(frame_stabilized) 

            # Write the frame to the file
            frame_out = cv2.hconcat([frame, frame_stabilized])

            # If the image is too big, resize it.
            #if(frame_out.shape[1] > 1920):
            #    frame_out = cv2.resize(frame_out, (int(frame_out.shape[1]/2), int(frame_out.shape[0]/2)));

            cv2.imshow("Before and After", frame_out)
            cv2.waitKey(int(1/self.fps * 1000))
            self.out.write( frame_out )
        cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        # Close windows
        

def main( args ):
    if args.verbose:
        log = Logger(args)
    if not args.inPath:
        matches = [m for m in os.listdir(args.inPath) if m.find(".mp4") != -1]
        if matches:
            args.inPath = matches[0]
        else:
            print("No candidate video files found in working directory - provide input video with -i <path to video file>")
            import sys; sys.exit()

    args.cwd = os.path.abspath(
        os.path.dirname(args.inPath))

    vid = Video( args )
    print( f"Video processed sucessfully.  Output to \n {vid.fullpath}" )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--inPath", default = None,
                        help="inputPath to file, no entry defaults to first path matching $PWD/*.mpeg")
    parser.add_argument("-o","--outPath", default = None,
                        help="outputPath to write stabilized video frames, no entry defaults to $PWD/stabilizedVideo.mpeg")
    parser.add_argument("-v","--verbose", action = "store_true", default = False, help="Enable additional logging")
    args = parser.parse_args()
    main( args )
