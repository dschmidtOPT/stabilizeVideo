""" Initialize computer vision library """

if __name__ == "__main__":
    import os
    import sys
    sys.addpath(os.getcwd())
    import cv2
    from . import main
    main.main()

