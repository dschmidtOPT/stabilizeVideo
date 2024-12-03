#!/usr/bin/python3
#----------------------------------------------------------------------------
#    Ref: https://faculty.sites.iastate.edu/jia/files/inline-files/homogeneous-transform.pdf
#        Rotation about the coordinate axes
#        Coordinates are as follows: 
#           +Z downward into ocean, +X forward brng=0deg, +y starboard brng = 90deg 
#                    |    
#             /\     |+z ---->
#            [  ]    v    +x
#        ~~~~~~~~~~~~~~~~~~~~~~~
#    Purpose: 
#        Create buoy orientation visualization from SCX-21 Sat Compass
#    Requirements:
#        Port configured with $PFEC.GPatt data with format <heading>,<roll>,<pitch>
#
#---------------------------------------------------------------------------- 

import pygame
import serial
import numpy as np
from pygame.locals import *
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import numpy as np
import pylab

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.plot_utils import Frame
from pytransform3d import rotations as pr

WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
TEXTCOLOR = (  0,   0,   0)
(width, height) = (300, 300)
step = 2 #Size in pixels for each step
halfX = int(width/2) 
halfY = int(height/2)
radius = 25
port = "/dev/ttyr02"

#legs = np.hstack(np.eye(3), 
#       np.array([ [0],[0],[0] ]),


class IMU:
    vx = 0
    vy = 0
    vz = 0
    wx = 0
    wy = 0
    wz = 0


class myArrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def setupRead( port ):
    return serial.Serial(port=port, baudrate=38400) 

def setupPlot( ):
    global fig, ax
    fig = plt.figure(figsize=(6,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    #bframe = Frame( np.eye(4), label = "Buoy")
    #wframe = Frame( np.eye(4), label = "World")
    #bframe.add_frame( ax )
    plt.ion()
    return fig,ax

def buildRotMat( ax, hdg=0, roll=0, pitch=0):
    '''rotx = np.array([1, 0, 0, 0],
                    [0, np.cos( roll ), −1* np.sin( roll ), 0],
                    [0 sin θx cos θx 0],
                    [0, 0, 0, 1])
    roty = np.array([np.cos( pitch ), 0, np.sin( pitch ), 0]
                    [0, 1, 0, 0],
                    [−1* np.sin( pitch ), 0, np.cos(pitch), 0],
                    [0, 0, 0, 1])
    rotz = np.array([np.cos( hdg ), −1*np.sin( hdg ), 0, 0],
                    [np.sin( hdg ), np.cos( hdg ), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1])
    np.linalg()
    '''
    legs = np.array([1,1,1])
    pr.plot_basis(ax,
    pr.active_matrix_from_extrinsic_euler_xyz([hdg, pitch, roll]), 1.5 * legs,
    lw=5)
    #pr.plot_axis_angle(ax, [0, 0, 1, gamma], 0.5 * p)
    ax.scatter3D(0,0,0)
    return ax

    

def buildLegs( line, ax):
    try:
        hdg,roll,pitch, = [np.deg2rad(float(s)) for s in line.split(b"*")[0].split(b",")[2:]]
        
    except Exception as e:
        print(e)
        print(line)
        import pdb; pdb.set_trace()
        return
    #print( "hdg:",hdg, "pitch:",pitch,"roll",roll)
    ax = buildRotMat( ax, hdg, roll, pitch )
    return ax
    

    

        

def main():
    loop = True
    ser = setupRead( port )

    # Initialize plot #
    #fig,ax = setupPlot()

    # Fill background
    fig = pylab.figure(figsize=[4, 4], # Inches
                   dpi=100)        # 100 dots per inch, so the resulting buffer is 400x400 pixels
    fig = plt.figure(figsize=(6,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(0, 0, 0)

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()

    pygame.init()

    window = pygame.display.set_mode((width, height), DOUBLEBUF)
    screen = pygame.display.get_surface()


    size = canvas.get_width_height()
    surf = pygame.image.fromstring(raw_data, size, "RGB")
    screen.blit(surf, (0,0))
    pygame.display.flip()

    crashed = False


#    background = pygame.Surface(screen.get_size())
#    background = background.convert()
#    background.fill(( 0, 0, 0))
#    clearback = pygame.Surface(screen.get_size())
#    clearback = background.convert()
#    clearback.fill((250, 250, 250))

    # Blit everything to the screen
#    screen.blit(background, (0, 0))
#    pygame.display.flip()


    # Event loop
    #clock = pygame.time.Clock()
    posX,posY = ( halfX, halfY )
    dx = 0
    dy = 0
    while loop:
        line = ser.readline()
        #ax = buildLegs( line, ax )
        ax.view_init(30, 30)
        plt.draw()
        plt.pause(.001)
     
    plt.show()
    if False:
#        plt.clf()
        
    
    #    fig.canvas.flush_events()
        #plt.show()

        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            if keys[ pygame.K_RIGHT ]:
                dx += step
            if keys[ pygame.K_LEFT ]:
                dx -= step
            if keys[ pygame.K_DOWN ]:
                dy += step
            if keys[ pygame.K_UP ]:
                dy -= step
            if event.type == QUIT:
                loop = False
                ser.close()
                return
        ### Process key changes ###
        posX += dx
        posY += dy
        if posX > (width + 2*radius):
            posX = -2*radius
        if posX < -2*radius:
            posX = width + (2*radius)
        if posY > height + (2*radius):
            posY = -2*radius
        if posY < -2*radius:
            posY = width + (2*radius)
        # Clear background
        #background.fill((0, 0, 0))
        # Draw axes
        #pygame.draw.line( background, RED, ( 0,  halfY), ( width, halfY ) )
        #pygame.draw.line( background, RED, ( halfX, 0 ), ( halfX, height ) )
        #pygame.draw.circle( background, BLUE, ( posX, posY ), radius )
        screen.blit( surf, ( 0, 0 ) )

        #clock.tick( 10 )
        pygame.display.flip( )
        #pygame.event.pump()


if __name__ == '__main__': 
    main()
