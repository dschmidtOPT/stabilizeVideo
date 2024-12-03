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

import serial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.backends.backend_agg as agg
from matplotlib.patches import FancyArrowPatch
import numpy as np


class State:
    hdg = 0
    pitch = 0 
    roll = 0

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def rotmat( hdg,roll, pitch):
    rotx = np.array([[1, 0, 0, 0],
                    [0, np.cos( roll ), -np.sin( roll ), 0],
                    [0, np.sin( roll ), np.cos(roll), 0],
                    [0, 0, 0, 1]])
    roty = np.array([[ np.cos( pitch ), 0, np.sin( pitch ), 0],
                     [0, 1, 0, 0],
                     [ -np.sin( pitch ), 0, np.cos(pitch), 0],
                     [0, 0, 0, 1]])
    rotz = np.array([[np.cos( hdg ), -np.sin( hdg ), 0, 0],
                    [np.sin( hdg ), np.cos( hdg ), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    temp = np.matmul(np.eye(4), rotz)
    temp = np.matmul( temp, rotx )
    final = np.matmul( temp, roty )
    return final
    

def setupRead( port = "/dev/ttyr02" ):
    return serial.Serial(port=port, baudrate=38400) 

def parseStream( ll ):
    try:
        hdg,roll,pitch, = [np.deg2rad(float(s)) for s in ll.split(b"*")[0].split(b",")[2:]]
        State.hdg = hdg
        State.pitch = pitch
        State.roll = roll
        valid = True
    except Exception as e:
        print(e)
        print(ll)
        hdg   = State.hdg
        pitch = State.pitch
        roll  = State.roll
        valid = False
    mat = rotmat( hdg, roll, pitch )
    return mat, valid

# Create a figure and an axes object
fig  = plt.figure( figsize = ( 6, 6 ) )
ax = fig.add_subplot( 111, projection = '3d' )
ax.set_title( "Buoy Orientation" )

# Set up the initial plot
scale = 0.65
leg = np.eye( 4 ) * scale
line1, = ax.plot([0,leg[0][0]], [0,leg[0][1]], [0,leg[0][2]], color = "r")
line2, = ax.plot([0,leg[1][0]], [0,leg[1][1]], [0,leg[1][2]], color = "b")
line3, = ax.plot([0,leg[2][0]], [0,leg[2][1]], [0,leg[2][2]], color = "g")

lineN, = ax.plot([0,leg[0][0]], [0,leg[0][1]], [0,leg[0][2]], color = "k", linestyle="--", linewidth=0.5)
N = ax.text(scale, 0, 0, "N", color = "k")
lineE, = ax.plot([0,leg[1][0]], [0,leg[1][1]], [0,leg[1][2]], color = "k", linestyle="--",linewidth=0.5)
W = ax.text(0, scale, 0, "W", color = "k")
lineU, = ax.plot([0,leg[2][0]], [0,leg[2][1]], [0,leg[2][2]], color = "k", linestyle="--",linewidth=0.5)
U = ax.text(0, 0, scale, "UP", color = "k")

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Animation loop
loop = True
ser = setupRead()
az =30
el =30
step = 2
dx,dy,dz=0,0,0
while loop:
    # Update the data
    ll = ser.readline()
    mat,valid = parseStream( ll )
    if not valid:
        continue

    line1.set_data_3d( [0,mat[0][0]], [0,mat[0][1]], [0, mat[0][2]] )
    F = ax.text( mat[0][0], mat[0][1], mat[0][2], "FRWD", color = "r")
    line2.set_data_3d( [0,mat[1][0]], [0,mat[1][1]], [0, mat[1][2]] )
    S = ax.text( mat[1][0], mat[1][1], mat[1][2], "LEFT", color = "b")
    line3.set_data_3d( [0,mat[2][0]], [0,mat[2][1]], [0, mat[2][2]] )
    Axis = ax.text( mat[2][0], mat[2][1], mat[2][2], "TOP", color = "g")
    

    # Redraw the figure
    plt.draw( )
    plt.pause( 0.025 )

ser.close()


plt.show()
