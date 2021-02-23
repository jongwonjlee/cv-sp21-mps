import os
import numpy as np
import matplotlib.pyplot as plt


def get_wall_z_image(Z_val, fx, fy, cx, cy, szx, szy):
    Z = Z_val*np.ones((szy, szx), dtype=np.float32)
    return Z


def get_road_z_image(H_val, fx, fy, cx, cy, szx, szy):
    y = np.arange(szy).reshape(-1,1)*1.
    y = np.tile(y, (1, szx))
    Z = np.zeros((szy, szx), dtype=np.float32)
    Z[y > cy] = H_val*fy / (y[y>cy]-cy)
    Z[y <= cy] = np.NaN
    return Z


def plot_optical_flow(ax, Z, of_x, of_y, cx, cy, szx, szy, s=16):
    """
    param ax : object for plotting a figure
    param Z : scene to be observed
    param of_x : magnitude of optical flow along x-axis in camera coordinate
    param of_y : magnitude of optical flow along y-axis in camera coordinate
    """
    # Visualize optical flow map
    x, y = np.meshgrid(np.arange(szx), np.arange(szy))
    ax.imshow(Z, alpha=0.5, origin='upper')
    q = ax.quiver(x[::s,::s], y[::s,::s], of_x[::s,::s], -of_y[::s, ::s])
    # ax.quiverkey(q, X=0.5, Y=0.9, U=20, 
    #              label='Quiver key length = 20', labelpos='N')
    ax.axvline(cx)
    ax.axhline(cy)
    ax.set_xlim([0, szx])
    ax.set_ylim([szy, 0])
    ax.axis('equal')

def create_optical_flow(Z, fx, fy, cx, cy, szx, szy, t, w, eps=.001):
    """
    param Z: depth map
    param t: translational velocity (list or np array with (3,))
    param w: rotational velocity (list or np array with (3,))
    return of_x: magnitude of optical flow along x-axis in camera coordinate
    return of_y: magnitude of optical flow along y-axis in camera coordinate 
    """
    assert(fx == fy)
    f = fx

    # Cast t and w to be column vectors
    if type(t) == list or t.shape != (3, 1):
        t = np.array(t)[:, np.newaxis]
    if type(w) == list or w.shape != (3, 1):
        w = np.array(w)[:, np.newaxis]
    
    # Prepare empty optical flow data to be filled in
    of_x = np.empty((Z.shape))
    of_y = np.empty((Z.shape))

    # Fill data
    for v in range(szy):
        for u in range(szx):
            depth = Z[v, u]           # depth of the object w.r.t. camera center
            X = depth / f * (u - cx)  # X coordinate of the object w.r.t. the camera center in 3D
            Y = depth / f * (v - cy)  # Y coordinate of the object w.r.t. the camera center in 3D
            
            x = f * X / depth   # x coordinate of pixel w.r.t. the camera center in 2D
            y = f * Y / depth   # y coordinate of pixel w.r.t. the camera center in 2D

            # Construct matrices for translational and rotational velocity
            T = np.array([[-f, 0, x], 
                          [0, -f, y]])
            R = np.array([[x*y, -(x**2+f**2),  y],
                          [(y**2+f**2), -x*y, -x]])
            
            # Compute optical flow
            # eps has been added to avoid zero division
            of = T @ t / (depth + eps) + R @ w / f
            of_x[v, u] = of[0]
            of_y[v, u] = of[1]
        
    return of_x, of_y

if __name__ == "__main__":
    # Current code directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Focal length along X and Y axis. In class we assumed the same focal length 
    # for X and Y axis. but in general they could be different. We are denoting 
    # these by fx and fy, and assume that they are the same for the purpose of
    # this MP.
    fx = fy = 128.

    # Size of the image
    szy = 256
    szx = 384

    # Center of the image. We are going to assume that the principal point is at 
    # the center of the image.
    cx = 192
    cy = 128

    # Gets the image of the ground plane that is 3m below the camera.
    Z1 = get_road_z_image(3., fx, fy, cx, cy, szx, szy)
    # Gets the image of a wall 2m in front of the camera.
    Z2 = get_wall_z_image(2., fx, fy, cx, cy, szx, szy)

    # Plot Z1 and Z2
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))
    ax1.imshow(Z1)
    ax2.imshow(Z2)
    plt.savefig(os.path.join(current_dir, 'map.png'))

    # Create data to simulate
    dataset = {
        '01': {'Z': Z1, 't': [0,0,1], 'w':[0,0,0]},
        '02': {'Z': Z1, 't': [1,0,0], 'w':[0,0,0]},
        '03': {'Z': Z2, 't': [0,0,1], 'w':[0,0,0]},
        '04': {'Z': Z2, 't': [1,1,1], 'w':[0,0,0]},
        '05': {'Z': Z2, 't': [0,0,0], 'w':[0,1,0]},
        }
    
    for data_name, data in dataset.items():
        Z = data['Z']
        t = data['t']
        w = data['w']
        
        of_x, of_y = create_optical_flow(Z, fx, fy, cx, cy, szx, szy, t, w)
        
        # Plot the obtained optical flow
        f = plt.figure(figsize=(13.5,9))
        plot_optical_flow(f.gca(), Z, of_x, of_y, cx, cy, szx, szy, s=16)
        f.savefig(os.path.join(current_dir, f'{data_name}.pdf'), bbox_inches='tight')
