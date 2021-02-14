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


def plot_optical_flow(ax, Z, u, v, cx, cy, szx, szy, s=16):
    """
    param ax : object for plotting a figure
    param Z : scene to be observed
    param u : magnitude of optical flow along x axis
    param v : magnitude of optical flow along y axis
     
    """
    # Visualize optical flow map
    x, y = np.meshgrid(np.arange(szx), np.arange(szy))
    ax.imshow(Z, alpha=0.5, origin='upper')
    q = ax.quiver(x[::s,::s], y[::s,::s], u[::s,::s], -v[::s, ::s])
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
    """
    # Cast t and w to be column vectors
    if type(t) == list or t.shape != (3, 1):
        t = np.array(t)[:, np.newaxis]
    if type(w) == list or w.shape != (3, 1):
        w = np.array(w)[:, np.newaxis]
    
    u = np.random.rand(*Z.shape)
    v = np.random.rand(*Z.shape)

    # Fill data
    for y in range(szy):
        for x in range(szx):
            depth = Z[y, x]           # depth to the object from camera center
            X = depth / fx * (x - cx)  # X coordinate of the object w.r.t. the camera center
            Y = depth / fy * (y - cy)  # Y coordinate of the object w.r.t. the camera center
            
            # Compute optical flow
            T = np.array([[-1, 0, X], 
                        [0, -1, Y]])
            R = np.array([[X*Y, -(1+X**2),  Y],
                        [(1+Y**2), -X*Y, -X]])
            of = T @ t / (depth + eps) + R @ w
            u[y, x] = of[0]
            v[y, x] = of[1]
        
    return u, v

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
        
        u, v = create_optical_flow(Z, fx, fy, cx, cy, szx, szy, t, w)
        
        # Plot the obtained optical flow
        f = plt.figure(figsize=(13.5,9))
        plot_optical_flow(f.gca(), Z, u, v, cx, cy, szx, szy, s=16)
        f.savefig(os.path.join(current_dir, f'{data_name}.pdf'), bbox_inches='tight')
