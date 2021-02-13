import numpy as np
from generate_scene import get_ball
import matplotlib.pyplot as plt

# specular exponent
k_e = 50

def render(Z, N, A, S, 
           point_light_loc, point_light_strength, 
           directional_light_dirn, directional_light_strength,
           ambient_light, k_e):
  # To render the images you will need the camera parameters, you can assume
  # the following parameters. (cx, cy) denote the center of the image (point
  # where the optical axis intersects with the image, f is the focal length.
  # These parameters along with the depth image will be useful for you to
  # estimate the 3D points on the surface of the sphere for computing the
  # angles between the different directions.
  h, w = A.shape
  cx, cy = w / 2, h /2
  f = 128.

  # Input direction
  vi_p = np.empty((h, w, 3))
  vi_d = np.empty((h, w, 3))
  # Reflective direction
  si_p = np.empty((h, w, 3))
  si_d = np.empty((h, w, 3))
  # Output direction
  vr = np.empty((h, w, 3))

  # Fill data
  for v in range(h):
    for u in range(w):
      depth = Z[v, u]           # depth to the object from camera center
      X = depth / f * (u - cx)  # X coordinate of the object w.r.t. the camera center
      Y = depth / f * (v - cy)  # Y coordinate of the object w.r.t. the camera center
      
      # incident light direction
      vi_p[v, u, :] = np.array([X - point_light_loc[0][0], Y - point_light_loc[0][1], depth - point_light_loc[0][2]])
      vi_d[v, u, :] = np.array(directional_light_dirn[0])
      # specular reflection direction (see Section 2.2, Equation 2.89 in Szeliski)
      # FIXME: which one is correct?
      # si_p[v, u, :] = np.matmul(2 * N[v, u, :] * N[v, u, :].T - np.eye(3), vi_p[v, u, :])
      # si_d[v, u, :] = np.matmul(2 * N[v, u, :] * N[v, u, :].T - np.eye(3), vi_d[v, u, :])
      si_p[v, u, :] = vi_p[v, u, :] - 2 * np.dot(vi_p[v, u, :], N[v, u, :]) * N[v, u, :]
      si_d[v, u, :] = vi_d[v, u, :] - 2 * np.dot(vi_d[v, u, :], N[v, u, :]) * N[v, u, :]
      # viewing direction
      vr[v, u, :] = np.array([X, Y, depth])
  
  # Normalize above
  vi_p /= np.linalg.norm(vi_p, axis=2)[:, :, np.newaxis]
  vi_d /= np.linalg.norm(vi_d, axis=2)[:, :, np.newaxis]
  si_p /= np.linalg.norm(si_p, axis=2)[:, :, np.newaxis]
  si_d /= np.linalg.norm(si_d, axis=2)[:, :, np.newaxis]
  vr /= np.linalg.norm(vr, axis=2)[:, :, np.newaxis]

  # Ambient Term
  Ia = A * ambient_light
  
  # Diffuse Term
  point_light_strength = point_light_strength[0]
  directional_light_strength = directional_light_strength[0]
  
  Id_p = A * point_light_strength * np.clip(np.einsum('ijk,ijk->ij', vi_p, N), 0, None)
  Id_d = A * directional_light_strength * np.clip(np.einsum('ijk,ijk->ij', vi_d, N), 0, None)
  
  # Specular Term
  # FIXME: wether clip or not?
  Is_p = S * point_light_strength * pow(np.clip(np.einsum('ijk,ijk->ij', vr, si_p), 0, None), k_e)
  Is_d = S * directional_light_strength * pow(np.clip(np.einsum('ijk,ijk->ij', vr, si_d), 0, None), k_e)

  # Sum up terms above
  I = Ia + Id_p + Id_d + Is_p + Is_d

  # Prepare for output
  I = np.minimum(I, 1) * 255
  I = I.astype(np.uint8)
  I = np.repeat(I[:,:,np.newaxis], 3, axis=2)
  return I

def main():
  for specular in [True, False]:
    # get_ball function returns:
    # - Z (depth image: distance to scene point from camera center, along the
    # Z-axis)
    # - N is the per pixel surface normals (N[:,:,0] component along X-axis
    # (pointing right), N[:,:,1] component along X-axis (pointing down),
    # N[:,:,0] component along X-axis (pointing into the scene)),
    # - A is the per pixel ambient and diffuse reflection coefficient per pixel,
    # - S is the per pixel specular reflection coefficient.
    Z, N, A, S = get_ball(specular=specular)

    # Strength of the ambient light.
    ambient_light = 0.5
    
    # For the following code, you can assume that the point sources are located
    # at point_light_loc and have a strength of point_light_strength. For the
    # directional light sources, you can assume that the light is coming _from_
    # the direction indicated by directional_light_dirn, and with strength
    # directional_light_strength. The coordinate frame is centered at the
    # camera, X axis points to the right, Y-axis point down, and Z-axis points
    # into the scene.
    
    # Case I: No directional light, only point light source that moves around
    # the object. 
    point_light_strength = [1.5]
    directional_light_dirn = [[1, 0, 0]]
    directional_light_strength = [0.0]
    
    fig, axes = plt.subplots(4, 4, figsize=(15,10))
    axes = axes.ravel()[::-1].tolist()
    for theta in np.linspace(0, np.pi*2, 16): 
      point_light_loc = [[10*np.cos(theta), 10*np.sin(theta), -3]]
      I = render(Z, N, A, S, point_light_loc, point_light_strength, 
                 directional_light_dirn, directional_light_strength,
                 ambient_light, k_e)
      ax = axes.pop()
      ax.imshow(I)
      ax.set_axis_off()
    plt.savefig(f'specular{specular:d}_move_point.png', bbox_inches='tight')
    plt.close()

    # Case II: No point source, just a directional light source that moves
    # around the object.
    point_light_loc = [[0, -10, 2]]
    point_light_strength = [0.0]
    directional_light_strength = [2.5]
    
    fig, axes = plt.subplots(4, 4, figsize=(15,10))
    axes = axes.ravel()[::-1].tolist()
    for theta in np.linspace(0, np.pi*2, 16): 
      directional_light_dirn = [np.array([np.cos(theta), np.sin(theta), .1])]
      directional_light_dirn[0] = \
        directional_light_dirn[0] / np.linalg.norm(directional_light_dirn[0])
      I = render(Z, N, A, S, point_light_loc, point_light_strength, 
                 directional_light_dirn, directional_light_strength,
                 ambient_light, k_e) 
      ax = axes.pop()
      ax.imshow(I)
      ax.set_axis_off()
    plt.savefig(f'specular{specular:d}_move_direction.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
  main()
