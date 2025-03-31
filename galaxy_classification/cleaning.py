from photutils.detection import find_peaks
from astropy.modeling import models, fitting
import numpy as np
from matplotlib.patches import Ellipse
from PIL import Image

def refine_center(img, refine=False):

  # Assuming `im` is your intensity image (2D numpy array)
  # Normalize the image to [0, 1]
  normalized_im = img / np.max(img[128-20:128+20, 128-20:128+20])

  # Threshold at 0.5 of the max value
  threshold = 0.5
  peaks = find_peaks(normalized_im, threshold=threshold, )
  if (peaks is None) or (len(peaks) == 0):
    return None
    #raise ValueError("No bright blob found in the image.")
  sqdist = np.sqrt((peaks['x_peak'] - 128)**2 + (peaks['y_peak'] - 128)**2)
  idx = np.argmin( sqdist )
  approx_center = (peaks['x_peak'][idx], peaks['y_peak'][idx])

  if (refine):
    # Define a small region around the approximate center for fitting
    x_min, x_max = int(approx_center[0] - 20), int(approx_center[0] + 20)
    y_min, y_max = int(approx_center[1] - 20), int(approx_center[1] + 20)
    sub_image = normalized_im[y_min:y_max, x_min:x_max]

    # Create a grid for the sub-image
    y, x = np.mgrid[y_min:y_max, x_min:x_max]

    # Fit a 2D Gaussian model
    gaussian_init = models.Gaussian2D(amplitude=np.max(sub_image),
                      x_mean=approx_center[0],
                      y_mean=approx_center[1],
                      x_stddev=5,
                      y_stddev=5,
                      theta=0)
    fitter = fitting.LevMarLSQFitter()

    try:
      gaussian_fit = fitter(gaussian_init, x, y, sub_image)
    except Exception as e:
      print(e)
      return None

    # Extract the fitted parameters
    x_center, y_center = gaussian_fit.x_mean.value, gaussian_fit.y_mean.value
    x_std, y_std = gaussian_fit.x_stddev.value, gaussian_fit.y_stddev.value
    theta = gaussian_fit.theta.value

    # Define the ellipse parameters
    a = 2 * x_std  # Semi-major axis
    b = 2 * y_std  # Semi-minor axis

    center = (x_center, y_center)
    return dict(approx_center=approx_center, center=center, ellipse=dict(x_std=x_std, y_std=y_std, theta=theta, a=a, b=b))
  else:
    center = approx_center
    x_std, y_std = 5,5
    theta=0
    a = 2 * x_std
    b = 2 * y_std
    return dict(approx_center=approx_center, center=center, ellipse=dict(x_std=x_std, y_std=y_std, theta=theta, a=a, b=b))

  