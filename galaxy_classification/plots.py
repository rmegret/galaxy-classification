from .cleaning import refine_center
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
import astropy as ap
from matplotlib.patches import Ellipse

def get_keys(item):
    if ('source_id' in item):
       key = f"S#{item.source_id}"
    else:
       key = f"#{int(item.name)}"
    if ('id_str' in item):
        id_str = item['id_str']
    else:
        id_str = ''
    return f"{key}: {id_str}"

def show_label(ax, im, item, label='galaxy'):
    h,w = im.shape[:2]

    ax.imshow(im)

    L = item[label]
    if (L):
       color = 'y'
    else:
       color = 'c'
    ax.text(20, 20, 'Label: {}'.format(L), color=color)
    ax.plot(w/2,h/2,'b+')
    keys = get_keys(item)
    #ax.text(10,240, keys, color=color, size=7)
    ax.set_title(keys, size=7)

def show_pred(ax, im, item, pred='galaxy_pred', label='galaxy'):
    h,w = im.shape[:2]

    ax.imshow(im)

    if (pred is not None):
       P = item[pred]
    else:
       P = None
    L = item[label]
    correct = np.around(P == L)
    if correct:
      color = 'g'
    else:
      color = 'r'
    if (pred):
        ax.text(20, 30, 'Pred: {:.2f}'.format(P), color=color)
    ax.text(20, 60, 'Label: {}'.format(L), color=color)
    ax.plot([130,230],[20,20],color=color)
    ax.plot([130+100*L],[25],'^',color='g')
    ax.plot([130+100*P],[20],'o',color=color)
    ax.text(130, 40, 'Err {:.2f}'.format(abs(L-P)), color=color)
    ax.plot(w/2,h/2,'b+')
    keys = get_keys(item)
    ax.text(10,240, keys, color=color, size=7)
    ax.set_title(keys, size=7)

def show_peak(ax, im, item):
  ax.imshow(im)

  img = np.array(im).sum(axis=2)
  #img = np.array(im)[:,:,0]
  result = refine_center(img)

  approx_center = result['approx_center']
  x_center, y_center = result['center']
  x_std, y_std = result['ellipse']['x_std'], result['ellipse']['y_std']
  theta = result['ellipse']['theta']
  a, b = result['ellipse']['a'], result['ellipse']['b']

  ax.imshow(im, origin='lower', cmap='gray')
  ax.add_patch(Ellipse((x_center, y_center), 2*a, 2*b, angle=np.degrees(theta), edgecolor='red', facecolor='none'))
  ax.plot(128,128, 'mx', label='Marked center')
  ax.plot(approx_center[0], approx_center[1], 'm+', label='Approx Center')
  ax.plot(x_center, y_center, 'rx', label='Fitted Center')
  keys = get_keys(item)
  ax.set_title(keys, size=7)

def show_crops(catalog, nrows=None, ncols=5, axsize=3, plotfun='pred', plotfunargs=None, imdir=None):
  """utility function to visualise galaxies and labels/predictions
  
  catalog: DataFrame with columns file_loc
  nrows: numbero of subplot rows
  ncols: number of subplot columns
  axsize: w or (w,h) size of individual axes
  plotfun: type of display, as a predefined ('pred', 'peak'...) or as
      a function to plot with prototype plotfun(ax, im, item)
      where im is the image, item is a DataFrame row extracted from catalog
  """

  N = catalog.shape[0]
  #print(N)
  if (nrows is None):
    nrows = (N+ncols-1)//ncols

  # Ensure axsize is a tuple (width, height)
  if isinstance(axsize, (int, float)):  # Scalar number
    axsize = (axsize, axsize)

  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(axsize[0]*ncols, axsize[1]*nrows), squeeze=False)
  axes = axes.ravel()
  for n in range(ncols*nrows):
    axes[n].axis('off')
    if (n>=N): continue;

    item = catalog.iloc[n]

    if (imdir is None):
       IMDIR = Path()
    else:
       IMDIR = Path(imdir)
    im = np.array(Image.open(IMDIR/item['file_loc']))

    #print(item)

    if plotfunargs is None:
       plotfunargs = {}

    if ((plotfun is None) or (plotfun == 'imshow')):
        axes[n].imshow(im)
        keys = get_keys(item)
        axes[n].text(10,240, keys, color='g', size=7)
        axes[n].set_title(keys, size=7)
    elif (plotfun == 'label'):
        show_label(axes[n], im, item, **plotfunargs)
    elif (plotfun == 'pred'):
        show_pred(axes[n], im, item, **plotfunargs)
    elif (plotfun == 'peak'):
        show_peak(axes[n], im, item, **plotfunargs)
    else:
        plotfun(axes[n], im, item, **plotfunargs)
  return fig, axes