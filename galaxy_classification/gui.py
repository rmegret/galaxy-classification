import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from PIL import Image
from pathlib import Path

class GalaxySelector:
    def __init__(self, catalog, imdir=None):
        """
        Initialize the GalaxySelector GUI.

        catalog: DataFrame with columns 'file_loc' and other metadata.
        imdir: Directory containing the images.
        """

        self.wasinteractive = plt.isinteractive()
        plt.ioff()

        self.catalog = catalog
        self.imdir = Path(imdir) if imdir else Path()
        self.current_index = 0
        self.selected_center = None

        # Widgets
        self.slider = widgets.IntSlider(value=0, min=0, max=len(catalog)-1)
        self.text = widgets.IntText(value=0)
        self.prev_button = widgets.Button(description="<", layout=widgets.Layout(width="20px"))
        self.next_button = widgets.Button(description=">", layout=widgets.Layout(width="20px"))
        self.confirm_button = widgets.Button(description="Confirm")
        self.cancel_button = widgets.Button(description="Cancel")
        self.close_button = widgets.Button(description="Close")
        self.output = widgets.Output()
        self.outputfig = widgets.Output()

        # Event handlers
        self.slider.observe(self.on_slider_change, names='value')
        self.text.observe(self.on_text_change, names='value')
        self.prev_button.on_click(self.on_prev)
        self.next_button.on_click(self.on_next)
        self.confirm_button.on_click(self.on_confirm)
        self.cancel_button.on_click(self.on_cancel)
        self.close_button.on_click(self.on_close)

        # Layout
        action_controls = widgets.HBox([self.confirm_button, self.cancel_button, self.close_button])
        navigation_controls = widgets.HBox([self.prev_button, self.text, self.next_button, self.slider,])
        self.controls = widgets.VBox([navigation_controls, action_controls])
        display(self.controls, self.outputfig, self.output)

        # Init plot structure
        with self.outputfig:
            fig, ax = plt.subplots()
            self.fig = fig
            self.ax = ax
            ax.imshow([[0]])
            fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click(event))
            fig.canvas.draw() #plt.show()
            display(fig)

        plt.pause(0.1)

        # Initial plot
        with self.outputfig:
            self.plot_image()

    def on_slider_change(self, change):
        with self.output:
            clear_output(wait=True)
            self.current_index = change['new']
            self.text.value = self.current_index
            self.plot_image()

    def on_text_change(self, change):
        with self.output:
            clear_output(wait=True)
            self.current_index = change['new']
            self.slider.value = self.current_index
            self.plot_image()

    def on_prev(self, button):
        if self.current_index > 0:
            self.current_index -= 1
            self.slider.value = self.current_index

    def on_next(self, button):
        if self.current_index < len(self.catalog) - 1:
            self.current_index += 1
            self.slider.value = self.current_index

    def on_confirm(self, button):
        with self.output:
            if self.selected_center:
                print(f"Confirmed new center for image {self.current_index}: {self.selected_center}")
            else:
                print(f"No new center selected for image {self.current_index}.")

    def on_cancel(self, button):
        with self.output:
            print(f"Canceled changes for image {self.current_index}.")
        self.selected_center = None
        self.plot_image()

    def on_close(self, button):
        with self.output:
            print(f"Closing GUI.")
        self.selected_center = None
        self.controls.close_all()
        if (self.wasinteractive):
            plt.ion()

    # Ensure the click event is connected properly
    def on_click(self, event):
        with self.output:
            try:
                print(f"on_click")
                item = self.catalog.iloc[self.current_index]
                if event.xdata and event.ydata:
                    self.selected_center = (event.xdata, event.ydata)
                    print(f"Selected new center: {self.selected_center}")
                    self.catalog.loc[item.name, 'center_x'] = int(event.xdata)
                    self.catalog.loc[item.name, 'center_y'] = int(event.ydata)
                    self.plot_image()
            except Exception as e:
                print(e)

    def plot_image(self):
        with self.output:
            #clear_output(wait=True)
            print('plot_image')
            try:
                item = self.catalog.iloc[self.current_index]
                im = np.array(Image.open(self.imdir / item['file_loc']))

                self.selected_center = item[['center_x','center_y']].to_list()
                print(f'selected_center {self.selected_center} for idx={self.current_index}, loc={item.name}')

                #fig, ax = plt.subplots()
                fig = self.fig
                ax = self.ax
                with self.outputfig:
                    ax.clear()
                    ax.imshow(im)
                    ax.set_title(f"#{item.source_id} {item.id_str}")

                    h,w = im.shape[:2]

                    # Plot selected center if available
                    if self.selected_center:
                        ax.plot(w/2,h/2, 'r+', label='Image Center')
                        ax.plot(self.selected_center[0], self.selected_center[1], 'gx', label='Peak Center')
                        ax.legend()

                #fig.canvas.mpl_connect('button_press_event', on_click)
                #plt.show()
            except Exception as e:
                print(e)

import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from pathlib import Path

class GalaxySelectorPlotly:
    def __init__(self, catalog, imdir=None):
        """
        Initialize the GalaxySelectorPlotly GUI.

        catalog: DataFrame with columns 'file_loc' and other metadata.
        imdir: Directory containing the images.
        """
        self.catalog = catalog
        self.imdir = Path(imdir) if imdir else Path()
        self.current_index = 0
        self.selected_center = None
        self.center_modified = False

        # Widgets
        self.slider = widgets.IntSlider(value=0, min=0, max=len(catalog)-1)
        self.text = widgets.IntText(value=0)
        self.prev_button = widgets.Button(description="<", layout=widgets.Layout(width="20px"))
        self.next_button = widgets.Button(description=">", layout=widgets.Layout(width="20px"))
        self.confirm_button = widgets.Button(description="Confirm")
        self.cancel_button = widgets.Button(description="Cancel")
        self.delete_button = widgets.Button(description="Delete")
        self.output = widgets.Output()
        self.outputfig = widgets.Output()

        # Event handlers
        self.slider.observe(self.on_slider_change, names='value')
        self.text.observe(self.on_text_change, names='value')
        self.prev_button.on_click(self.on_prev)
        self.next_button.on_click(self.on_next)
        self.confirm_button.on_click(self.on_confirm)
        self.cancel_button.on_click(self.on_cancel)
        self.delete_button.on_click(self.on_delete)

        # Layout
        navigation_controls = widgets.HBox([self.prev_button, self.text, self.next_button, self.slider])
        action_controls = widgets.HBox([self.confirm_button, self.cancel_button, self.delete_button])
        self.controls = widgets.VBox([navigation_controls, action_controls])
        display(self.controls, self.outputfig, self.output)

        with self.outputfig:
            # Create the Plotly figure
            fig = go.FigureWidget(data=[])

            im = np.array(Image.open(self.imdir / catalog.iloc[0]['file_loc']))
            t = go.Image(z=im, name='Image')
            fig.add_trace(t)
            self.image = [d for d in fig.data if d.name == 'Image'][0]

            height, width = im.shape[:2]

            # Add a callback for click events
            fig.update_layout(
                #title=f"Image {self.current_index}",
                clickmode='event+select',
                margin=dict(l=0, r=0, t=0, b=0),
                modebar=dict(orientation='v'),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    scaleanchor="y",
                    scaleratio=1,
                    range=[0, width],  # Constrain zoom to image width
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    scaleanchor="x",
                    scaleratio=1,
                    range=[0, height],  # Constrain zoom to image height
                ),
            )

            # Enable scroll zoom explicitly
            fig.show(config={"scrollZoom": True})  # Ensure scrollZoom is enabled
            display(fig)

        self.fig = fig

        # Initial plot
        self.image_changed()
        self.plot_image()

    def image_changed(self):
        self.selected_center = None
        item = self.catalog.iloc[self.current_index]
        self.selected_center = item[['center_x','center_y']].to_list()
        print(f'selected_center {self.selected_center} for idx={self.current_index}, loc={item.name}')
        self.center_modified = False

    def on_slider_change(self, change):
        self.current_index = change['new']
        self.text.value = self.current_index
        self.image_changed()
        self.plot_image()

    def on_text_change(self, change):
        self.current_index = change['new']
        self.slider.value = self.current_index
        self.image_changed()
        self.plot_image()

    def on_prev(self, button):
        if self.current_index > 0:
            self.current_index -= 1
            self.slider.value = self.current_index

    def on_next(self, button):
        if self.current_index < len(self.catalog) - 1:
            self.current_index += 1
            self.slider.value = self.current_index

    def on_confirm(self, button):
        with self.output:
            clear_output(wait=True)
            if self.selected_center:
                item = self.catalog.iloc[self.current_index]
                print(f"Confirmed new center for image {self.current_index}: {self.selected_center}")
                self.catalog.loc[item.name, 'center_x'] = self.selected_center[0]
                self.catalog.loc[item.name, 'center_y'] = self.selected_center[1]
                self.center_modified = False
                self.catalog.loc[item.name, 'manual_center'] = True
            else:
                self.catalog.loc[item.name, 'center_x'] = np.nan
                self.catalog.loc[item.name, 'center_y'] = np.nan
                print(f"No new center selected for image {self.current_index}.")
        self.plot_image()

    def on_cancel(self, button):
        with self.output:
            clear_output(wait=True)
            print(f"Canceled changes for image {self.current_index}.")
            self.center_modified = False
            #self.selected_center = None
            self.selected_center = item[['center_x','center_y']].to_list() # Revert to saved center
            self.plot_image()

    def on_delete(self, button):
        with self.output:
            clear_output(wait=True)
            print(f"Delete peak for image {self.current_index}.")
            item = self.catalog.iloc[self.current_index]
            self.catalog.loc[item.name, 'center_x'] = np.nan
            self.catalog.loc[item.name, 'center_y'] = np.nan
            self.selected_center = item[['center_x','center_y']].to_list() # Revert to saved center
            self.center_modified = False
            self.catalog.loc[item.name, 'manual_center'] = False
            self.plot_image()

    def on_click(self, trace, points, selector):
        with self.output:
            print('on_click')
            if points.point_inds:
                x, y = points.xs[0], points.ys[0]
                self.selected_center = (x, y)
                with self.output:
                    clear_output(wait=True)
                    print(f"Selected new center: {self.selected_center}")
                self.center_modified = True
                self.plot_image()

    def plot_image(self):
        with self.output:
            clear_output(wait=True)
            try:
                item = self.catalog.iloc[self.current_index]
                with self.output:
                    print(f"#{item.source_id}: {item.id_str}")
                im = np.array(Image.open(self.imdir / item['file_loc']))

                # Create the plotly figure
                #fig = go.FigureWidget(data=[go.Image(z=im)])

                fig = self.fig

                #fig.data = []  # Clear all traces
                self.image.z=im
                #fig.add_trace(t)

                fig.data = [trace for trace in fig.data if trace.name != 'Peak']

                # Add the selected center if available
                if self.selected_center:
                    color = 'red' if self.center_modified else 'green'
                    fig.add_trace(go.Scatter(
                        x=[self.selected_center[0]],
                        y=[self.selected_center[1]],
                        mode='markers',
                        marker=dict(color=color, size=5, ),
                        name='Peak'
                    ))

                # Add a callback for click events
                fig.update_layout(
                    #title=f"Image {self.current_index}",
                    clickmode='event+select',
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                dataimage = data = [d for d in fig.data if isinstance(d,go.Image)][0]
                dataimage.on_click(lambda trace, points, selector: self.on_click(trace, points, selector))

                fig.update_traces()

                # Display the figure
                #fig.show()
                #display(fig)
            except Exception as e:
                with self.output:
                    print(f"Error plotting image: {e}")

import plotly.express as px
import plotly.graph_objects as go

def plot_embeddings_plotly(embeddings, labels=None, hover_data=None, title="Embeddings Visualization"):
    """
    Visualize embeddings in 2D or 3D using Plotly.

    embeddings: numpy array of shape (n_samples, 2) or (n_samples, 3)
    labels: Optional, array-like of shape (n_samples,) for coloring points
    hover_data: Optional, array-like of shape (n_samples,) for hover information
    title: Title of the plot
    """
    try:
        # Determine if the embeddings are 2D or 3D
        dim = embeddings.shape[1]
        if dim not in [2, 3]:
            raise ValueError("Embeddings must have 2 or 3 dimensions.")

        # Create a DataFrame for easier handling
        import pandas as pd
        data = pd.DataFrame(embeddings, columns=['x', 'y'] if dim == 2 else ['x', 'y', 'z'])
        if labels is not None:
            data['label'] = labels
        if hover_data is not None:
            data['hover'] = hover_data

        # Create the plot
        if dim == 2:
            fig = px.scatter(
                data,
                x='x',
                y='y',
                color='label' if labels is not None else None,
                hover_data=['hover'] if hover_data is not None else None,
                title=title
            )
        else:  # 3D case
            fig = px.scatter_3d(
                data,
                x='x',
                y='y',
                z='z',
                color='label' if labels is not None else None,
                hover_data=['hover'] if hover_data is not None else None,
                title=title
            )

        # Update layout for better visualization
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(aspectmode="data") if dim == 3 else None
        )

        # Show the plot
        fig.show()

    except Exception as e:
        print(f"Error in plot_embeddings_plotly: {e}")
