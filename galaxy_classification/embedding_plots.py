import pandas as pd
import plotly.express as px

from ipywidgets import HTML, VBox, HBox
from plotly import graph_objects as go

def plot_embeddings_plotly(catalog):
  """
  Scatter plot of the embeddings of size (n,2).
  catalog provides metadata of the samples, such as catalog.iloc[k]['file_loc'] is the image path
  to the image of embedding embeddings[k,:].
  Hovering over each point shows a thumbnail of the corresponding image.
  """
  # Ensure embeddings are in a DataFrame for easier handling
  df = catalog.copy() #pd.DataFrame(embeddings, columns=['x', 'y'])
  #df.index = catalog.index
  df['image'] = catalog['file_loc']+'_thumb.jpg'  # Add image paths from the catalog
  df['source_id'] = catalog['source_id']
  df['label'] = catalog['galaxy']
  df['pred'] = catalog['galaxy_pred']#predictions[:,0,0] #catalog['galaxy_pred']
  df['pred_class'] = catalog['galaxy_pred'].round()
  df['correct'] = df['pred_class']==df['label']
  df['vis_correct'] = (df['label'].astype(float)-0.5) * (0.5+0.5*(df['correct'].astype(float))) + 0.5 # 0,1 if correct, 0.25,0.75 if incorrect
  df['center_x'] = catalog['center_x']#predictions[:,0,0] #catalog['galaxy_pred']
  df['center_y'] = catalog['center_y']#predictions[:,0,0] #catalog['galaxy_pred']
  df.reset_index(drop=True, inplace=True)

  redblue=df['label'].apply(lambda x: 'red' if x else 'blue')

  fig = go.FigureWidget()
  fig.add_scatter(x=df['x'],y=df['y'],
                  customdata=df[["image",'source_id']].values.reshape([len(df), 2]),
                  mode="markers",
                  marker=dict(size=6, 
                              #color=df['pred'],
                              #color=df['label'],
                              color=df['vis_correct'],
                              symbol=df['label'].apply(lambda x: 'cross' if x else 'x'),
                              colorscale='RdBu')
                  );
  fig.update_layout(width=600, height=600, margin=dict(l=10, r=10, t=10, b=10))  # Adjust the height here as needed

  template="""ind={ind}<br>#{source_id}<br><img src='{image}' style='width:256px; height:256px;' onclick="handleClick('{ind}');"/>
          <br>Source=#{source_id}  correct={correct}
          <br>Label: {label}
          <br>Pred: {pred}"""

  html = HTML("Placeholder")

  from IPython.display import Javascript, display
  # import ipywidgets as widgets
  # js_trigger = widgets.Text(value="HIDDEN TRIGGER", description="Hidden Trigger", layout={'visibility': 'hidden'})

  # # Attach a callback that triggers when the value changes
  # def on_js_call(change):
  #     on_image_click(change['new'])
  #     html.value += f'<br>on_js_call({change["new"]})<br>'

  # js_trigger.observe(on_js_call, names='value')

  # # Display widgets
  # display(js_trigger)


  # def on_image_click(ind):
  #       # Python function to handle the click event
  #       print(f"Image clicked: {ind}")  # Replace this with your desired functionality
  #       html.value = html.value + f"<br>on_image_click {ind}"
  # # Register the Python function in the global namespace
  # from IPython import get_ipython
  # ipython = get_ipython()
  # if "on_image_click" in ipython.user_ns:
  #   del ipython.user_ns["on_image_click"]
  # ipython.push({"on_image_click": on_image_click})

  def update(trace, points, state):
      ind = points.point_inds[0]
      row = df.iloc[ind].to_dict()
      html.value = template.format(**row, ind=ind) + """
      <br>
      END
      """
  display(Javascript("console.log('JavaScript is working!');"))
  # display(Javascript("""console.log(document, document.getElementById('msgDiv'))
  #       let msgDiv = document.getElementById('msgDiv')
  #       if (msgDiv) {
  #         msgDiv.innerHTML = 'INIT';
  #       }
  #       var handleClick = function(ind) {
  #         let msgDiv = document.getElementById('msgDiv')
  #         if (msgDiv) {
  #           msgDiv.innerHTML = 'handleClick: ' + ind;
  #         }
  #         console.log('handleClick',ind)
  #         var kernel = IPython.notebook.kernel;
  #         kernel.execute("on_image_click(" + ind + ")");
  #   }"""))

  fig.data[0].on_hover(update)

  display(HBox([fig, html]))


# https://gist.github.com/raphaeljolivet/f076256e589f67c028a6bffaab279601
def interactive_plot(df, plot, template, event="hover") :
    """
    Make a plot react on hover or click of a data point and update a HTML preview below it.
    **template** Should be a string and contain placeholders like {colname} to be replaced by the value
    of the corresponding data row.
    
    """

    html = HTML("Placeholder")

    def update(trace, points, state):
        ind = points.point_inds[0]
        row = df.loc[ind].to_dict()
        html.value = template.format(**row)


    if event == "hover" :
        fig.data[0].on_hover(update)
    else :
        fig.data[0].on_click(update)

    return VBox([fig, html])