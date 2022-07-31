from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None):
        self.fig = Figure()
        self.fig.set_tight_layout(True) # to properly scale all elements on canvas        

        super(MplCanvas, self).__init__(self.fig)