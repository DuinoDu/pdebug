# -*- coding: utf-8 -*-
"""plot using plotly.
"""
import os
import socket
import tempfile
from functools import partial
from typing import Dict, List, Optional, Set, Tuple, Union

from pdebug.utils.env import PLOTLY_INSTALLED

import cv2
import numpy as np

if PLOTLY_INSTALLED:
    import plotly.graph_objects as go
    from plotly.offline import plot
    from plotly.subplots import make_subplots

from .colormap import Colormap

__all__ = [
    "line",
    "lines",
    "line_thick",
    "histogram",
    "histogram2d",
    "CED",
    "error_curve",
    "points",
    "scatter_points",
]


output_root = os.environ.get("VISPERSON_CACHE", "./")
if not os.path.exists(output_root):
    os.makedirs(output_root)


def histogram(
    data: Union[List, np.ndarray],
    name: Union[List[str], str] = "",
    xlabel: str = "index",
    ylabel: str = "count",
    xbins: Optional[int] = None,
    title: str = "histogram",
    output: str = "index.html",
) -> None:
    """
    Draw histogram.

    Args:
        data: input data, ndim is 1 or 2.
        name: data name, with same length.
        xlabel: x axis label name.
        ylabel: y axis label name.
        title: figure name.
        output: saved output html name.

    Example:
        >>> import numpy as np
        >>> from visp.plotly import histogram
        >>> data = [np.random.rand() for _ in range(1000)]
        >>> histogram(data)
        >>> # output is saved in index.html
        >>> multi_data = [data, data, data]
        >>> histogram(multi_data, ['a', 'b', 'c'])

    """
    # keep name as long as data
    assert PLOTLY_INSTALLED, "Please install plotly."
    if isinstance(data, list):
        if isinstance(data[0], list):
            assert isinstance(name, list)
            assert len(data) == len(name)
        else:
            data, name = [data], [name]
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            data, name = [data], [name]
        elif data.ndim == 2:
            assert len(data) == len(name)
        else:
            raise ValueError("Not support ndim={data.ndim}, 1 or 2 is allowed")

    fig = go.Figure()
    for each_name, each_data in zip(name, data):
        fig.add_trace(go.Histogram(x=each_data, name=each_name, xbins=xbins))
    fig.update_layout(
        barmode="overlay",
        xaxis_title_text=xlabel,
        yaxis_title_text=ylabel,
        title_text=title,
    )
    fig.update_traces(opacity=0.75)
    output = os.path.join(output_root, output)
    plot(
        fig,
        filename=output,
        auto_open=False,
        link_text="Powered by visp",
        show_link=True,
    )


def histogram2d(
    x,
    y,
    name=None,
    xlabel="x",
    ylabel="y",
    title="figure",
    output="index.html",
    **kwargs
):
    """
    plot histogram2d
    """
    assert PLOTLY_INSTALLED, "Please install plotly."
    traces = []
    try:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if y.ndim == 1:
            y = y[np.newaxis, :]
        if name is None:
            name = [""]
    except Exception as e:
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
            if name is None:
                name = [""]
        else:
            assert name is not None, "Please provide name when y is list"

    if not isinstance(name, list):
        name = [name]
    assert len(x) == len(y)

    for i in range(len(y)):
        each_name = name[i]
        each_y = y[i]
        each_x = x[i]
        assert len(each_x) == len(each_y)
        sort_index = np.argsort(np.asarray(each_x))
        x_sorted = np.asarray(each_x)[sort_index]
        y_sorted = np.asarray(each_y)[sort_index]
        trace = go.Histogram2d(
            x=x_sorted,
            y=y_sorted,
            name=each_name,
            autobinx=kwargs.get("autobinx", True),
            autobiny=kwargs.get("autobiny", True),
            xbins=kwargs.get("xbins", None),
            ybins=kwargs.get("ybins", None),
            coloraxis="coloraxis",
        )
        fig = go.Figure()
        fig.add_trace(trace)
        if len(y) > 1:
            title_i = title + " (%d)" % i
            output_i = output[:-5] + "_%02d.html" % i
        else:
            title_i = title
            output_i = output

        fig.layout = go.Layout(
            title=go.layout.Title(text=title_i),
            xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=xlabel)),
            yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=ylabel)),
        )
        output_save = os.path.join(output_root, output_i)
        plot(
            fig,
            filename=output_save,
            auto_open=False,
            link_text="Powered by visp",
            show_link=True,
        )


def lines(
    data,
    name,
    xlabel="x",
    ylabel="y",
    title="figure",
    output="index.html",
    x_data=None,
):
    """
    plot multi-lines data
    """
    assert PLOTLY_INSTALLED, "Please install plotly."
    colors = Colormap(len(data))
    traces = []
    for ind, each_data in enumerate(data):
        each_name = name[ind]
        color = colors[ind]
        x = list(range(len(each_data))) if x_data is None else x_data
        trace = go.Scatter(
            x=x, y=each_data, name=each_name, marker=dict(color=color)
        )
        traces.append(trace)

    layout = go.Layout(
        title=go.layout.Title(text=title),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=xlabel)),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=ylabel)),
    )

    fig = go.Figure(data=traces, layout=layout)
    plot(
        fig,
        filename=output,
        auto_open=False,
        link_text="Powered by visp",
        show_link=True,
    )


def line(
    x,
    y,
    name=None,
    xlabel="x",
    ylabel="y",
    title="figure",
    output="index.html",
    mode=None,
):
    """
    plot multi-lines data
    """
    assert PLOTLY_INSTALLED, "Please install plotly."
    traces = []

    if np.asarray(x).ndim == 1:
        x = [x]
    if np.asarray(y).ndim == 1:
        y = [y]
        if name is None:
            name = [""]
    else:
        assert name is not None, "Please provide name when y's dimension > 1"
    assert len(x) == len(y)

    for i in range(len(y)):
        each_name = name[i]
        each_y = y[i]
        each_x = x[i]
        assert len(each_x) == len(each_y)
        sort_index = np.argsort(np.asarray(each_x))
        x_sorted = np.asarray(each_x)[sort_index]
        y_sorted = np.asarray(each_y)[sort_index]
        trace = go.Scatter(x=x_sorted, y=y_sorted, name=each_name, mode=mode)
        traces.append(trace)

    layout = go.Layout(
        title=go.layout.Title(text=title),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=xlabel)),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=ylabel)),
    )

    fig = go.Figure(data=traces, layout=layout)
    output = os.path.join(output_root, output)
    plot(
        fig,
        filename=output,
        auto_open=False,
        link_text="Powered by visp",
        show_link=True,
    )


def line_thick(
    x,
    y,
    name=None,
    xlabel="x",
    ylabel="y",
    title="figure",
    output="index.html",
):
    """
    line with lower and upper bound.
    """
    assert PLOTLY_INSTALLED, "Please install plotly."
    if not isinstance(y, list):
        y = [y]
        name = [""]

    traces = []
    colors = []
    for c in Colormap(len(y)):
        fill_color = "rgba(%d,%d,%d,0.2)" % (c[0], c[1], c[2])
        line_color = "rgb(%d,%d,%d)" % (c[0], c[1], c[2])
        colors.append([fill_color, line_color])

    for ind, _y in enumerate(y):
        each_name = name[ind]
        color = colors[ind]
        # compute upper and lower
        data = dict()
        assert len(x) == len(_y)
        for x_i, y_i in zip(x, _y):
            if x_i in data:
                data[x_i].append(y_i)
            else:
                data[x_i] = [y_i]

        _x = sorted(data.keys())
        x_rev = _x[::-1]
        y_middle = [np.mean(data[k]) for k in _x]
        y_upper = [max(data[k]) for k in _x]
        y_lower = [min(data[k]) for k in _x]
        y_lower = y_lower[::-1]

        traces.append(
            go.Scatter(
                x=_x + x_rev,
                y=y_upper + y_lower,
                fill="toself",
                fillcolor=color[0],
                line_color="rgba(255,255,255,0)",
                showlegend=False,
                name=each_name,
            )
        )
        traces.append(
            go.Scatter(
                x=_x,
                y=y_middle,
                line_color=color[1],
                name=each_name,
            )
        )

    layout = go.Layout(
        title=go.layout.Title(text=title),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=xlabel)),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=ylabel)),
    )

    fig = go.Figure(data=traces, layout=layout)
    output = os.path.join(output_root, output)
    plot(
        fig,
        filename=output,
        auto_open=False,
        link_text="Powered by visp",
        show_link=True,
    )


def points(
    x,
    y,
    names=None,
    xlabel="x",
    ylabel="y",
    title="figure",
    output="index.html",
):
    """
    plot multiple points data.
    """
    assert PLOTLY_INSTALLED, "Please install plotly."
    traces = []
    assert len(x) == len(y)

    for i in range(len(y)):
        each_name = names[i] if names else None
        each_y = y[i]
        each_x = x[i]
        trace = go.Scatter(
            x=[each_x], y=[each_y], name=each_name, mode="markers"
        )
        traces.append(trace)

    layout = go.Layout(
        title=go.layout.Title(text=title),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=xlabel)),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=ylabel)),
    )

    fig = go.Figure(data=traces, layout=layout)
    output = os.path.join(output_root, output)
    plot(
        fig,
        filename=output,
        auto_open=False,
        link_text="Powered by visp",
        show_link=True,
    )


def scatter_points(
    x,
    y,
    names=None,
    xlabel="x",
    ylabel="y",
    title="figure",
    output="index.html",
):
    """
    plot multiple scatter points data.
    """
    assert PLOTLY_INSTALLED, "Please install plotly."
    traces = []
    if len(y) == len(x):
        y = [y]

    for i in range(len(y)):
        each_name = names[i] if names else None
        each_y = y[i]
        each_x = x
        assert len(each_x) == len(each_y)
        trace = go.Scatter(x=each_x, y=each_y, name=each_name, mode="markers")
        traces.append(trace)

    layout = go.Layout(
        title=go.layout.Title(text=title),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=xlabel)),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=ylabel)),
    )

    fig = go.Figure(data=traces, layout=layout)
    output = os.path.join(output_root, output)
    plot(
        fig,
        filename=output,
        auto_open=False,
        link_text="Powered by visp",
        show_link=True,
    )


# custom task


def CED(errs, names, normalized=False, max_value=100):
    """
    Cumulative Error Distribution.
    """
    if normalized:
        raise NotImplementedError
    data = []
    for err in errs:
        err = np.sort(err)
        data_i = []
        for i in range(max_value):
            data_i.append(np.sum(err < i) / len(err))
        data.append(data_i)
    lines(
        data,
        names,
        xlabel="error",
        ylabel="Images Proportion",
        title="CED",
        output="ced.html",
    )


def error_curve(errs, names):
    """
    error by order.
    """
    data = []
    for err in errs:
        err = np.sort(err)
        data_i = []
        for i in np.arange(0, 1, 0.1):
            error = np.mean(err[: int(i * len(err))])
            data_i.append(error)
        data.append(data_i)
    lines(
        data,
        names,
        xlabel="image proportion",
        ylabel="error",
        title="分位误差",
        output="error.html",
    )
