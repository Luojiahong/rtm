import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.dates as mdates
from obspy.geodetics import gps2dist_azimuth
from .stack import get_peak_coordinates
import utm
from datetime import datetime
from xarray import DataArray
import os
import pygmt
from . import RTMWarning


# Set universal GMT font size
session = pygmt.clib.Session()
session.create('')
session.call_module('gmtset', 'FONT=11p')
session.destroy()

# Marker size for PyGMT
SYMBOL_SIZE = 0.1  # [inches]

# Symbol pen size thickness
SYMBOL_PEN = 0.75  # [pt]

# Fraction of total map extent (width) to use for scalebar
SCALE_FRAC = 1/10

# Width of grid and DEM preview plots
PREVIEW_PLOT_WIDTH = 6  # [in]

# Define some conversion factors
KM2M = 1000    # [m/km]
M2KM = 1/KM2M  # [km/m]
DEG2KM = 111   # [km/deg.] APPROX value at equator


def plot_time_slice(S, processed_st, time_slice=None, label_stations=True,
                    dem=None, cont_int=5, annot_int=50):
    """
    Plot a time slice through :math:`S` to produce a map-view plot. If time is
    not specified, then the slice corresponds to the maximum of :math:`S` in
    the time direction.

    Args:
        S (:class:`~xarray.DataArray`): The stack function :math:`S`
        processed_st (:class:`~obspy.core.stream.Stream`): Pre-processed
            Stream; output of :func:`~rtm.waveform.process_waveforms` (This is
            needed because Trace metadata from this Stream are used to plot
            stations on the map)
        time_slice (:class:`~obspy.core.utcdatetime.UTCDateTime`): Time of
            desired time slice. The nearest time in :math:`S` to this specified
            time will be plotted. If `None`, the time corresponding to
            :math:`\max(S)` is used (default: `None`)
        label_stations (bool): Toggle labeling stations with network and
            station codes (default: `True`)
        dem (:class:`~xarray.DataArray`): Overlay time slice on a user-supplied
            DEM from :class:`~rtm.grid.produce_dem` (default: `None`)
        cont_int (int): Contour interval [m] for plots with DEM data
        annot_int (int): Annotated contour interval [m] for plots with DEM data
            (these contours are thicker and labeled)

    Returns:
        :class:`pygmt.Figure`: Output figure
    """

    st = processed_st.copy()

    # Get coordinates of stack maximum
    time_max, y_max, x_max, peaks, props = get_peak_coordinates(S)

    # In either case, we convert from UTCDateTime to np.datetime64
    if time_slice:
        time_to_plot = np.datetime64(time_slice)
    else:
        time_to_plot = np.datetime64(time_max)

    slice = S.sel(time=time_to_plot, method='nearest')
    slice.data[slice.data == 0] = np.nan  # Replace zeros with NaN
    if S.UTM and dem is not None:
        # Mask areas outside of DEM extent
        dem_slice = dem.sel(x=slice.x, y=slice.y, method='nearest')  # Select subset of DEM that slice occupies
        slice.data[np.isnan(dem_slice.data)] = np.nan

    fig = pygmt.Figure()

    if S.UTM:
        plot_width = 6  # [inches]
    else:
        plot_width = 8  # [inches]

    # Define coordinates of stations
    if S.UTM:
        sta_x = []
        sta_y = []
        for tr in processed_st:
            utm_x, utm_y, _, _ = utm.from_latlon(tr.stats.latitude,
                                                 tr.stats.longitude,
                                                 force_zone_number=S.UTM['zone'])
            sta_x.append(utm_x)
            sta_y.append(utm_y)
    else:
        sta_x = [tr.stats.longitude for tr in st]
        sta_y = [tr.stats.latitude for tr in st]

    # Define region
    if not S.UTM:
        # Rescale from 0-360 degrees
        xmin = (np.hstack([sta_x, S.x.min()]) % 360).min()
        xmax = (np.hstack([sta_x, S.x.max()]) % 360).max()
        buffer = 0  # [deg.]
    else:
        xmin = np.hstack([sta_x, S.x.min()]).min()
        xmax = np.hstack([sta_x, S.x.max()]).max()
        buffer = 0.03 * (xmax - xmin) # 3% buffer [m]
    ymin = np.hstack([sta_y, S.y.min()]).min()
    ymax = np.hstack([sta_y, S.y.max()]).max()

    region = [np.floor(xmin - buffer), np.ceil(xmax + buffer),
              np.floor(ymin - buffer), np.ceil(ymax + buffer)]

    # Define projection and a sensible scalebar length
    if S.UTM:
        # Just Cartesian
        proj = f'X{plot_width}i/0'
        # [m] Rounded to nearest `nearest` m
        nearest = 100
        scale_length = np.round((region[1] - region[0]) * SCALE_FRAC / nearest) * nearest
    else:
        # Albers
        proj = _albers(region, plot_width)
        # [km] Rounded to nearest `nearest` km
        nearest = 200
        scale_length = np.round(((region[1] - region[0]) * SCALE_FRAC * DEG2KM) / nearest) * nearest

    # Add title
    time_round = np.datetime64(slice.time.values + np.timedelta64(500, 'ms'), 's').astype(datetime)  # Nearest second
    title = 'Time: {}'.format(time_round)

    # Add celerity info if applicable
    if hasattr(S, 'celerity'):
        title = title + f'    Celerity: {S.celerity:g} m/s'

    # Label as global max if applicable
    if slice.time.values == time_max:
        title = 'GLOBAL MAXIMUM    ' + title

    fig.basemap(projection=proj, region=region, frame=['af', f'+t"{title}"'])
    if S.UTM:
        fig.basemap(frame=['SW', 'xa+l"UTM easting (m)"',
                                 'ya+l"UTM northing (m)"' ])

    # If unprojected plot, draw coastlines
    if not S.UTM:
        fig.coast(A='100+l', water='lightblue', land='lightgrey',
                  shorelines=True)

    if S.UTM and dem is not None:
        transp = 30  # [%]
    else:
        transp = 50  # [%]

    # If projected plot and a DEM is provided, draw contours
    if S.UTM and dem is not None:
        # Assumes meters!
        fig.grdcontour(dem, interval=cont_int, annotation=f'{annot_int}+u" m"')

    # Make heatmap of slice
    pygmt.makecpt(cmap='viridis', series=[np.nanmin(slice.data),
                                          np.nanmax(slice.data)])
    fig.grdview(slice, cmap=True, T='+s', t=transp)

    # Make colorbar
    if S.UTM:
        # Ensure colorbar height equals map height
        aspect_ratio = (region[3] - region[2]) / (region[1] - region[0])
        position = f'JMR+o0.9i/0+w{plot_width * aspect_ratio}i/0.15i'
    else:
        position = f'JCB+o0/0.5i+h+w{plot_width * 0.75}i/0.15i'
    fig.colorbar(position=position, frame=['a', 'x+l"Stack amplitude"'])

    # Plot the center of the grid
    if S.UTM:
        x_0, y_0, _, _ = utm.from_latlon(*S.grid_center[::-1],
                                         force_zone_number=S.UTM['zone'])
    else:
        x_0, y_0 = S.grid_center
    fig.plot(x_0, y_0, style=f'c{SYMBOL_SIZE}i', color='limegreen',
             pen=f'{SYMBOL_PEN}p', label='"Grid center"')

    # Plot stack maximum
    fig.plot(x_max, y_max, style=f'a{SYMBOL_SIZE}i', color='red',
             pen=f'{SYMBOL_PEN}p', label=f'"Stack max"')
    if not S.UTM:
        # Append coordinates of stack max (dummy symbol for second line in legend)
        fig.plot(0, 0, t=100, pen='white',
                 label=f'"({y_max:.4f}, {x_max:.4f})"')

    # Plot stations
    fig.plot(sta_x, sta_y, style=f'i{SYMBOL_SIZE}i', color='orange',
             pen=f'{SYMBOL_PEN}p', label=f'Station')
    if label_stations:
        for x, y, tr in zip(sta_x, sta_y, processed_st):
            fig.text(x=x, y=y, text=f'{tr.stats.network}.{tr.stats.station}',
                     font='10p,white=~1p', justify='LM', D='0.1i/0')

    # Add legend
    fig.legend(position='JTL+jTL+o0.2i', box='+gwhite+p1p')

    # Add scalebar (valid in center of map)
    if S.UTM:
        fig.basemap(L='JBR+jBR+o0.5i+lm+w{}'.format(scale_length))
    else:
        fig.basemap(L='JBR+jBR+o1i+f+l+w{}k+c{}/{}'.format(scale_length,
                                                           *S.grid_center))

    # Show figure
    fig.show(method='external')

    return fig


def plot_record_section(st, origin_time, source_location, plot_celerity=None,
                        label_waveforms=True):
    """
    Plot a record section based upon user-provided source location and origin
    time. Optionally plot celerity for reference, with two plotting options.

    Args:
        st (:class:`~obspy.core.stream.Stream`): Any Stream object with
            `tr.stats.latitude`, `tr.stats.longitude` attached
        origin_time (:class:`~obspy.core.utcdatetime.UTCDateTime`): Origin time
            for record section
        source_location (tuple): Tuple of (`lat`, `lon`) specifying source
            location
        plot_celerity: Can be either `'range'` or a single celerity or a list
            of celerities. If `'range'`, plots a continuous swatch of
            celerities from 260-380 m/s. Otherwise, plots specific celerities.
            If `None`, does not plot any celerities (default: `None`)
        label_waveforms (bool): Toggle labeling waveforms with network and
            station codes (default: `True`)

    Returns:
        :class:`~matplotlib.figure.Figure`: Output figure
    """

    st_edit = st.copy()

    for tr in st_edit:
        tr.stats.distance, _, _ = gps2dist_azimuth(*source_location,
                                                   tr.stats.latitude,
                                                   tr.stats.longitude)

    st_edit.trim(origin_time)

    fig = plt.figure(figsize=(12, 8))

    st_edit.plot(fig=fig, type='section', orientation='horizontal',
                 fillcolors=('black', 'black'))

    ax = fig.axes[0]

    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    if label_waveforms:
        for tr in st_edit:
            ax.text(1.01, tr.stats.distance / 1000,
                    f'{tr.stats.network}.{tr.stats.station}',
                    verticalalignment='center', transform=trans, fontsize=10)
        pad = 0.1  # Move colorbar to the right to make room for labels
    else:
        pad = 0.05  # Matplotlib default for vertical colorbars

    if plot_celerity:

        # Check if user requested a continuous range of celerities
        if plot_celerity == 'range':
            inc = 0.5  # [m/s]
            celerity_list = np.arange(220, 350 + inc, inc)  # [m/s] Includes
                                                            # all reasonable
                                                            # celerities
            zorder = -1

        # Otherwise, they provided specific celerities
        else:
            # Type conversion
            if type(plot_celerity) is not list:
                plot_celerity = [plot_celerity]

            celerity_list = plot_celerity
            celerity_list.sort()
            zorder = None

        # Create colormap of appropriate length
        cmap = plt.cm.get_cmap('rainbow', len(celerity_list))
        colors = [cmap(i) for i in range(cmap.N)]

        xlim = np.array(ax.get_xlim())
        y_max = ax.get_ylim()[1]  # Save this for re-scaling axis

        for celerity, color in zip(celerity_list, colors):
            ax.plot(xlim, xlim * celerity / 1000, label=f'{celerity:g}',
                    color=color, zorder=zorder)

        ax.set_ylim(top=y_max)  # Scale y-axis to pre-plotting extent

        # If plotting a continuous range, add a colorbar
        if plot_celerity == 'range':
            mapper = plt.cm.ScalarMappable(cmap=cmap)
            mapper.set_array(celerity_list)
            cbar = fig.colorbar(mapper, label='Celerity (m/s)', pad=pad,
                                aspect=30)
            cbar.ax.minorticks_on()

        # If plotting discrete celerities, just add a legend
        else:
            ax.legend(title='Celerity (m/s)', loc='lower right', framealpha=1,
                      edgecolor='inherit')

    ax.set_ylim(bottom=0)  # Show all the way to zero offset

    time_round = np.datetime64(origin_time + 0.5, 's').astype(datetime)  # Nearest second
    ax.set_xlabel('Time (s) from {}'.format(time_round))
    ax.set_ylabel('Distance (km) from '
                  '({:.4f}, {:.4f})'.format(*source_location))

    fig.tight_layout()
    fig.show()

    return fig


def plot_st(st, filt, equal_scale=False, remove_response=False,
            label_waveforms=True):
    """
    Plot Stream waveforms in a publication-quality figure. Multiple plotting
    options, including filtering.

    Args:
        st (:class:`~obspy.core.stream.Stream`): Any Stream object
        filt (list): A two-element list of lower and upper corner frequencies
            for filtering. Specify `None` if no filtering is desired.
        equal_scale (bool): Set equal scale for all waveforms (default:
            `False`)
        remove_response (bool): Remove response by applying sensitivity
        label_waveforms (bool): Toggle labeling waveforms with network and
            station codes (default: `True`)

    Returns:
        :class:`~matplotlib.figure.Figure`: Output figure
    """

    st_plot = st.copy()
    ntra = len(st)
    tvec = st_plot[0].times('matplotlib')

    if remove_response:
        print('Applying sensitivity')
        st_plot.remove_sensitivity()

    if filt:
        print('Filtering between %.1f-%.1f Hz' % (filt[0], filt[1]))

        st_plot.detrend(type='linear')
        st_plot.taper(max_percentage=.01)
        st_plot.filter("bandpass", freqmin=filt[0], freqmax=filt[1], corners=2,
                       zerophase=True)

    if equal_scale:
        ym = np.max(st_plot.max())

    fig, ax = plt.subplots(figsize=(8, 6), nrows=ntra, sharex=True)

    for i, tr in enumerate(st_plot):
        ax[i].plot(tvec, tr.data, 'k-')
        ax[i].set_xlim(tvec[0], tvec[-1])
        if equal_scale:
            ax[i].set_ylim(-ym, ym)
        else:
            ax[i].set_ylim(-tr.data.max(), tr.data.max())
        plt.locator_params(axis='y', nbins=4)
        ax[i].tick_params(axis='y', labelsize=8)
        ax[i].ticklabel_format(useOffset=False, style='plain')

        if tr.stats.channel[1] == 'D':
            ax[i].set_ylabel('Pressure [Pa]', fontsize=8)
        else:
            ax[i].set_ylabel('Velocity [m/s]', fontsize=8)

        if label_waveforms:
            ax[i].text(.85, .9,
                       f'{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}',
                       verticalalignment='center', transform=ax[i].transAxes)

    # Tick locating and formatting
    locator = mdates.AutoDateLocator()
    ax[-1].xaxis.set_major_locator(locator)
    ax[-1].xaxis.set_major_formatter(_UTCDateFormatter(locator))
    fig.autofmt_xdate()

    fig.tight_layout()
    plt.subplots_adjust(hspace=.12)
    fig.show()

    return fig


def plot_stack_peak(S, plot_max=False, ax=None):
    """
    Plot the stack function (at the spatial stack max) as a function of time.

    Args:
        S: :class:`~xarray.DataArray` containing the stack function :math:`S`
        plot_max (bool): Plot maximum value with red circle (default: `False`)
        ax (:class:`~matplotlib.axes.Axes`): Pre-existing axes to plot into

    Returns:
        :class:`~matplotlib.figure.Figure`: Output figure
    """

    s_peak = S.max(axis=(1, 2)).data

    if not ax:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()  # Get figure to which provided axis belongs
    ax.plot(S.time, s_peak, 'k-')
    if plot_max:
        stack_maximum = S.where(S == S.max(), drop=True).squeeze()
        marker_kwargs = dict(marker='*', color='red', edgecolor='black', s=150,
                             zorder=5, clip_on=False)
        if stack_maximum.size > 1:
            max_indices = np.argwhere(~np.isnan(stack_maximum.data))
            ax.scatter(stack_maximum[tuple(max_indices[0])].time.data,
                       stack_maximum[tuple(max_indices[0])].data,
                       **marker_kwargs)
            warnings.warn(f'Multiple global maxima ({len(stack_maximum.data)}) '
                          'present in S!', RTMWarning)
        else:
            ax.scatter(stack_maximum.time.data, stack_maximum.data,
                       **marker_kwargs)

    ax.set_xlim(S.time[0].data, S.time[-1].data)
    ax.set_ylim(bottom=0)  # Never can go below zero
    ax.set_ylabel('Max stack amplitude')

    # Tick locating and formatting
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(_UTCDateFormatter(locator))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    return fig


def plot_grid_preview(grid):
    """
    Plot a preview of the grid of trial sources.

    Args:
        grid (:class:`~xarray.DataArray`): Grid to plot, e.g. output of
            :func:`~rtm.grid.define_grid`
    """

    # Make the grid pixel-registered for GMT
    x_new = grid.x.values.copy()
    y_new = grid.y.values.copy()
    x_new -= grid.spacing / 2
    y_new -= grid.spacing / 2
    x_new = np.hstack([x_new, x_new[-1] + grid.spacing])
    y_new = np.hstack([y_new, y_new[-1] + grid.spacing])
    grid_preview = DataArray(np.zeros((y_new.size, x_new.size)),
                             coords=[('y', y_new), ('x', x_new)])

    fig = pygmt.Figure()

    region = [np.floor(x_new.min()), np.ceil(x_new.max()),
              np.floor(y_new.min()), np.ceil(y_new.max())]

    if grid.UTM:
        # Just Cartesian
        proj = f'X{PREVIEW_PLOT_WIDTH}i/0'
    else:
        # Albers
        proj = _albers(region, PREVIEW_PLOT_WIDTH)

    fig.basemap(projection=proj, region=region, frame='af')
    if grid.UTM:
        fig.basemap(frame=['SW', 'xa+l"UTM easting (m)"',
                           'ya+l"UTM northing (m)"'])

    # If unprojected plot, draw coastlines
    if not grid.UTM:
        fig.coast(A='100+l', water='lightblue', land='lightgrey',
                  shorelines=True)

    # Note that trial source locations are at the CENTER of each plotted
    # grid box
    pygmt.makecpt(A='100+a')
    fig.grdview(grid_preview, Q='sm', cmap=True, meshpen='0.5p')

    # Plot the center of the grid
    if grid.UTM:
        x_0, y_0, zone_number, _ = utm.from_latlon(*grid.grid_center[::-1])
    else:
        x_0, y_0, = grid.grid_center
    fig.plot(x_0, y_0, style=f'c{SYMBOL_SIZE}i', color='limegreen',
             pen=f'{SYMBOL_PEN}p', label='"Grid center"')

    # Add a legend
    fig.legend(position='JTL+jTL+o0.2i', box='+gwhite+p1p')

    # Show figure
    fig.show(method='external')


def plot_dem(dem, external_file):
    """
    Plot a DEM hillshade.

    Args:
        dem (:class:`~xarray.DataArray`): Projected DEM, e.g. output of
            :func:`~rtm.grid.produce_dem`
        external_file (str): Filename of external DEM file used. `None` if
            SRTM data was used
    """

    proj = f'X{PREVIEW_PLOT_WIDTH}i/0'

    x = dem.x.values
    y = dem.y.values

    # This will perfectly trim axis to DEM extent, considering registration
    region = [np.floor(x.min()) - dem.spacing / 2,
              np.ceil(x.max()) + dem.spacing / 2,
              np.floor(y.min()) - dem.spacing / 2,
              np.ceil(y.max()) + dem.spacing / 2]

    fig = pygmt.Figure()

    # Create title
    if external_file:
        source_label = os.path.abspath(external_file)
    else:
        source_label = '1 arc-second SRTM data'
    title = '{}, resampled to {} m spacing'.format(source_label,
                                                   dem.spacing)

    # Create basemap
    fig.basemap(projection=proj, region=region,
                frame=['af', f'+t"{title}"'])
    fig.basemap(frame=['SW', 'xa+l"UTM easting (m)"',
                       'ya+l"UTM northing (m)"'])

    # Plot hillshade
    with pygmt.helpers.GMTTempFile() as tmp_grd:
        session = pygmt.clib.Session()
        session.create('')
        with session.virtualfile_from_grid(dem) as dem_file:
            session.call_module('grdgradient',
                                f'{dem_file} -A-45 -Nt1- -G{tmp_grd.name}')
        session.destroy()
        fig.grdimage(dem, cmap='magma', E=300, Q=True, I=tmp_grd.name)

    # Plot the center of the grid
    x_0, y_0, *_ = utm.from_latlon(*dem.grid_center[::-1])
    fig.plot(x_0, y_0, style=f'c{SYMBOL_SIZE}i', color='limegreen',
             pen=f'{SYMBOL_PEN}p', label='"Grid center"')

    # Add a legend
    fig.legend(position='JTL+jTL+o0.2i', box='+gwhite+p1p')

    # Add a colorbar
    aspect_ratio = (region[3] - region[2]) / (region[1] - region[0])
    position = f'JMR+o0.9i/0+w{PREVIEW_PLOT_WIDTH * aspect_ratio}i/0.15i'
    fig.colorbar(position=position, frame=['a', 'x+l"Elevation (m)"'])

    # Show figure
    fig.show(method='external')


def _albers(region, width):
    """
    Create an
    `Albers conic equal area projection <https://docs.generic-mapping-tools.org/6.0/gmt.html#jb-full>`__
    for use with PyGMT.

    Args:
        region (list): Plotting region as ``[xmin, xmax, ymin, ymax]`` [deg.]
        width (int or float): Plot width [inches]

    Returns:
        str: The formatted projection string
    """

    # This is a good projection to use since it preserves area
    proj = 'B{}/{}/{}/{}/{}i'.format(np.mean(region[0:2]),
                                     np.mean(region[2:4]),
                                     region[2], region[3],
                                     width)
    return proj


# Subclass ConciseDateFormatter (modifies __init__() and set_axis() methods)
class _UTCDateFormatter(mdates.ConciseDateFormatter):
    def __init__(self, locator, tz=None):
        super().__init__(locator, tz=tz, show_offset=True)

        # Re-format datetimes
        self.formats[5] = '%H:%M:%S.%f'
        self.zero_formats = self.formats
        self.offset_formats = [
            'UTC time',
            'UTC time in %Y',
            'UTC time in %B %Y',
            'UTC time on %Y-%m-%d',
            'UTC time on %Y-%m-%d',
            'UTC time on %Y-%m-%d',
        ]

    def set_axis(self, axis):
        self.axis = axis

        # If this is an x-axis (usually is!) then center the offset text
        if self.axis.axis_name == 'x':
            offset = self.axis.get_offset_text()
            offset.set_horizontalalignment('center')
            offset.set_x(0.5)
