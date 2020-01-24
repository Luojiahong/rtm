import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib import dates
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from .stack import get_peak_coordinates
import utm
from . import RTMWarning
import pygmt


# Set universal GMT font size
session = pygmt.clib.Session()
session.create('')
session.call_module('gmtset', 'FONT=11p')
session.destroy()

# Marker size for PyGMT
SYMBOL_SIZE = 0.1  # [inches]

# Fraction of total map extent (width) to use for scalebar
SCALE_FRAC = 1/10

# Define some conversion factors
KM2M = 1000    # [m/km]
M2KM = 1/KM2M  # [km/m]
DEG2KM = 111   # [km/deg.] APPROX value at equator


def plot_time_slice(S, processed_st, time_slice=None, label_stations=True,
                    dem=None):
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

    Returns:
        :class:`pygmt.Figure`: Output figure
    """

    st = processed_st.copy()

    # Get coordinates of stack maximum in (latitude, longitude)
    time_max, y_max, x_max, peaks, props = get_peak_coordinates(S)

    # In either case, we convert from UTCDateTime to np.datetime64
    if time_slice:
        time_to_plot = np.datetime64(time_slice)
    else:
        time_to_plot = np.datetime64(time_max)

    slice = S.sel(time=time_to_plot, method='nearest')
    slice.data[slice.data == 0] = np.nan  # Replace zeros with NaN

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

    if S.UTM:
        # Just Cartesian
        proj = f'X{plot_width}i/0'
        # [m] Rounded to nearest `nearest` m
        nearest = 100
        scale_length = np.round((region[1] - region[0]) * SCALE_FRAC / nearest) * nearest

    else:
        # This is a good projection to use since it preserves area
        proj = 'B{}/{}/{}/{}/{}i'.format(np.mean(region[0:2]),
                                         np.mean(region[2:4]),
                                         region[2], region[3],
                                         plot_width)
        # [km] Rounded to nearest `nearest` km
        nearest = 200
        scale_length = np.round(((region[1] - region[0]) * SCALE_FRAC * DEG2KM) / nearest) * nearest

    # Add title
    title = f'Time: {UTCDateTime(slice.time.values.astype(str)).datetime}'

    # Add celerity info if applicable
    if hasattr(S, 'celerity'):
        title = title + f'    Celerity: {S.celerity:g} m/s'

    # Label as global max if applicable
    if slice.time.values == time_max:
        title = 'GLOBAL MAXIMUM    ' + title

    fig.basemap(projection=proj, region=region, frame=['af', f'+t"{title}"'])
    if S.UTM:
        fig.basemap(frame=['SW', 'xa+l"UTM Easting (m)"',
                                 'ya+l"UTM Northing (m)"' ])

    # If unprojected plot, draw coastlines
    if not S.UTM:
        fig.coast(A='100+l', water='lightblue', land='lightgrey',
                  shorelines=True)

    if S.UTM:
        transp = 20  # [%]
    else:
        transp = 30  # [%]

    session = pygmt.clib.Session()
    session.create('')
    with session.virtualfile_from_grid(slice) as grid_file:
        pygmt.makecpt(cmap='magma', series=[slice.data.min(),
                                            slice.data.max()], reverse=True)
        session.call_module('grdview', f'{grid_file} -C -T+s -t{transp}')
    session.destroy()

    # Make colorbar
    if S.UTM:
        # Ensure colorbar height equals map height
        aspect_ratio = (region[3] - region[2]) / (region[1] - region[0])
        position=f'JMR+o0.9i/0+w{plot_width * aspect_ratio}i/0.15i'
    else:
        position=f'JCB+o0/0.5i+h+w{plot_width * 0.75}i/0.15i'
    fig.colorbar(position=position, frame=['a', 'x+l"Stack amplitude"'])

    # If projected plot and a DEM is provided, draw contours
    if S.UTM and dem is not None:
        num_contours = 30
        interval = np.round((dem.max() - dem.min()) / num_contours)
        fig.grdcontour(dem, interval=interval.data, annotation='50+u" m"')  # Assumes meters!

    # Plot the center of the grid
    if S.UTM:
        x_0, y_0, _, _ = utm.from_latlon(*S.grid_center[::-1],
                                         force_zone_number=S.UTM['zone'])
    else:
        x_0, y_0 = S.grid_center
    fig.plot(x_0, y_0, style=f'c{SYMBOL_SIZE}i', color='limegreen', pen=True,
             label='"Grid center"')

    # Plot stack maximum
    if S.UTM:
        # UTM formatting
        label = f'({x_max:.0f}, {y_max:.0f})'
    else:
        # Lat/lon formatting
        label = f'({y_max:.4f}, {x_max:.4f})'
    fig.plot(x_max, y_max, style=f'd{SYMBOL_SIZE}i', color='red', pen=True,
             label=f'"Stack maximum"')
    # Dummy symbol for second line in legend
    fig.plot(0, 0, t=100, pen='white', label=f'"{label}"')

    # Plot stations
    fig.plot(sta_x, sta_y, style=f'i{SYMBOL_SIZE}i', color='100', pen=True,
             label=f'Station')
    if label_stations:
        for x, y, tr in zip(sta_x, sta_y, processed_st):
            fig.text(x=x, y=y, text=f'{tr.stats.network}.{tr.stats.station}',
                     font='10p,100=~1p', justify='LM', D='0.1i/0')

    # Add legend
    fig.legend(position='JTL+jTL+o0.2i', box='+gwhite+p1p')

    # Add scalebar (valid in center of map)
    if S.UTM:
        fig.basemap(L='JBR+jBR+o0.5i+lm+w{}'.format(scale_length))
    else:
        fig.basemap(L='JBR+jBR+o1i+f+l+w{}k+c{}/{}'.format(scale_length,
                                                           *S.grid_center))

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

    ax.set_xlabel(f'Time (s) from {origin_time.datetime}')
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
    tvec = dates.date2num(st_plot[0].stats.starttime.datetime) + st_plot[0].times()/86400

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

    fig, ax = plt.subplots(figsize=(8, 6), nrows=ntra, ncols=1)

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

        ax[i].xaxis_date()
        if i < ntra-1:
            ax[i].set_xticklabels('')

        if label_waveforms:
            ax[i].text(.85, .9,
                       f'{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}',
                       verticalalignment='center', transform=ax[i].transAxes)

    ax[-1].set_xlabel('UTC Time')

    fig.tight_layout()
    plt.subplots_adjust(hspace=.12)
    fig.show()

    return fig


def plot_stack_peak(S, plot_max=False):
    """
    Plot the peak of the stack as a function of time.

    Args:
        S: :class:`~xarray.DataArray` containing the stack function :math:`S`
        plot_max (bool): Plot maximum value with red circle (default: `False`)

    Returns:
        :class:`~matplotlib.figure.Figure`: Output figure
    """

    s_peak = S.max(axis=(1, 2)).data

    fig, ax = plt.subplots(figsize=(8, 4), nrows=1, ncols=1)
    ax.plot(S.time, s_peak, 'k-')
    if plot_max:
        stack_maximum = S.where(S == S.max(), drop=True).squeeze()
        if stack_maximum.size > 1:
            max_indices = np.argwhere(~np.isnan(stack_maximum.data))
            ax.plot(stack_maximum[tuple(max_indices[0])].time,
                    stack_maximum[tuple(max_indices[0])].data, 'ro')
            warnings.warn(f'Multiple global maxima ({len(stack_maximum.data)}) '
                          'present in S!', RTMWarning)
        else:
            ax.plot(stack_maximum.time, stack_maximum.data, 'ro')

    ax.set_xlim(S.time[0].data, S.time[-1].data)
    ax.set_xlabel('UTC Time')
    ax.set_ylabel('Peak Stack Amplitude')

    return fig
