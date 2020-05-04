import numpy as np
from xarray import DataArray
from osgeo import gdal, osr
from obspy.geodetics import gps2dist_azimuth
import utm
import time
import shutil
import os
import subprocess
import warnings
from .travel_time import celerity_travel_time, fdtd_travel_time
import pygmt
from .stack import calculate_semblance
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

gdal.UseExceptions()  # Allows for more Pythonic errors from GDAL

MIN_CELERITY = 220  # [m/s] Used for travel time buffer calculation

OUTPUT_DIR = 'rtm_dem'  # Name of directory to place rtm_dem_*.tif files into
                        # (created in same dir as where function is called)

TMP_TIFF = 'rtm_dem_tmp.tif'  # Filename to use for temporary I/O

# Values in brackets below are latitude, longitude, x_radius, y_radius, spacing
TEMPLATE = 'rtm_dem_{:.4f}_{:.4f}_{}x_{}y_{}m.tif'

NODATA = -9999  # Nodata value to use for output rasters

RESAMPLE_ALG = 'cubicspline'  # Algorithm to use for resampling
                              # See https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r
                              # Good options are 'bilinear', 'cubic',
                              # 'cubicspline', and 'lanczos'

# Define some conversion factors
KM2M = 1000     # [m/km]


def define_grid(lon_0, lat_0, x_radius, y_radius, spacing, projected=False,
                plot_preview=False):
    """
    Define the spatial grid of trial source locations. Grid can be defined in
    either a latitude/longitude or projected UTM (i.e., Cartesian) coordinate
    system.

    Args:
        lon_0 (int or float): [deg] Longitude of grid center
        lat_0 (int or float): [deg] Latitude of grid center
        x_radius (int or float): [deg or m] Distance from `lon_0` to edge of
            grid (i.e., radius)
        y_radius (int or float): [deg or m] Distance from `lat_0` to edge of
            grid (i.e., radius)
        spacing (int or float): [deg or m] Desired grid spacing
        projected (bool): Toggle whether a latitude/longitude or UTM grid is
            used. If `True`, a UTM grid is used and the units of `x_radius`,
            `y_radius`, and `spacing` are interpreted in meters instead of in
            degrees (default: `False`)
        plot_preview (bool): Toggle plotting a preview of the grid for
            reference and troubleshooting (default: `False`)

    Returns:
        :class:`~xarray.DataArray` object containing the grid coordinates and
        metadata
    """

    print('-------------')
    print('DEFINING GRID')
    print('-------------')

    # Make coordinate vectors
    if projected:
        x_0, y_0, zone_number, _ = utm.from_latlon(lat_0, lon_0)
    else:
        x_0, y_0, = lon_0, lat_0
    # Using np.linspace() in favor of np.arange() due to precision advantages
    x, dx = np.linspace(x_0 - x_radius, x_0 + x_radius,
                        int(2 * x_radius / spacing) + 1, retstep=True)
    y, dy = np.linspace(y_0 - y_radius, y_0 + y_radius,
                        int(2 * y_radius / spacing) + 1, retstep=True)

    # Basic grid checks
    if dx != spacing:
        warnings.warn(f'Requested spacing of {spacing} does not match '
                      f'np.linspace()-returned x-axis spacing of {dx}.',
                      RTMWarning)
    if dy != spacing:
        warnings.warn(f'Requested spacing of {spacing} does not match '
                      f'np.linspace()-returned y-axis spacing of {dy}.',
                      RTMWarning)
    if not (x_0 in x and y_0 in y):
        warnings.warn('(x_0, y_0) is not located in grid. Check for numerical '
                      'precision problems (i.e., rounding error).', RTMWarning)
    if projected:
        # Create list of grid corners
        corners = dict(SW=(x.min(), y.min()),
                       NW=(x.min(), y.max()),
                       NE=(x.max(), y.max()),
                       SE=(x.max(), y.min()))
        for label, corner in corners.items():
            # "Un-project" back to latitude/longitude
            lat, lon = utm.to_latlon(*corner, zone_number, northern=lat_0 >= 0,
                                     strict=False)
            # "Re-project" to UTM
            _, _, new_zone_number, _ = utm.from_latlon(lat, lon)
            if new_zone_number != zone_number:
                warnings.warn(f'{label} grid corner locates to UTM zone '
                              f'{new_zone_number} instead of origin UTM zone '
                              f'{zone_number}. Consider reducing grid extent '
                              'or using an unprojected grid.', RTMWarning)

    # Create grid
    data = np.full((y.size, x.size), np.nan)  # Initialize an array of NaNs
    # Add grid "metadata"
    attrs = dict(grid_center=(lon_0, lat_0),
                 x_radius=x_radius,
                 y_radius=y_radius,
                 spacing=spacing)
    if projected:
        # Add the projection information to the grid metadata
        attrs['UTM'] = dict(zone=zone_number, southern_hemisphere=lat_0 < 0)
    else:
        attrs['UTM'] = None
    grid_out = DataArray(data, coords=[('y', y), ('x', x)], attrs=attrs)

    print('Done')

    # Plot grid preview, if specified
    if plot_preview:
        print('Generating grid preview plot...')

        # Make the grid pixel-registered for GMT
        x_new = grid_out.x.values.copy()
        y_new = grid_out.y.values.copy()
        x_new -= spacing / 2
        y_new -= spacing / 2
        x_new = np.hstack([x_new, x_new[-1] + spacing])
        y_new = np.hstack([y_new, y_new[-1] + spacing])
        grid_preview = DataArray(np.zeros((y_new.size, x_new.size)),
                                 coords=[('y', y_new), ('x', x_new)])

        fig = pygmt.Figure()

        if projected:
            plot_width = 6  # [inches]
        else:
            plot_width = 8  # [inches]

        region = [np.floor(x_new.min()), np.ceil(x_new.max()),
                  np.floor(y_new.min()), np.ceil(y_new.max())]

        if projected:
            # Just Cartesian
            proj = f'X{plot_width}i/0'
        else:
            # This is a good projection to use since it preserves area
            proj = 'B{}/{}/{}/{}/{}i'.format(np.mean(region[0:2]),
                                             np.mean(region[2:4]),
                                             region[2], region[3],
                                             plot_width)

        fig.basemap(projection=proj, region=region, frame='af')
        if projected:
            fig.basemap(frame=['SW', 'xa+l"UTM easting (m)"',
                               'ya+l"UTM northing (m)"'])

        # If unprojected plot, draw coastlines
        if not projected:
            fig.coast(A='100+l', water='lightblue', land='lightgrey',
                      shorelines=True)

        # Note that trial source locations are at the CENTER of each plotted
        # grid box
        pygmt.makecpt(A='100+a')
        fig.grdview(grid_preview, Q='sm', cmap=True, meshpen='0.5p')

        # Plot the center of the grid
        fig.plot(x_0, y_0, style=f'c{SYMBOL_SIZE}i', color='limegreen',
                 pen=f'{SYMBOL_PEN}p', label='"Grid center"')

        # Add a legend
        fig.legend(position='JTL+jTL+o0.2i', box='+gwhite+p1p')

        # Show figure
        fig.show(method='external')

        print('Done')

    return grid_out


def produce_dem(grid, external_file=None, plot_output=True, output_file=False):
    """
    Produce a digital elevation model (DEM) with the same extent, spacing, and
    UTM projection as an input projected RTM grid. The data source can be
    either a user-supplied file or global SRTM 1 arc-second (~30 m) data taken
    from the GMT server. A DataArray and (optionally) a GeoTIFF file are
    created. Optionally plot the output DEM. Output GeoTIFF files are placed in
    ``./rtm_dem`` (relative to where this function is called).

    **NOTE 1**

    The filename convention for output files is

    ``rtm_dem_<lat_0>_<lon_0>_<x_radius>x_<y_radius>y_<spacing>m.tif``

    See the docstring for :func:`define_grid` for details/units.

    **NOTE 2**

    GMT caches downloaded SRTM tiles in the directory ``~/.gmt/server/srtm1``.
    If you're concerned about space, you can delete this directory. It will
    be created again the next time an SRTM tile is requested.

    Args:
        grid (:class:`~xarray.DataArray`): Projected :math:`(x, y)` grid; i.e.,
            output of :func:`define_grid` with `projected=True`
        external_file (str): Filename of external DEM file to use. If `None`,
            then SRTM data is used (default: `None`)
        plot_output (bool): Toggle plotting a hillshade of the output DEM -
            useful for identifying voids or artifacts (default: `True`)
        output_file (bool): Toggle creation of output GeoTIFF file (default:
            `False`)

    Returns:
        2-D :class:`~xarray.DataArray` of elevation values with identical shape
        to input grid.
    """

    print('--------------')
    print('PROCESSING DEM')
    print('--------------')

    # If an external DEM file was not supplied, use SRTM data
    if not external_file:

        print('No external DEM file provided. Will use 1 arc-second SRTM data '
              'from GMT server. Checking for GMT 6...')

        # Check for GMT 6
        if shutil.which('gmt') is not None:
            # GMT exists! Now check version
            gmt_version = subprocess.check_output(['gmt', '--version'],
                                                  text=True).strip()
            if gmt_version[0] < '6':
                raise ValueError('GMT 6 is required, but you\'re using GMT '
                                 f'{gmt_version}.')
        else:
            raise OSError('GMT not found on your system. Install GMT 6 and '
                          'try again.')

        print(f'GMT version {gmt_version} found.')

        # Find corners going clockwise from SW (in UTM coordinates)
        corners_utm = [(grid.x.values.min(), grid.y.values.min()),
                       (grid.x.values.min(), grid.y.values.max()),
                       (grid.x.values.max(), grid.y.values.max()),
                       (grid.x.values.max(), grid.y.values.min())]

        # Convert to lat/lon
        lats = []
        lons = []
        for corner in corners_utm:
            lat, lon = utm.to_latlon(*corner,
                                     zone_number=grid.UTM['zone'],
                                     northern=not grid.UTM['southern_hemisphere'])
            lats.append(lat)
            lons.append(lon)

        # [deg] (lonmin, lonmax, latmin, latmax)
        # np.floor and np.ceil here ensure we download more data than we need
        region = [np.floor(min(lons)),
                  np.ceil(max(lons)),
                  np.floor(min(lats)),
                  np.ceil(max(lats))]

        # This command requires GMT 6
        args = ['gmt', 'grdcut', '@srtm_relief_01s', '-V',
                '-R{}/{}/{}/{}'.format(*region),
                '-G{}=gd:GTiff'.format(TMP_TIFF)]
        subprocess.run(args)

        # Remove extra nodata metadata file
        os.remove(TMP_TIFF + '.aux.xml')

        # Remove gmt.history if one was created (depends on global GMT setting)
        try:
            os.remove('gmt.history')
        except OSError:
            pass

        input_raster = TMP_TIFF

    # If an external DEM file was supplied, use it
    else:

        abs_path = os.path.abspath(external_file)

        # Check if file actually exists
        if not os.path.exists(abs_path):
            raise FileNotFoundError(abs_path)

        print(f'Using external DEM file:\n\t{abs_path}')

        input_raster = external_file

    # Define target spatial reference system using grid metadata
    dest_srs = osr.SpatialReference()
    proj_string = '+proj=utm +zone={} +datum=WGS84'.format(grid.UTM['zone'])
    if grid.UTM['southern_hemisphere']:
        proj_string += ' +south'
    dest_srs.ImportFromProj4(proj_string)

    # Control whether or not an output GeoTIFF is produced
    if output_file:
        # Create output raster filename/path
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        output_name = TEMPLATE.format(*grid.grid_center[::-1], grid.x_radius,
                                      grid.y_radius, grid.spacing)
        # Write to file
        output = os.path.join(OUTPUT_DIR, output_name)
        format = 'GTiff'
    else:
        # Write to memory (see https://stackoverflow.com/a/48706963)
        output = ''
        format = 'VRT'

    # Resample input raster, whether it be from an external file or SRTM
    ds = gdal.Warp(output, input_raster, dstSRS=dest_srs, format=format,
                   dstNodata=NODATA,
                   outputBounds=(grid.x.min() - grid.spacing/2,
                                 grid.y.min() - grid.spacing/2,
                                 grid.x.max() + grid.spacing/2,
                                 grid.y.max() + grid.spacing/2),
                   xRes=grid.spacing, yRes=grid.spacing,
                   resampleAlg=RESAMPLE_ALG, copyMetadata=False,
                   )

    # Read resampled DEM into DataArray, set nodata values to np.nan
    dem = grid.copy()
    dem.data = np.flipud(ds.GetRasterBand(1).ReadAsArray())
    dem.data[dem.data == NODATA] = np.nan  # Set NODATA values to np.nan
    dem.data[dem.data < 0] = 0  # Set negative values (underwater?) to 0

    ds = None  # Removes dataset from memory

    # Remove temporary tiff file if it was created
    try:
        os.remove(TMP_TIFF)
    except OSError:
        pass

    # Warn user about possible units ambiguity - we can't check this directly!
    warnings.warn('Elevation units are assumed to be in meters!', RTMWarning)

    if output_file:
        print(f'Created output DEM file:\n\t{os.path.abspath(output)}')

    print('Done')

    if plot_output:

        print('Generating DEM hillshade plot...')

        plot_width = 6  # [inches]
        proj = f'X{plot_width}i/0'

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
        position = f'JMR+o0.9i/0+w{plot_width * aspect_ratio}i/0.15i'
        fig.colorbar(position=position, frame=['a', 'x+l"Elevation (m)"'])

        # Show figure
        fig.show(method='external')

        print('Done')

    return dem


def grid_search(processed_st, grid, time_method, starttime=None, endtime=None,
                stack_method='sum', window=None, overlap=0.5, **time_kwargs):
    """
    Perform a grid search over :math:`x` and :math:`y` and return a 3-D object
    with dimensions :math:`(t, y, x)`. If a UTM grid is used, then the UTM
    :math:`(x, y)` coordinates for each station (`tr.stats.utm_x`,
    `tr.stats.utm_y`) are added to `processed_st`. Optionally provide a 2-D
    array of elevation values to enable 3-D distance computation.

    Args:
        processed_st (:class:`~obspy.core.stream.Stream`): Pre-processed data;
            output of :func:`~rtm.waveform.process_waveforms`
        grid (:class:`~xarray.DataArray`): Grid to use; output of
            :func:`define_grid`
        time_method (str): Method to use for calculating travel times. One of
            `'celerity'` or `'fdtd'`

                * `'celerity'` A single celerity is assumed for
                  propagation. Distances are either 2-D or 3-D (if a DEM
                  is supplied)

                * `'fdtd'` Travel times are calculated using a
                  finite-difference time-domain algorithm which accounts
                  for wave interactions with topography. Only valid for
                  UTM grids

        starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time
            for grid search (default: `None`, which translates to
            `processed_st[0].stats.starttime`)
        endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time for
            grid search (default: `None`, which translates to
            `processed_st[0].stats.endtime`)
        stack_method (str): Method to use for stacking aligned waveforms. One
            of `'sum'`, `'product'`, or `'semblance'` (default: `'sum'`)

                * `'sum'` Sum the aligned waveforms sample-by-sample

                * `'product'` Multiply the aligned waveforms
                  sample-by-sample. Results in a more spatially
                  concentrated stack maximum

                * `'semblance'` Multi-channel coherence computed over
                  defined time windows

        window (int or float): Time window [s] needed for `'semblance'`
            stacking (default: `None`)

        overlap (int or float): Overlap of time window for `'semblance'`
            stacking (default: `0.5`). Must be between 0 and 1, inclusive.

        **time_kwargs: Keyword arguments to be passed on to
            :func:`~rtm.travel_time.celerity_travel_time` or
            :func:`~rtm.travel_time.fdtd_travel_time` functions. For details,
            see the docstrings of those functions.

    Returns:
        :class:`~xarray.DataArray` containing the 3-D :math:`(t, y, x)` stack
        function
    """

    # Check that the requested method works with the provided grid
    if time_method == 'fdtd' and not grid.UTM:
        raise NotImplementedError('The FDTD method is not implemented for '
                                  'unprojected (regional) grids.')

    # Get data dimensions
    npts_st = processed_st[0].stats.npts
    nsta = processed_st.count()

    # Use Stream times to define global time axis for S
    if stack_method == 'semblance':
        if not window:
            raise ValueError('Window must be defined for method '
                             f'\'{stack_method}\'.')

        times = np.arange(processed_st[0].stats.starttime,
                          processed_st[0].stats.endtime, window * (1-overlap))

        # Define number of samples per window and increment
        winlen_samp = window * processed_st[0].stats.sampling_rate
        samp_inc = (1-overlap) * winlen_samp

        # Sample pointer for window-based stack
        samples_stack = np.arange(0, npts_st, samp_inc)

        # Add final window to account for potential uneven number of samples
        if samples_stack[-1] < npts_st - 1:
            samples_stack = np.hstack((samples_stack, npts_st))
        samples_stack = samples_stack.astype(np.int, copy=False)

    else:
        # sample by sample-based stack
        times = processed_st[0].times(type='utcdatetime')

    # Expand grid dimensions in time
    S = grid.expand_dims(time=times.astype('datetime64[ns]')).copy()

    # Project stations in processed_st to UTM if necessary
    if grid.UTM:
        for tr in processed_st:
            tr.stats.utm_x, tr.stats.utm_y = _project_station_to_utm(tr, grid)
            tr.stats.utm_zone = grid.UTM['zone']

    # Call appropriate travel time array creation function
    if time_method == 'celerity':
        travel_times = celerity_travel_time(grid, processed_st, **time_kwargs)
        # Store celerity in S attributes
        S.attrs['celerity'] = time_kwargs['celerity']
    elif time_method == 'fdtd':
        travel_times = fdtd_travel_time(grid, processed_st, **time_kwargs)
    else:
        raise ValueError(f'Travel time calculation method \'{time_method}\' '
                         'not recognized. Method must be either \'celerity\' '
                         'or \'fdtd\'.')

    print('----------------------')
    print('PERFORMING GRID SEARCH')
    print(f'Method = \'{stack_method}\'')
    print('----------------------')

    st = processed_st.copy()

    # Determine the number of samples to be subtracted from travel times
    remove_samp = np.round(np.abs(travel_times.data) *
                           st[0].stats.sampling_rate).astype(int)
    remove_samp[remove_samp > npts_st] = 0

    # Create empty temporary data and stack arrays
    dtmp = np.zeros((nsta, npts_st))
    ny, nx = S.shape[1::]  # Don't count time dimension
    stk = np.empty((nx, ny, len(times)))

    total_its = nx * ny
    counter = 0
    tic = time.time()

    for i, x in enumerate(S.x.values):
        for j, y in enumerate(S.y.values):
            for k, tr in enumerate(st):

                # Number of samples to subtract
                nrem = remove_samp[k, j, i]

                if nrem > 0:
                    # Number of zeroes for padding
                    nrem_zero = np.zeros(nrem)
                    dtmp[k, :] = np.hstack((tr.data[nrem:], nrem_zero))

                else:
                    dtmp[k, :] = tr.data

            if stack_method == 'sum':
                stk[i, j, :] = np.sum(dtmp, axis=0)

            elif stack_method == 'product':
                stk[i, j, :] = np.product(dtmp, axis=0)

            elif stack_method == 'semblance':
                semb = []
                for t in range(len(samples_stack)-1):
                    semb.append(calculate_semblance(
                        dtmp[:, samples_stack[t]:samples_stack[t+1]]))
                stk[i, j, :] = np.array(semb)

            else:
                raise ValueError(f'Stack method \'{stack_method}\' not '
                                 'recognized. Method must be either '
                                 '\'sum\' or \'product\'.')

            S.loc[dict(x=x, y=y)] = stk[i, j, :]

            # Print grid search progress
            counter += 1
            print('{:.1f}%'.format((counter / total_its) * 100), end='\r')

    # Remap for specified start and end times if provided
    if starttime:
        S = S.where((S.time >= np.datetime64(starttime)), drop=True)
    if endtime:
        S = S.where((S.time <= np.datetime64(endtime)), drop=True)

    toc = time.time()
    print(f'Done (elapsed time = {toc-tic:.1f} s)')

    if stack_method == 'product':
        warnings.warn('Watch out for zeros in the stack function due to zeroed '
                      'Traces with this stack method!', RTMWarning)

    return S


def calculate_time_buffer(grid, max_station_dist):
    """
    Utility function for estimating the amount of time needed for an infrasound
    signal to propagate from a source located anywhere in the RTM grid to the
    station farthest from the RTM grid center. This "travel time buffer" helps
    ensure that enough data is downloaded.

    Args:
        grid (:class:`~xarray.DataArray`): Grid to use; output of
            :func:`define_grid`
        max_station_dist (int or float): [km] The longest distance from the
            grid center to a station

    Returns:
        Maximum travel time [s] expected for a source anywhere in the grid to
        the station farthest from the grid center
    """

    # If projected grid, just calculate Euclidean distance for diagonal
    if grid.UTM:
        grid_diagonal = np.linalg.norm([grid.x_radius, grid.y_radius])  # [m]

    # If unprojected grid, find the "longer" of the two possible diagonals
    else:
        center_lon, center_lat = grid.grid_center
        corners = [(center_lat + grid.y_radius,
                    center_lon + grid.x_radius),
                   (center_lat - grid.y_radius,
                    center_lon - grid.x_radius)]
        diags = [gps2dist_azimuth(*corner, center_lat,
                                  center_lon)[0] for corner in corners]
        grid_diagonal = np.max(diags)  # [m]

    # Maximum distance a signal would have to travel is the longest distance
    # from the grid center to a station, PLUS the longest distance from the
    # grid center to a grid corner
    max_propagation_dist = max_station_dist*KM2M + grid_diagonal  # [m]

    # Calculate maximum travel time
    time_buffer = max_propagation_dist / MIN_CELERITY  # [s]

    return time_buffer


def _project_station_to_utm(tr, grid):
    """
    Projects `tr.latitude`, `tr.longitude` into the UTM zone of the input grid.
    Issues a warning if the coordinates of `tr` would locate to another UTM
    grid instead. (The implication here is that the user is trying to use an
    oversized UTM grid and is better off using an unprojected grid instead.)

    Args:
        tr: A :class:`~obspy.core.trace.Trace` containing station coordinates
        grid (:class:`~xarray.DataArray`): Projected :math:`(x, y)` grid; i.e.,
            output of :func:`define_grid` with `projected=True`

    Returns:
        List of [`utm_x`, `utm_y`] coordinates for station associated with `tr`
    """

    grid_zone_number = grid.UTM['zone']
    *station_utm, _, _ = utm.from_latlon(tr.stats.latitude,
                                         tr.stats.longitude,
                                         force_zone_number=grid_zone_number)

    # Check if station is outside of grid UTM zone
    _, _, station_zone_number, _ = utm.from_latlon(tr.stats.latitude,
                                                   tr.stats.longitude)
    if station_zone_number != grid_zone_number:
        warnings.warn(f'{tr.id} locates to UTM zone {station_zone_number} '
                      f'instead of grid UTM zone {grid_zone_number}. Consider '
                      'reducing station search extent or using an unprojected '
                      'grid.', RTMWarning)

    return station_utm
