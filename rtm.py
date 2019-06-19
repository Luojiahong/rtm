#%% (1) Grab and process the data

from obspy import UTCDateTime
import json
from waveform_utils import gather_waveforms, process_waveforms

# Start and end of time window containing (suspected) events
STARTTIME = UTCDateTime('2016-05-22T07:45:00')
ENDTIME = STARTTIME + 30*60

FREQ_MIN = 0.5          # [Hz] Lower bandpass corner
FREQ_MAX = 2            # [Hz] Upper bandpass corner

DECIMATION_RATE = 0.05  # [Hz] New sampling rate to use for decimation

SMOOTH_WIN = 120        # [s] Smoothing window duration

AGC_WIN = 250           # [s] AGC window duration
AGC_METHOD = 'gismo'    # Method to use for AGC, specify 'gismo' or 'walker'

# watc_credentials.json contains a single line with format ["user", "password"]
with open('watc_credentials.json') as f:
    watc_username, watc_password = json.load(f)

st = gather_waveforms(source='IRIS', network='AK,TA',
                      station='HOM,M19K,M22K,O20K,O22K,RC01',
                      starttime=STARTTIME, endtime=ENDTIME,
                      remove_response=True, watc_username=watc_username,
                      watc_password=watc_password)

agc_params = dict(win_sec=AGC_WIN, method=AGC_METHOD)

st_proc = process_waveforms(st, freqmin=FREQ_MIN, freqmax=FREQ_MAX,
                            envelope=True, smooth_win=SMOOTH_WIN,
                            decimation_rate=DECIMATION_RATE, agc_params=None,
                            normalize=True, plot_steps=False)

#%% (2) Define grid

from grid_utils import define_grid

LON_0 = -152.9902  # [deg] Longitude of grid center
LAT_0 = 60.0183    # [deg] Latitude of grid center

PROJECTED = True

if PROJECTED:
    X_RADIUS = 50000  # [m] E-W grid radius (half of grid "width")
    Y_RADIUS = 50000  # [m] N-S grid radius (half of grid "height")
    SPACING = 5000    # [m] Grid spacing

else:
    X_RADIUS = 5   # [deg] E-W grid radius (half of grid "width")
    Y_RADIUS = 5   # [deg] N-S grid radius (half of grid "height")
    SPACING = 0.5  # [deg] Grid spacing

grid = define_grid(lon_0=LON_0, lat_0=LAT_0, x_radius=X_RADIUS,
                   y_radius=Y_RADIUS, spacing=SPACING, projected=PROJECTED,
                   plot=False)

#%% (3) Grid search

from obspy.geodetics import gps2dist_azimuth
import numpy as np
import utm
import warnings
import time

STACK_METHOD = 'sum'  # Choose either 'sum' or 'product'

CELERITY_LIST = [295, 300, 305, 310]  # [m/s]

# Define global time axis using the first Trace of the input Stream
times = st_proc[0].times(type='utcdatetime')

# Expand grid dimensions in celerity and time
stack_array = grid.expand_dims(dict(celerity=np.float64(CELERITY_LIST))).copy()
stack_array = stack_array.expand_dims(dict(time=times.astype('datetime64[ns]'))).copy()

# Pre-allocate NumPy array to store Streams for each grid point
shifted_streams = np.empty(shape=stack_array.shape[1:], dtype=object)

total_its = np.product(stack_array.shape[1:])  # Don't count the time dimension
counter = 0
tic = time.process_time()
for i, celerity in enumerate(stack_array['celerity'].values):

    for j, y_coord in enumerate(stack_array['y']):

        for k, x_coord in enumerate(stack_array['x']):

            st = st_proc.copy()

            for tr in st:

                if grid.attrs['UTM']:
                    grid_zone_number = grid.attrs['UTM']['zone']
                    *station_utm, _, _ = utm.from_latlon(tr.stats.latitude,
                                                         tr.stats.longitude,
                                                         force_zone_number=grid_zone_number)

                    # Check if station is outside of grid UTM zone
                    _, _, station_zone_number, _ = utm.from_latlon(tr.stats.latitude,
                                                                   tr.stats.longitude)
                    if station_zone_number != grid_zone_number:
                        warnings.warn(f'{tr.id} locates to UTM zone '
                                      f'{station_zone_number} instead of grid UTM '
                                      f'zone {grid_zone_number}. Consider '
                                      'reducing station search extent or using an '
                                      'unprojected grid.')

                    # Distance is in meters
                    distance = np.linalg.norm(np.array(station_utm) - np.array([x_coord, y_coord]))

                else:
                    # Distance is in meters
                    distance, _, _ = gps2dist_azimuth(y_coord, x_coord,
                                                      tr.stats.latitude,
                                                      tr.stats.longitude)

                time_shift = distance / celerity  # [s]
                tr.stats.starttime = tr.stats.starttime - time_shift
                tr.stats.processing.append(f'RTM: Shifted by -{time_shift:.2f} s')

            # Trim to time limits of input Stream
            st.trim(times[0], times[-1], pad=True, fill_value=0)

            if STACK_METHOD == 'sum':
                stack = np.sum([tr.data for tr in st], axis=0)

            elif STACK_METHOD == 'product':
                stack = np.product([tr.data for tr in st], axis=0)

            else:
                raise ValueError(f'Stack method \'{STACK_METHOD}\' not '
                                 'recognized. Method must be either \'sum\' or '
                                 '\'product\'.')

            # Assign the stacked time series to this latitude/longitude point
            stack_array.loc[dict(x=x_coord, y=y_coord,
                                 celerity=celerity)] = stack

            # Save the time-shifted Stream
            shifted_streams[i, j, k] = st

            # Print grid search progress
            counter += 1
            print('{:.1f}%'.format((counter / total_its) * 100))

toc = time.process_time()
print(f'Done (elapsed time = {toc-tic:.1f} s)')

#%% (4) Plot

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen

# Get coordinates of peak
max_coords = stack_array.where(stack_array == stack_array.max(),
                               drop=True).squeeze()
t_max = max_coords['time'].values
c_max = max_coords['celerity'].values
y_max = max_coords['y'].values
x_max = max_coords['x'].values

if stack_array.attrs['UTM']:
    proj = ccrs.UTM(**stack_array.attrs['UTM'])
    transform = proj
else:
    # This is a good projection to use since it preserves area
    proj = ccrs.AlbersEqualArea(central_longitude=LON_0,
                                central_latitude=LAT_0,
                                standard_parallels=(
                                    stack_array['y'].values.min(),
                                    stack_array['y'].values.max())
                                )
    transform = ccrs.PlateCarree()

fig, ax = plt.subplots(figsize=(10, 10),
                       subplot_kw=dict(projection=proj))

# Since projected grids cover less area and may not include coastlines,
# use a background image to provide geographical context (can be slow)
if stack_array.attrs['UTM']:
    zoom_level = 8
    ax.add_image(Stamen(style='terrain-background'), zoom_level)

# Since unprojected grids have regional/global extent, just show the
# coastlines
else:
    scale = '50m'
    feature = cfeature.LAND.with_scale(scale)
    ax.add_feature(feature, facecolor=cfeature.COLORS['land'],
                   edgecolor='black')
    ax.background_patch.set_facecolor(cfeature.COLORS['water'])

stack_array.sel(time=t_max, celerity=c_max).plot.pcolormesh(ax=ax, alpha=0.5,
                                                            transform=transform)

# Plot center of grid
ax.scatter(LON_0, LAT_0, s=100, color='red', marker='*',
           transform=ccrs.Geodetic())

# Plot stations
for tr in st_proc:
    ax.scatter(tr.stats.longitude,  tr.stats.latitude, color='black',
               transform=ccrs.Geodetic())
    ax.text(tr.stats.longitude, tr.stats.latitude,
            '  {}.{}'.format(tr.stats.network, tr.stats.station),
            verticalalignment='center_baseline', horizontalalignment='left',
            transform=ccrs.Geodetic())

fig.show()

# Processed (input) Stream
fig = plt.figure()
st_proc.plot(fig=fig)
fig.show()

# Time-shifted (output) Stream
inds = np.argwhere(stack_array.data == stack_array.data.max())[0]
st = shifted_streams[tuple(inds[1:])]
fig = plt.figure()
st.plot(fig=fig)
fig.show()

# Stack function
fig, ax = plt.subplots()
stack_array.sel(y=y_max, x=x_max, celerity=c_max).plot(ax=ax)
fig.show()
