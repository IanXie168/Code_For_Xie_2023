# Make sure the environment is good
import metpy
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import numpy as np
import pandas as pd 
from minisom import MiniSom
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt 
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from wrf import (getvar, ALL_TIMES)
from matplotlib.colors import LinearSegmentedColormap
import cartopy.io.shapereader as shapereader
import cartopy.io.shapereader as shapereader
from wrf import get_cartopy
import cmaps
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import shapely.geometry as sgeom
from copy import copy
import warnings
warnings.filterwarnings("ignore")


## *************************************************************************** ##
## *************************************************************************** ##
def read_ERA5_data():
    """ This function aims to load the ERA5 reanalysis data, including precipitation, temperature, winds at 700 hPa, calculating the IVT """
    
    ##--- read the geopotential height data
    f = xr.open_dataset("/glade/work/zhixing/Analysis_DATA/SNOWIE_seasonal_Data/ERA5_reanalysis/hgt_700hPa_201701-03.nc")
    lon = f.longitude; lat = f.latitude; time = f.time; hgt_700 = f.z / 9.8
    # ind_n = 0; ind_s = -20; ind_w = 20;  ind_e = -20
    # [:, ind_n:ind_s, ind_w:ind_e]
    map_data_700 = hgt_700[:,:,:]    
    dim0 = map_data_700.shape[0]; dim1 = map_data_700.shape[1]; dim2 = map_data_700.shape[2]
    tod_map_hgt_700 = map_data_700.values.reshape(dim0, dim1*dim2)

    ##--- read the precipitation data
    f = xr.open_dataset("/glade/work/zhixing/Analysis_DATA/SNOWIE_seasonal_Data/ERA5_reanalysis/precipitation_201701-03_1hr.nc")
    precip = f.tp[:,:,:]; 

    ##--- read the temperature data
    f = xr.open_dataset("/glade/work/zhixing/Analysis_DATA/SNOWIE_seasonal_Data/ERA5_reanalysis/Temperature_201701-03_1hr.nc")
    temp_700 = f.t[:, 1,:,:]; 

    ##--- read the horizontal wind/ meridional wind data
    f = xr.open_dataset("/glade/work/zhixing/Analysis_DATA/SNOWIE_seasonal_Data/ERA5_reanalysis/Winds_201701-03_1hr.nc")
    uwind_700 = f.u[:,6,:,:];  vwind_700 = f.v[:,6,:,:];
    
    ##--- calculate the integrated water vapor flux
    file_qv = "/glade/work/zhixing/Analysis_DATA/SNOWIE_seasonal_Data/ERA5_reanalysis/cal_IVT_q_201701-03.nc"
    file_ua = "/glade/work/zhixing/Analysis_DATA/SNOWIE_seasonal_Data/ERA5_reanalysis/cal_IVT_ua_201701-03.nc"
    file_va = "/glade/work/zhixing/Analysis_DATA/SNOWIE_seasonal_Data/ERA5_reanalysis/cal_IVT_va_201701-03.nc"

    ds_qv = xr.open_dataset(file_qv)
    ds_ua = xr.open_dataset(file_ua)
    ds_va = xr.open_dataset(file_va)

    p        = ds_qv.level.values * 100
    qv       = ds_qv.q.values # time 2160; levels 20; latitude 101; longitude 141
    ua       = ds_ua.u.values
    va       = ds_va.v.values

    delta_p_list = []; layer_ua_list = []; layer_va_list = []; layer_shum_list = []
    for i in range(p.shape[0]-1):
        bottom_p = p[i]; bottom_ua = ua[:,i]; bottom_va = va[:,i]; bottom_qv = qv[:,i]
        upper_p  = p[i+1]; upper_ua = ua[:,i+1]; upper_va = va[:,i+1]; upper_qv = qv[:,i+1]
        delta_p  = -(bottom_p - upper_p); layer_ua = (bottom_ua + upper_ua)/2; 
        layer_va = (bottom_va + upper_va)/2; layer_qv = (bottom_qv + upper_qv)/2
        layer_shum = layer_qv / (layer_qv + 1)
        delta_p_list.append(delta_p); layer_ua_list.append(layer_ua); layer_va_list.append(layer_va); layer_shum_list.append(layer_shum)

    ## Equation: (layer_shum*layer_ua + layer_shum*layer_va) * delta_p / g
    MFT_list_u = []; MFT_list_v = []; MFT_list = []
    for i,j,k,l in zip(layer_shum_list, layer_ua_list, layer_va_list, delta_p_list):
        g   = 9.8

        MFT_u = i * j * l / g
        MFT_v = i * k * l / g
        MFT   = np.sqrt(MFT_u ** 2 + MFT_v ** 2)
        MFT_list_u.append(MFT_u); MFT_list_v.append(MFT_v); MFT_list.append(MFT)

    MFT_array_u = np.array(MFT_list_u); MFT_array_v = np.array(MFT_list_v); MFT_array = np.array(MFT_list)
    IVMFT_u     = np.sum(MFT_array_u, axis = 0); IVMFT_v     = np.sum(MFT_array_v, axis = 0); IVMFT     = np.sum(MFT_array, axis = 0)

    
    return map_data_700, tod_map_hgt_700, precip, uwind_700, vwind_700, temp_700, IVMFT_u, IVMFT_v, lat, lon

## *************************************************************************** ##
## *************************************************************************** ##

def extract_wrf_rainnc_data(variable):
    """ This function aims to load the hourly precip and snowfall data """
    
    if variable == 'precipitation':
        data_file    = xr.open_dataset("/glade/work/zhixing/Analysis_DATA/SNOWIE_seasonal_Data/vars-merge/RAINNC_snowie_Timely.nc")
        temp = data_file.variables['RAINNC_TIMELY'].data # Units: mm
    # if you want to extract the snow data
    if variable == 'snow':
        data_file    = xr.open_dataset("/glade/work/zhixing/Analysis_DATA/SNOWIE_seasonal_Data/vars-merge/SNOWNC_Timely.nc")
        temp = data_file.variables['SNOWNC_TIMELY'].data # Units: mm
    
    lons = data_file.variables['XLONG'].data; 
    lats = data_file.variables['XLAT'].data; 
    time = np.array(data_file.variables['Time'].data, dtype='datetime64[s]')
    
    ## investigation period (from January to March)
    Jan   = np.datetime64('2017-01-01', 'ns')
    Mar   = np.datetime64('2017-04-01', 'ns')
    time_ind  = np.where((time>=Jan) & (time<Mar))
    time  = time[time_ind]
    temp  = temp[time_ind] ## temp variables are rainnc or snownc, it depends on "the input variable"
    
    return temp, time, lons, lats


## *************************************************************************** ##
## *************************************************************************** ##

def filter_data(variable, area, threshold):
    """ This function aims to filter the data in a specific region (i.e. in Payette region) or by a threshold (i.e. regional precip/snow >= certain value) """
    
    if area == 'Payette':
        ind_lon = np.where((lons[-1]>=-117) & (lons[-1]<=-114)) ## lons_W = -117, lons_E = -114
        ind_lat = np.where((lats[:,0]>=43) & (lats[:,0]<=45)) ## lats_S = 43, lats_N = 45, select a rectangle region of interest
        ind_lat_s = ind_lat[0][0]; ind_lat_e = ind_lat[0][-1]+1; ind_lon_s = ind_lon[0][0]; ind_lon_e = ind_lon[0][-1]+1 
        Payette_lons      = lons[ind_lat_s:ind_lat_e, ind_lon_s:ind_lon_e]
        Payette_lats      = lats[ind_lat_s:ind_lat_e, ind_lon_s:ind_lon_e]
    
        if variable == 'precipitation':
            payette_rain  = rain[:,ind_lat_s:ind_lat_e, ind_lon_s:ind_lon_e]
            area_mean = np.nanmean(payette_rain, axis=(1,2))
            ind  = np.where(area_mean>threshold)
            filter_payette_temp = payette_rain[ind]
            print("Payette region RAINNC shape:", payette_rain.shape, "\nPayette region filtered RAINNC shape:", filter_payette_temp.shape)

        if variable == 'snow':
            payette_snow  = snow[:,ind_lat_s:ind_lat_e, ind_lon_s:ind_lon_e]
            area_mean = np.nanmean(payette_snow, axis=(1,2))
            ind  = np.where(area_mean>threshold)
            filter_payette_temp = payette_snow[ind]
            print("Payette region SNOWNC shape:", payette_snow.shape, "\nPayette region filtered SNOWNC shape:", filter_payette_temp.shape)
        
        return Payette_lons, Payette_lats, filter_payette_temp
    
    else:
        
        if variable == 'precipitation':
            area_mean = np.nanmean(rain, axis=(1,2))
            ind  = np.where(area_mean>threshold)
            filter_temp = rain[ind]
            print("Whole region RAINNC shape:", rain.shape, "\nWhole region filtered RAINNC shape:", filter_temp.shape)

        if variable == 'snow':
            area_mean = np.nanmean(snow, axis=(1,2))
            ind  = np.where(area_mean>threshold)
            filter_temp = snow[ind]
            print("Whole region SNOWNC shape:", snow.shape, "\nWhole region filtered SNOWNC shape:", filter_temp.shape)
            
        return filter_temp

## *************************************************************************** ##
## *************************************************************************** ##

def SOM_processing(NODE1, NODE2, sigma, l_r, n_f, r_s, iterations, max_iter):
    """ Apply the self-organinzg map algorithm, return the clusters indexes (which maps projected in which node) """
    
    # Before applying the algorithm, multiple tests should be conducted to determine how many nodes to use, the neighborhood radius, learning rate, etc...
    # And pick the "best combination" of parameters 
    
    # Apply Self-Organizing Map
    som       = MiniSom(NODE1, NODE2, tod_map_hgt_700.shape[1], sigma=sigma,\
        learning_rate= l_r, neighborhood_function=n_f, random_seed=r_s)
    
    for i in range(max_iter):
        rand_i = np.random.randint(len(tod_map_hgt_700))
        som.update(tod_map_hgt_700[rand_i], som.winner(tod_map_hgt_700[rand_i]), i, iterations)
    w_x, w_y  = zip(*[som.winner(d) for d in tod_map_hgt_700])
    w_x       = np.array(w_x); w_y       = np.array(w_y)
    qe        = som.quantization_error(tod_map_hgt_700); te      = som.topographic_error(tod_map_hgt_700)
    print("quantization error:", qe)
    print("topography error:", te)
    print("SOM\t Finished...")
    
    return w_x, w_y

## *************************************************************************** ##
## *************************************************************************** ##

def Link_processing(NODE1, NODE2, w_x, w_y, time):
    """ This function aims to return a indexical dictionary that documenting the indexes of each map projected into certain node """
    
    # Link data with clusters
    dic_cluster = {}
    dic_time    = {}
    for a in range(NODE1):
        for b in range(NODE2):
            temp = list([])
            temp_time = list([])
            for i, j, k in zip(w_x, w_y, range(w_x.shape[0])):
                position = (i, j)
                if position == (a, b):
                    temp.append(k)
                    temp_time.append(time[k])
                    dic_cluster[position] = temp
                    dic_time[position]    = temp_time
    print("Link data with cluster\t Finished...\n")
    return dic_cluster, dic_time


def Count_processing(NODE1, NODE2, dic_cluster):
    """ Counting the numbers of maps projected in certain node """
    
    # Count number in each node.
    for i in range(NODE1):
        for j in range(NODE2):
            loc = (i, j)
            print("position: ", loc, "\t", len(dic_cluster[loc]))

def get_filter_dic_cluster(var, dic_cluster, threshold):
    """ get the filtered indexical dictionary, via regoinal averaged precip >= a certain value """
    
    filter_dic_cluster = {}
    for loc, index_list in dic_cluster.items():
        temp_list = []
        for i in index_list:
            if var[i].mean() < threshold: ## threshold
                continue
            else:
                temp_list.append(i)
        filter_dic_cluster[loc] = temp_list
    print("get filter dic_cluster\t Finished...\n")
    return filter_dic_cluster




def Plot_ERA5_data_preprocessing(dic_cluster, geo_ht, temp, ua, va, ivt_u, ivt_v, precip):
    """ This function aims to get the composite 2d maps for each node, including geopotential heights, temperature, winds at 700hPa and IVT, precip maps """
    
    # Averaged maps on each node
    dic_hgt_700_ERA5_map = {};  dic_pcp_ERA5_map = {}; dic_temp_700_ERA5_map = {}; dic_u_700_ERA5_map = {}; dic_v_700_ERA5_map = {}
    dic_IVMFT_u_ERA5_map = {};  dic_IVMFT_v_ERA5_map  = {}; 
    for loc, index_list in dic_cluster.items():
        dic_hgt_700_ERA5_map[loc]  = np.mean(geo_ht[index_list], axis=0)
        dic_temp_700_ERA5_map[loc] = np.mean(temp[index_list], axis=0)
        dic_u_700_ERA5_map[loc]    = np.mean(ua[index_list], axis=0)
        dic_v_700_ERA5_map[loc]    = np.mean(va[index_list], axis=0)
        dic_IVMFT_u_ERA5_map[loc]  = np.mean(ivt_u[index_list], axis=0)
        dic_IVMFT_v_ERA5_map[loc]  = np.mean(ivt_v[index_list], axis=0)
        dic_pcp_ERA5_map[loc] = np.mean(precip[index_list], axis=0)

        
    return dic_hgt_700_ERA5_map, dic_temp_700_ERA5_map, dic_u_700_ERA5_map,\
dic_v_700_ERA5_map, dic_IVMFT_u_ERA5_map, dic_IVMFT_v_ERA5_map, dic_pcp_ERA5_map




def plot_synoptic_circulation(NODE1, NODE2, dic_hgt_700_ERA5_map, dic_temp_700_ERA5_map, dic_IVMFT_u_ERA5_map, dic_IVMFT_v_ERA5_map,lon_era5,lat_era5,dic_cluster):
    """ This function aims to plot the composite synoptic circulation maps in each node """
    
    cMap = []
    for value, colour in zip([0,22,24.5,25,26,31],["Blue","lightskyblue","white","white","salmon","red"]):
        ## self-defined color bar
        cMap.append((value/31, colour))
    customColourMap2 = LinearSegmentedColormap.from_list("custom", cMap)


    # Use Integrated Water Vapor Flux instead of the winds in 700 hPa
    xx, yy = np.meshgrid(lon_era5,lat_era5)
    # xx, yy = xx[ind_n:ind_s,ind_w:ind_e], yy[ind_n:ind_s,ind_w:ind_e]
    # Data projection; NARR Data is Earth Relative
    dataproj  = ccrs.PlateCarree()
    plotproj  = ccrs.PlateCarree()

    fig, axs = plt.subplots(NODE1,NODE2,figsize=(16, 12), subplot_kw = {'projection':plotproj}, sharex=True, sharey=True)

    for state in shapereader.Reader("/glade/u/home/zhixing/ShapeFile/admin_1_province_country_10m/ne_10m_admin_1_states_provinces.shp").records():
        if state.attributes['name_id'] == 'Idaho': 
            Idaho = state.geometry
            print("Idaho State Found!")
            break
    else:
        raise ValueError('Unable to find the Idaho boundary.')
    count = 0
    title_list = ['CZF',  'CSWF', 'WZF',  'WSWF']
    for i in range(NODE1):
        for j in range(NODE2):
            axs[i,j].coastlines('50m', linewidth=0.75)
            axs[i,j].add_feature(cfeature.STATES, linewidth=0.5)
            axs[i,j].set_extent([-140, -110, 30, 55],ccrs.PlateCarree())
            axs[i,j].add_geometries([Idaho], ccrs.Geodetic(), edgecolor='k',
                              facecolor='none', linewidth=7, alpha = 0.6)

            ## plot the temperature contour
            data = dic_temp_700_ERA5_map[(i,j)] - 273.15 ## K convert into degree celcius
            lvls = np.arange(-24, 8) 
            #customColourMap; cmaps.MPL_coolwarm
            cf_tem  = axs[i,j].contourf(xx, yy, data, transform=dataproj, cmap=customColourMap2, levels = lvls)

            ## plot the IVMFT fields
            u_data = dic_IVMFT_u_ERA5_map[(i,j)]
            v_data = dic_IVMFT_v_ERA5_map[(i,j)]
            wind_slice = (slice(None, None, 5), slice(None, None, 5))
            wind_barb = axs[i,j].quiver(xx[wind_slice], yy[wind_slice],u_data[wind_slice],\
                                v_data[wind_slice], color='black', transform=dataproj, scale_units='inches', scale = 300, headwidth=2.5 )
            if i==0 and j==1:
                qk = axs[i,j].quiverkey(wind_barb, 0.62, 1.03, 100, r'IWVFT 100 kg/(m*s)', labelpos='E',
                                   coordinates='axes', fontproperties={'size':'x-large'})

            ## geopotential height at 700 mb
            data = dic_hgt_700_ERA5_map[(i,j)]
            cl_level  = axs[i,j].contour(xx, yy, data, colors="gray", transform = dataproj, linewidths = 3, inline=False) 
            axs[i,j].clabel(cl_level, fmt="%i", fontsize = 'x-large', colors = 'k')
            axs[i,j].set_title(title_list[count], y=1.05, fontsize = 'xx-large')
            count += 1
            # axs[i,j].set_title("Node " + str(count), fontsize = 'x-large')
            counts = len(dic_cluster[(i,j)]); fraction = round(counts*100 / 2160, 1)  
            axs[i,j].set_title(str(fraction) + "%", fontsize = 'x-large', loc = 'left')
            axs[i,j].set_xticks([-140, -130, -120, -110], crs=ccrs.PlateCarree())
            axs[i,j].set_yticks([30, 35, 40, 45, 50, 55], crs=ccrs.PlateCarree())
            lon_formatter = LongitudeFormatter(number_format='.0f',
                                               dateline_direction_label=True)
            lat_formatter = LatitudeFormatter(number_format='.0f',
                                              degree_symbol='°')
            axs[i,j].xaxis.set_major_formatter(lon_formatter)
            axs[i,j].yaxis.set_major_formatter(lat_formatter)
            axs[i,j].xaxis.set_minor_locator(ticker.AutoMinorLocator())
            axs[i,j].yaxis.set_minor_locator(MultipleLocator(1))
            axs[i,j].tick_params(which='major',labelsize='xx-large', width=2, length=5,  pad=2)
            axs[i,j].tick_params(which='minor',labelsize='small', width=1, length=3,  pad=1)

    #--- add figure color bar
    cax  = plt.axes([0.98, 0.2, 0.015, 0.6]) 
    cbar = plt.colorbar(cf_tem, cax=cax,shrink=0.8,ticklocation='right')
    cbar.ax.tick_params(labelsize='xx-large') 
    cbar.set_label(label='Temperature (\u00b0C)', weight='bold', fontsize='xx-large')
    fig.tight_layout()
    
    
    
###-------------- below is the funciton for plotting ticks/ticklabels in lambert project
def find_side(ls, side):
    """
    Given a shapely LineString which is assumed to be rectangular, return the
    line corresponding to a given side of the rectangle.

    """
    # print(ls)
    # print(ls.bounds)
    minx, miny, maxx, maxy = ls.bounds
    points = {'left': [(minx, miny), (minx, maxy)],
              'right': [(maxx, miny), (maxx, maxy)],
              'bottom': [(minx, miny), (maxx, miny)],
              'top': [(minx, maxy), (maxx, maxy)],}
    return sgeom.LineString(points[side])

def lambert_xticks(ax, ticks):
    """Draw ticks on the bottom x-axis of a Lambert Conformal projection."""
    te = lambda xy: xy[0]
    lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2], b[3], n))).T
    xticks, xticklabels = _lambert_ticks(ax, ticks, 'bottom', lc, te)
    ax.xaxis.tick_bottom()
    ax.set_xticks(xticks)
    ax.set_xticklabels([ax.xaxis.get_major_formatter()(xtick) for xtick in xticklabels])

def lambert_yticks(ax, ticks):
    """Draw ticks on the left y-axis of a Lamber Conformal projection."""
    te = lambda xy: xy[1]
    lc = lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
    yticks, yticklabels = _lambert_ticks(ax, ticks, 'left', lc, te)
    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    ax.set_yticklabels([ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels])
    
def lambert_yticks_right(ax, ticks):
    """Draw ticks on the left y-axis of a Lamber Conformal projection."""
    te = lambda xy: xy[1]
    lc = lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
    yticks, yticklabels = _lambert_ticks(ax, ticks, 'left', lc, te)
    ax.yaxis.tick_right()
    ax.set_yticks(yticks)
    ax.set_yticklabels([ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels])

def _lambert_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """Get the tick locations and labels for an axis of a Lambert Conformal projection."""
    outline_patch = sgeom.LineString(ax.spines['geo'].get_path().vertices.tolist())
    axis = find_side(outline_patch, tick_location)
    n_steps = 30
    extent = ax.get_extent(ccrs.PlateCarree())
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs:
            tick = [None]
        else:
            tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])
    # Remove ticks that aren't visible:    
    ticklabels = copy(ticks)
    while True:
        try:
            index = _ticks.index(None)
        except ValueError:
            break
        _ticks.pop(index)
        ticklabels.pop(index)
    return _ticks, ticklabels