#!/usr/bin/env python

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# ** Copyright UCAR (c) 2015
# ** University Corporation for Atmospheric Research(UCAR)
# ** National Center for Atmospheric Research(NCAR)
# ** Research Applications Laboratory(RAL)
# ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
# ** 2016/08/22
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

'''
 Name:         *.py
 Author:       Kevin Sampson
               Associate Scientist
               National Center for Atmospheric Research
               ksampson@ucar.edu

 Created:      08/11/2015
 Modified:     08/22/2016
'''

'''
12/17/2015:

This script is labeled "low memory" because of a few key distinctions. First,
it saves the dictionary created by each node to a cPickle file. Then it constructs
the output correspondence netCDF file by reading each cPickle file individually,
only holding one core's worth of output at a time.


This script code is intended to generate regridding weights between a
netCDF file (ideally a GEOGRID file) or a high resolution grid in netCDF but
with a GEOGRID file to define the coordinate reference system. The process takes
the following steps:

    1) Use GEOGRID file to define the input coordinate reference system
    2) Save the NetCDF grid to in-memory raster format (optionally save to disk)
    3) Build Gridder object to speed up the spatial intersection process (must be a Cartesian grid!!)
    4) Calculate the spatial intersection between the input grid and the input polygons
    5) Export weights to netCDF.
    6) Optionally output a vector polygon mesh of the analysis grid, in GeoPackage format

03/06/2018

Renamed to grid2poly.py from WRF_Hydro_Regridding_MP_pickle_low_memory_20161104.py
Moved some user specified parameters toward the top
    '''

ALP = False                                                                     # If running on Windows machine ALP, must start python VirtualEnv

# Import Python Core Modules
import sys
import os
import time
import multiprocessing
import math                                                                     # Only needed for producing coordinate system information
import pickle

# Import Additionale Modules
from osgeo import ogr
from osgeo import osr
from osgeo import gdal
from osgeo import gdalconst
import netCDF4
import numpy

# Module settings
tic1 = time.time()
sys.dont_write_bytecode = True
gdal.UseExceptions()                                                            # this allows GDAL to throw Python Exceptions
gdal.PushErrorHandler('CPLQuietErrorHandler')
multiprocessing.freeze_support()

# ---------- Specifications by User ----- #

# Files and Directories
OutDir          = './weight_file'         # Output Directory
GRID_DIR        = './geospatial'          # Directory containing grid
GRID_NAME       = 'nldas_conus12k_EPSG4326.tif'      # grid name (this needs to be geotiff)
BASIN_DIR       = './geospatial'          # Directory contain basin polygon data
#BASIN_POLY      = 'conus_HUC12_merit_v7b.gpkg'  # Basin polygon data name: HUC12: CONUS_HUC12_ext_v5_merit_fixed_loca2_overlapped.gpkg
#BASIN_POLY      = 'HCDN_nhru_final_671.buff_fix_holes.CAMELSandTDX_areabias_fix.simp0.001.level1.gpkg'
BASIN_POLY      = 'gagesII_671_shp_geogr.gpkg'
#BASIN_ID_NAME   = 'HUCIDXint'          # polyg ID name (attribute field name) in polygon geometry data HUC12: HUCIDXint
BASIN_ID_NAME   = 'GAGE_ID'          # polyg ID name (attribute field name) in camels polygon geometry data
MAPPING_NC_NAME = 'spatialweights_nldas12km_to_camels.nc'          # mapping netCDF name (this is output)
OUT_GRID_POLY   = 'N/A'          # optional: name of GRID polygon
InDir  = 'Not Used'            # Input Directory

# Configurations
inDriverName = 'GPKG'                                        # currently working with only GPKG (Not successful if ESRI Shapefile used)
outDriverName = 'ESRI Shapefile'                             # if OutputGridPolys is true, grid polygon is generated with this format
ticker = 100                                                # Ticker for how often to print a message
SaveRaster = False                                           # Save the NetCDF grid as a GeoTiff file
OutputGridPolys = False                                      # Save output vector grid polygons
threshold = 1000000000000                                    # Number of square meters above which a polygon gets split up
splits = 100                                                  # Number of chunks to divide the large polygon into
NC_format = 'NETCDF4'                                        # NetCDF output format. Others: 'NETCDF4_CLASSIC'

# Multiprocessing parameters
CPU_Count = multiprocessing.cpu_count()                      # Find the number of CPUs to chunk the data into
Processors = 24                                             # To spread the processing over a set number of cores, otherwise use Processors = CPU_Count-1
pool_size = Processors*50                                  # Having a large number of worker processes can help with overly slow cores

# ---------- end of user sepcification -------#


# ---------- Setting up ----------------------#

# Create output directory if not exist
if not os.path.exists(OutDir):
  os.makedirs(OutDir)

# Input vector layer (basins, etc)
in_basins = os.path.join(BASIN_DIR, BASIN_POLY)
fieldname = BASIN_ID_NAME                                    # Unique identifier field

# Input raster grid
in_raster = os.path.join(GRID_DIR, GRID_NAME)

# Output polygon vector file
if OutputGridPolys == True:
    OutGridFile = os.path.join(OutDir, OUT_GRID_POLY)        # grid polygon name

# Output Weights file
regridweightnc = os.path.join(OutDir, MAPPING_NC_NAME)

# Output GeoTiff of the geogrid file
if SaveRaster == True:
    OutGTiff = os.path.join(OutDir, os.path.basename(in_geogrid).replace('.nc', '_' + geogridVariable + '.tif'))

# ---------- End of setting up ----------------#


# ---------- Classes ---------- #
class Gridder_Layer(object):
    '''Class with which to create the grid intersecting grid cells based on a feature
    geometry envelope. Provide grid information to initiate the class, and use getgrid()
    to generate a grid mesh and index information about the intersecting cells.

    Note:  The i,j index begins with (1,1) in the Lower Left corner.'''
    def __init__(self, DX, DY, x00, y00, nrows, ncols):
        self.DX = DX
        self.DY = DY
        self.x00 = x00
        self.y00 = y00
        self.nrows = nrows
        self.ncols = ncols

    def getgrid(self, envelope, layer):
        """Gridder.getgrid() takes as input an OGR geometry envelope, and will
        compute the grid polygons that intersect the evelope, returning a list
        of grid cell polygons along with other attribute information."""
        # Calculate the number of grid cells necessary
        xmin, xmax, ymin, ymax = envelope

        # Find the i and j indices
        i0 = int((xmin-self.x00)/self.DX // 1)                                  # Floor the value
        j0 = int(abs((ymax-self.y00)/self.DY) // 1)                             # Floor the absolute value
        i1 = int((xmax-self.x00)/self.DX // 1)                                  # Floor the value
        j1 = int(abs((ymin-self.y00)/self.DY) // 1)                             # Floor the absolute value

        # Create a new field on a layer. Add one attribute
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn('cellsize', ogr.OFTReal))
        layer.CreateField(ogr.FieldDefn('i_index', ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn('j_index', ogr.OFTInteger))
        LayerDef = layer.GetLayerDefn()                                         # Fetch the schema information for this layer

        # Build OGR polygon objects for each grid cell in the intersecting envelope
        for x in range(i0, i1+1):
            if x < 0 or x >= self.ncols:
                continue
            for y in reversed(range(j0, j1+1)):
                if y < 0 or y >= self.nrows:
                    continue
                id1 = (self.nrows*(x+1))-y                                      # This should give the ID of the cell from the lower left corner (1,1)

                # Calculating each grid cell polygon's coordinates
                x0 = self.x00 + (self.DX*x)
                x1 = x0 + self.DX
                y1 = self.y00 - (abs(self.DY)*y)
                y0 = y1 - abs(self.DY)

                # Create ORG geometry polygon object using a ring
                myRing = ogr.Geometry(type=ogr.wkbLinearRing)
                myRing.AddPoint(x0, y1)
                myRing.AddPoint(x1, y1)
                myRing.AddPoint(x1, y0)
                myRing.AddPoint(x0, y0)
                myRing.AddPoint(x0, y1)
                geometry = ogr.Geometry(type=ogr.wkbPolygon)
                geometry.AddGeometry(myRing)

                # Create the feature
                feature = ogr.Feature(LayerDef)                                     # Create a new feature (attribute and geometry)
                feature.SetField('id', id1)
                feature.SetField('cellsize', geometry.Area())
                feature.SetField('i_index', x+1)
                feature.SetField('j_index', self.nrows-y)
                feature.SetGeometry(geometry)                                      # Make a feature from geometry object
                layer.CreateFeature(feature)
                geometry = feature = None
                del x0, x1, y1, y0, id1
        return layer
# ---------- End Classes ---------- #

# ---------- Functions ---------- #
def checkfield(layer, fieldname, string1):
    '''Check for existence of provided fieldnames'''
    layerDefinition = layer.GetLayerDefn()
    fieldslist = []
    for i in range(layerDefinition.GetFieldCount()):
        print(layerDefinition.GetFieldDefn(i).GetName())
        fieldslist.append(layerDefinition.GetFieldDefn(i).GetName())
    if fieldname in fieldslist:
        i = fieldslist.index(fieldname)
        field_defn = layerDefinition.GetFieldDefn(i)
    else:
        print('    Field %s not found in input %s. Terminating...' %(fieldname, string1))
        raise SystemExit
    return field_defn, fieldslist

def getfieldinfo(field_defn, fieldname):
    '''Get information about field type for buildng the output NetCDF file later'''
    if field_defn.GetType() == ogr.OFTInteger:
        fieldtype = 'integer'
        print("found ID type of Integer")
    elif field_defn.GetType() == ogr.OFTInteger64:
        fieldtype = 'integer64'
        print("found ID type of Integer64")
    elif field_defn.GetType() == ogr.OFTReal:
        fieldtype = 'real'
        print("field type: OFTReal not currently supported in output NetCDF file.")
        raise SystemExit
    elif field_defn.GetType() == ogr.OFTString:
        fieldtype = 'string'
        print("found ID type of String")
    else:
        print("ID Type not found ... Exiting")
        raise SystemExit
    print("    Field Type for field '%s': %s (%s)" %(fieldname, field_defn.GetType(), fieldtype))
    return fieldtype

def loadPickle(fp):
    with open(fp, 'rb') as fh:
        listOfObj = pickle.load(fh)
    return listOfObj

def Read_GEOGRID_for_SRS(in_nc, Variable):
    '''Read NetCDF GEOGRID file as a GDAL raster object. Much of the code below was borrowed
    from https://github.com/rveciana/geoexamples/blob/master/python/wrf-NetCDF/read_netcdf.py'''

    tic = time.time()
    try:
        print('Input netCDF GEOGRID file: %s    Variable: %s' %(in_nc, Variable))
        ds_in = gdal.Open(in_nc)                                                # Open input netCDF file using GDAL
        subdatasets = ds_in.GetSubDatasets()                                    # Gather subdatasets from input netCDF file
        variables = [subdataset[1].split(" ")[1] for subdataset in subdatasets] # Gather variables in the input netCDF file
        print('Variables found in input NC file: %s' %(variables))
        if Variable in variables:
            src_ds = gdal.Open('NETCDF:"'+in_nc+'":%s' %(Variable))             # Open using NETCDF driver, file name, and variable
            metadata = ds_in.GetMetadata()                                      # Read metadata
            srcband = src_ds.GetRasterBand(1)                                   # Get raster band
            ncvar = srcband.ReadAsArray()                                       # Read variable as a numpy array
            ds_in = subdatasets = None

            # Initiate dictionaries of GEOGRID projections and parameters
            projdict = {1: 'Lambert Conformal Conic', 2: 'Polar Stereographic', 3: 'Mercator', 6: 'Cylindrical Equidistant'}

            # Read metadata for grid information
            map_pro = int(metadata['NC_GLOBAL#MAP_PROJ'])
            DX = float(metadata['NC_GLOBAL#DX'])
            DY = float(metadata['NC_GLOBAL#DY'])
            corner_lats = metadata['NC_GLOBAL#corner_lats']
            corner_lons = metadata['NC_GLOBAL#corner_lons']

            # Gather corner information [order = (LL center, ULcenter, URcenter, LRcenter, LLLedge, ULLedge, URRedge, LRRedge, LLBedge, ULUedge, URUedge, LRBedge, LLcorner, ULcorner, URcorner, LRcorner)
            corner_latslist = corner_lats.strip('{}').split(',')                # Create list of strings from the corner_lats attribute
            corner_lonslist = corner_lons.strip('{}').split(',')                # Create list of strings from the corner_lons attribute
            ##LLcenter = [corner_lonslist[0], corner_latslist[0]]               # Lower left of the Mass grid staggering
            #ULcenter = [corner_lonslist[1], corner_latslist[1]]               # Upper left of the Mass grid staggering
            ##URcenter = [corner_lonslist[2], corner_latslist[2]]               # Upper right of the Mass grid staggering
            ##LRcenter = [corner_lonslist[3], corner_latslist[3]]               # Lower right of the Mass grid staggering
            ##LLcorner = [corner_lonslist[12], corner_latslist[12]]             # Lower left of the Unstaggered grid
            ULcorner = [corner_lonslist[13], corner_latslist[13]]               # Upper left of the Unstaggered grid
            ##URcorner = [corner_lonslist[14], corner_latslist[14]]             # Upper right of the Unstaggered grid
            ##LRcorner = [corner_lonslist[15], corner_latslist[15]]             # Lower right of the Unstaggered grid

            # Pick a corner or center point to 'hang' the raster from
            lon = float(ULcorner[0])                                            #lon = float(ULcenter[0])
            lat = float(ULcorner[1])                                            #lat = float(ULcenter[1])

            # Read metadata for projection parameters
            if 'NC_GLOBAL#TRUELAT1' in metadata.keys():
                standard_parallel_1 = float(metadata['NC_GLOBAL#TRUELAT1'])
            if 'NC_GLOBAL#TRUELAT2' in metadata.keys():
                standard_parallel_2 = float(metadata['NC_GLOBAL#TRUELAT2'])
            if 'NC_GLOBAL#STAND_LON' in metadata.keys():
                central_meridian = float(metadata['NC_GLOBAL#STAND_LON'])
            if 'NC_GLOBAL#POLE_LAT' in metadata.keys():
                pole_latitude = float(metadata['NC_GLOBAL#POLE_LAT'])
            if 'NC_GLOBAL#POLE_LON' in metadata.keys():
                pole_longitude = float(metadata['NC_GLOBAL#POLE_LON'])
            if 'NC_GLOBAL#CEN_LAT' in metadata.keys():
                latitude_of_origin = float(metadata['NC_GLOBAL#CEN_LAT'])

            # Initiate OSR spatial reference object - See http://gdal.org/java/org/gdal/osr/SpatialReference.html
            proj1 = osr.SpatialReference()
            proj1.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

            # ---- NOTE: Below is experimental & untested ---- #

            # Use projection information from global attributes to populate OSR spatial reference object
            # See this website for more information on defining coordinate systems: http://gdal.org/java/org/gdal/osr/SpatialReference.html
            print('    Map Projection: %s' %projdict[int(metadata['NC_GLOBAL#MAP_PROJ'])])
            if map_pro == 1:
                # Lambert Conformal Conic
                if 'standard_parallel_2' in locals():
                    proj1.SetLCC(standard_parallel_1, standard_parallel_2, latitude_of_origin, central_meridian, 0, 0)
                    #proj1.SetLCC(double stdp1, double stdp2, double clat, double clong, double fe, double fn)        # fe = False Easting, fn = False Northing
                else:
                    proj1.SetLCC1SP(latitude_of_origin, central_meridian, 1, 0, 0)       # Scale = 1???
                    #proj1.SetLCC1SP(double clat, double clong, double scale, double fe, double fn)       # 1 standard parallell
            elif map_pro == 2:
                # Polar Stereographic
                phi1 = float(standard_parallel_1)                               # Set up pole latitude
                ### Back out the central_scale_factor (minimum scale factor?) using formula below using Snyder 1987 p.157 (USGS Paper 1395)
                ##phi = math.copysign(float(pole_latitude), float(latitude_of_origin))    # Get the sign right for the pole using sign of CEN_LAT (latitude_of_origin)
                ##central_scale_factor = (1 + (math.sin(math.radians(phi1))*math.sin(math.radians(phi))) + (math.cos(math.radians(float(phi1)))*math.cos(math.radians(phi))))/2
                # Method where central scale factor is k0, Derivation from C. Rollins 2011, equation 1: http://earth-info.nga.mil/GandG/coordsys/polar_stereographic/Polar_Stereo_phi1_from_k0_memo.pdf
                # Using Rollins 2011 to perform central scale factor calculations. For a sphere, the equation collapses to be much  more compact (e=0, k90=1)
                central_scale_factor = (1 + math.sin(math.radians(abs(phi1))))/2        # Equation for k0, assumes k90 = 1, e=0. This is a sphere, so no flattening
                print('        Central Scale Factor: %s' %central_scale_factor)
                #proj1.SetPS(latitude_of_origin, central_meridian, central_scale_factor, 0, 0)    # example: proj1.SetPS(90, -1.5, 1, 0, 0)
                proj1.SetPS(pole_latitude, central_meridian, central_scale_factor, 0, 0)    # Adjusted 8/7/2017 based on changes made 4/4/2017 as a result of Monaghan's polar sterographic domain. Example: proj1.SetPS(90, -1.5, 1, 0, 0)
                #proj1.SetPS(double clat, double clong, double scale, double fe, double fn)
            elif map_pro == 3:
                # Mercator Projection
                proj1.SetMercator(latitude_of_origin, central_meridian, 1, 0, 0)     # Scale = 1???
                #proj1.SetMercator(double clat, double clong, double scale, double fe, double fn)
            elif map_pro == 6:
                # Cylindrical Equidistant (or Rotated Pole)
                if pole_latitude != float(90) or pole_longitude != float(0):
                    # if pole_latitude, pole_longitude, or stand_lon are changed from thier default values, the pole is 'rotated'.
                    print('[PROBLEM!] Cylindrical Equidistant projection with a rotated pole is not currently supported.')
                    raise SystemExit
                else:
                    proj1.SetEquirectangular(latitude_of_origin, central_meridian, 0, 0)
                    #proj1.SetEquirectangular(double clat, double clong, double fe, double fn)
                    #proj1.SetEquirectangular2(double clat, double clong, double pseudostdparallellat, double fe, double fn)

            # Set Geographic Coordinate system (datum) for projection
            proj1.SetGeogCS('WRF-Sphere', 'Sphere', '', 6370000.0, 0.0)      # Could try 104128 (EMEP Sphere) well-known?
            #proj1.SetGeogCS(String pszGeogName, String pszDatumName, String pszSpheroidName, double dfSemiMajor, double dfInvFlattening)

            variables = src_ds = ncvar = metadata = srcband = ncvar = None
        else:
            print('Could not find variable: %s in file: %s' %(Variable, in_nc))

        # Set the origin for the output raster (in GDAL, usuall upper left corner) using projected corner coordinates
        wgs84_proj = osr.SpatialReference()
        wgs84_proj.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        wgs84_proj.ImportFromEPSG(4326)
        transform = osr.CoordinateTransformation(wgs84_proj, proj1)
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint_2D(lon, lat)
        point.Transform(transform)
        x00 = point.GetX(0)
        y00 = point.GetY(0)
        #x00 = point.GetX(0)-(DX/2)                                              # This mimicks the Arcpy method if you use the ULcenter[0]-(DX/2)
        #y00 = point.GetY(0)+abs(DY/2)                                           # This mimicks the Arcpy method if you use the ULcenter[1]+abs(DY/2)

    except RuntimeError as e:
        print('Unable to open %s' %in_nc)
        raise ImportError(e)
    print('Created projection definition from input NetCDF GEOGRID file %s in %.2fs.' %(in_nc, time.time()-tic))

    # Clear objects and return
    return proj1, DX, DY, x00, y00

def NetCDF_to_Raster(in_nc, Variable, proj_in=None, DX=1, DY=-1, x00=0, y00=0):
    '''This funciton takes in an input netCDF file, a variable name, the ouput
    raster name, and the projection definition and writes the grid to the output
    raster. This is useful, for example, if you have a FullDom netCDF file and
    the GEOGRID that defines the domain. You can output any of the FullDom variables
    to raster.'''

    tic = time.time()
    try:
        #print('        in_nc: %s    Variable: %s' %(in_nc, Variable))
        ds_in = gdal.Open(in_nc)                                                # Open input netCDF file using GDAL
        subdatasets = ds_in.GetSubDatasets()                                    # Gather subdatasets from input netCDF file
        variables = [subdataset[1].split(" ")[1] for subdataset in subdatasets] # Gather variables in the input netCDF file
        ds_in = subdatasets = None
        if Variable in variables:
            src_ds = gdal.Open('NETCDF:"'+in_nc+'":%s' %(Variable))             # Open using NETCDF driver, file name, and variable
            srcband = src_ds.GetRasterBand(1)                                   # Get raster band
            ncvar = srcband.ReadAsArray()                                       # Read variable as a numpy array

            # Set up driver for GeoTiff output
            driver = gdal.GetDriverByName('Mem')                                # Write to Memory
            if driver is None:
                print('    %s driver not available.' % 'Memory')

            # Set up the dataset and define projection/raster info
            DataSet = driver.Create('', srcband.XSize, srcband.YSize, 1, gdal.GDT_Float32)         # the '1' is for band 1.
            if proj_in is not None:
                DataSet.SetProjection(proj_in.ExportToWkt())
            if DX==1 and DY==-1:
                DataSet.SetGeoTransform((0, DX, 0, 0, 0, DY))                   # Default (top left x, w-e resolution, 0=North up, top left y, 0 = North up, n-s pixel resolution (negative value))
            else:
                DataSet.SetGeoTransform((x00, DX, 0, y00, 0, -DY))              # (top left x, w-e resolution, 0=North up, top left y, 0 = North up, n-s pixel resolution (negative value))
            DataSet.GetRasterBand(1).WriteArray(ncvar)                          # Write the array
            stats = DataSet.GetRasterBand(1).GetStatistics(0,1)                 # Calculate statistics
            #src_ds = srcband = None
            ncvar = driver = None

    except RuntimeError as e:
        print('Unable to open %s' %in_nc)
        raise ImportError(e)

    # Clear objects and return
    print('Created raster in-memory from input NetCDF file %s in %.2fs.' %(in_nc, time.time()-tic))
    return DataSet

def create_polygons_from_info(gridder_obj, proj1, outputFile, outDriverName, ticker=10000):
    '''This will take the grid info index created in Read_GEOGRID_for_SRS() and
    NetCDF_to_Raster() and produce a GeoPackage of the grid cells.'''

    print('Starting to produce polygon vector file from index')
    tic = time.time()

    # Check if files exist and delete
    if os.path.isfile(outputFile)==True:
        os.remove(outputFile)
        print('      Removed existing file: %s' %outputFile)

    # Now convert it to a vector file with OGR
    drv = ogr.GetDriverByName(outDriverName)
    if drv is None:
        print('      %s driver not available.' % outDriverName)
    else:
        print( '      %s driver is available.' % outDriverName)
        driver = ogr.GetDriverByName(outDriverName)
        datasource = driver.CreateDataSource(outputFile)
    if datasource is None:
        print('      Creation of output file failed.\n')
        raise SystemExit

    # Create output polygon vector file
    layer = datasource.CreateLayer('gridpolys', srs=proj1, geom_type=ogr.wkbPolygon)
    if layer is None:
        print('        Layer creation failed.\n')
        raise SystemExit
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('i_index', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('j_index', ogr.OFTInteger))
    LayerDef = layer.GetLayerDefn()

    point_ref=ogr.osr.SpatialReference()
    point_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    point_ref.ImportFromEPSG(4326)                                              # WGS84
    #coordTrans2 = ogr.osr.CoordinateTransformation(proj1, point_ref)            # Create transformation for converting to WGS84

    # Pull info out of th gridder object in order to create a polygon
    ncols = gridder_obj.ncols
    nrows = gridder_obj.nrows
    x00 = gridder_obj.x00
    y00 = gridder_obj.y00
    DX = gridder_obj.DX
    DY = gridder_obj.DY

    # Create polygon object that is fully inside the outer edge of the domain
    myRing = ogr.Geometry(type=ogr.wkbLinearRing)
    myRing.AddPoint(x00+(DX/2), y00-(DY/2))
    myRing.AddPoint(x00+(ncols*DX)-(DX/2), y00-(DY/2))
    myRing.AddPoint(x00+(ncols*DX)-(DX/2), y00-(nrows*DY)+(DY/2))
    myRing.AddPoint(x00+(DX/2), y00-(nrows*DY)+(DY/2))
    myRing.AddPoint(x00+(DX/2), y00-(DY/2))                                     #close ring
    geometry = ogr.Geometry(type=ogr.wkbPolygon)
    geometry.AssignSpatialReference(proj1)
    geometry.AddGeometry(myRing)

    tic2 = time.time()
    layer = gridder_obj.getgrid(geometry.GetEnvelope(), layer)
    print('      %s polygons returned in %s seconds.' %(layer.GetFeatureCount(), time.time()-tic2))
    myRing = geometry = None

    print('Done producing output vector polygon shapefile in %ss' %(time.time()-tic))
    datasource = layer = None

def split_vertical(polygon, peices=2):
    '''Creates a specified number of clipping geometries which are boxes used to
    clip an OGR feature. Returns a list of geometry objects which are verticaly
    split chunks of the original polygon.'''

    tic = time.time()

    # Get polygon geometry information
    polygeom = polygon.GetGeometryRef()
    polygeom.CloseRings()                                                       # Ensure all rings are closed

    # Get min/max
    xmin, xmax, ymin, ymax = polygeom.GetEnvelope()                             # Get individual bounds from bounding envelope
    horizontal_dist = xmax - xmin                                               # Distance across the horizontal plane

    # Create clipping geometries
    clippolys = []           # List of new polygons
    interval = horizontal_dist/peices                                           # Split the horizontal distance using numsplits
    for split in range(peices):

        # Create clip-box bounds
        x0 = xmin+(split*interval)                                              # X-min - changes with section
        x1 = xmin+((split+1)*interval)                                          # X-max - changes with section
        y0 = ymin                                                               # Y-min - always the same
        y1 = ymax                                                               # Y-max - always the same

        # Create geometry for clip box
        myRing = ogr.Geometry(type=ogr.wkbLinearRing)
        myRing.AddPoint(x0, y1)
        myRing.AddPoint(x1, y1)
        myRing.AddPoint(x1, y0)
        myRing.AddPoint(x0, y0)
        myRing.AddPoint(x0, y1)                                                 #close ring
        geometry = ogr.Geometry(type=ogr.wkbPolygon)
        geometry.AddGeometry(myRing)

        # Add to the list of clipping geometries to be returned
        clippolys.append(geometry)
    #print('         Polygon envelope split into %s sections in %.2fs seconds' %(peices, (time.time()-tic)))
    return clippolys

def perform_intersection(gridder_obj, proj1, coordTrans, layer, fieldname, ticker=10000, corenum=0):
    '''This function performs the intersection between two geometries.'''

    # Counter initiate
    counter = 0
    counter2 = 0

    # Test intersection with layer
    tic2 = time.time()
    spatialweights = {}                                                         # This yields the fraction of the key polygon that each overlapping polygon contributes
    regridweights = {}                                                          # This yields the fraction of each overlapping polygon that intersects the key polygon - for regridding
    other_attributes = {}
    allweights = {}                                                             # This dictionary will store the final returned data

    # Attempt using a grid layer returned by the gridder object
    #print('[{0: >3}]    Layer feature count: {1: ^8}'.format(corenum, layer.GetFeatureCount()))
    for feature2 in layer:
        id2 = feature2.GetField(fieldname)
        counter2 += 1
        geometry2 = feature2.GetGeometryRef()
        geometry2.Transform(coordTrans)                                         # Transform the geometry from layer CRS to layer1 CRS
        polygon_area = geometry2.GetArea()
        #print("%s" %(geometry2.ExportToWkt()))

        # Split into parts
        if polygon_area > threshold:
            print('[{0: >3}]    Polygon: {1: ^8} area = {2: ^12}. Splitting into {3: ^2} sections.'.format(corenum, id2, polygon_area, splits))
            Areas = []
            inters = 0
            clip_polys = split_vertical(feature2, splits)

            # Create temporary output polygon vector file to store the input feature
            drv1 = ogr.GetDriverByName('Memory')
            in_ds = drv1.CreateDataSource('in_ds')
            inlayer = in_ds.CreateLayer('in_ds', srs=proj1, geom_type=ogr.wkbPolygon)
            LayerDef = inlayer.GetLayerDefn()                                    # Fetch the schema information for this layer
            infeature = ogr.Feature(LayerDef)                                     # Create a new feature (attribute and geometry)
            infeature.SetGeometry(geometry2)                                       # Make a feature from geometry object
            inlayer.CreateFeature(infeature)

            for num,clipgeom in enumerate(clip_polys):
                tic3 = time.time()

                # Create temporary output polygon vector file
                out_ds = drv1.CreateDataSource('out_ds')
                outlayer = out_ds.CreateLayer('out_ds', srs=proj1, geom_type=ogr.wkbPolygon)
                outlayer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))   # Create a new field on a layer. Add one attribute

                # Create temporary in-memory feature layer to store clipping geometry
                clip_ds = drv1.CreateDataSource('clip_ds')
                cliplayer = clip_ds.CreateLayer('clip_ds', srs=proj1, geom_type=ogr.wkbPolygon)
                LayerDef2 = cliplayer.GetLayerDefn()                                    # Fetch the schema information for this layer
                clipfeature = ogr.Feature(LayerDef2)                                     # Create a new feature (attribute and geometry)
                clipfeature.SetGeometry(clipgeom)                                       # Make a feature from geometry object
                cliplayer.CreateFeature(clipfeature)

                # Perform clip
                inlayer.Clip(cliplayer, outlayer)

                # Read clipped polygon feature
                assert outlayer.GetFeatureCount() == 1                      # Make sure the output has only 1 feature
                feat = outlayer.GetNextFeature()                            # The output should only have one feature
                geometry3 = feat.GetGeometryRef()

                # Create a Layer so that the SetSpatialFilter method can be used (faster for very large geometry2 polygons)
                drv = ogr.GetDriverByName('Memory')
                dst_ds = drv.CreateDataSource('out')
                gridlayer = dst_ds.CreateLayer('out_%s' %corenum, srs=proj1, geom_type=ogr.wkbPolygon)
                gridlayer = gridder_obj.getgrid(geometry3.GetEnvelope(), gridlayer)     # Generate the grid layer
                gridlayer.SetSpatialFilter(geometry3)                                   # Use the SetSpatialFilter method to thin the layer's geometry

                # First find all intersection areas
                Areas += [[item.GetField(0), item.geometry().Intersection(geometry3).Area(), item.geometry().Area(), item.GetField(2), item.GetField(3)] for item in gridlayer]  # Only iterate over union once
                inters += len(Areas)
                counter += inters                                                       # Advance the counter

                #flush memory
                clipfeature = clipgeom = geometry3 = feat = outlayer = cliplayer = clipfeature = None  # destroy these
                print('[{0: >3}]          Chunk {1: ^2}. Time elapsed: {2: ^4.2f} seconds.'.format(corenum, num+1, (time.time()-tic3)))

            # Collapse all duplicates back down to 1 list
            AreaDict = {}
            for item in Areas:
                try:
                    AreaDict[item[0]][1] += item[1]
                except KeyError:
                    AreaDict[item[0]] = item
            Areas = AreaDict.values()

        else:
            '''This is the normal case where polygons are smaller or more uniform in size.'''

            # Create a Layer so that the SetSpatialFilter method can be used (faster for very large geometry2 polygons)
            drv = ogr.GetDriverByName('Memory')
            dst_ds = drv.CreateDataSource('out')
            gridlayer = dst_ds.CreateLayer('out_%s' %corenum, srs=proj1, geom_type=ogr.wkbPolygon)
            gridlayer = gridder_obj.getgrid(geometry2.GetEnvelope(), gridlayer)     # Generate the grid layer
            gridlayer.SetSpatialFilter(geometry2)                                   # Use the SetSpatialFilter method to thin the layer's geometry

            # First find all intersection areas
            Areas = [[item.GetField(0), item.geometry().Intersection(geometry2).Area(), item.geometry().Area(), item.GetField(2), item.GetField(3)] for item in gridlayer]  # Only iterate over union once

            # Use the intersection area to thin the other lists
            inters = len(Areas)
            counter += inters                                                       # Advance the counter

        # Calculate weight correction
        sum_area = 0.0
        for item in Areas:
            sum_area += item[1]
        weight_correction = sum_area/polygon_area
        if abs(weight_correction-1.0)>1e-7:
            weight_correction=1.0

        # Calculate area weights - for averaging
        spatialweights[id2] = [(item[0], (item[1]/polygon_area/weight_correction)) for item in Areas]

        # Calculate regrid weights - for conservative regridding
        regridweights[id2] = [(item[0], (item[1]/item[2])) for item in Areas]

        # Store i,j variables
        other_attributes[id2] = [[item[0], item[3], item[4]] for item in Areas]
        del gridlayer, Areas, inters, dst_ds, drv

    # Counter and printed information below
    print('[{0: >3}]      [{1: ^7} intersections processed in {2: ^4.2f} s] [{3: ^8.2f} features per second] [Processed {4: ^8} features in dest grid]'.format(corenum, counter, time.time()-tic2, (counter/(time.time()-tic2)), counter2))

    # Print run information
    #print('[{0: >3}]    Done gathering intersection information between layer 1 and layer 2 in {1: 8.2f} seconds'.format(corenum, time.time()-tic2))
    #print('[{0: >3}]    {1: ^10} polygons processed for intersection with grid. {2: ^10} total polygon intersections processed.'.format(corenum, counter2, counter))
    allweights[0] = spatialweights
    allweights[1] = regridweights
    allweights[2] = other_attributes

    # Pickle the dictionary into a file (saves memory)
    allweightsfile = os.path.join(OutDir, "%s.p" %corenum)
    with open(allweightsfile, "wb") as fp:
        pickle.dump(allweights, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # Clean up and return
    del allweights
    return allweightsfile

def split(a, n):
    '''Turn a list into a number of chunks. This function is used to split up a job
    into discrete parts and handle lits of uneven length.'''
    k, m = len(a) // n, len(a) % n
    if k == 0:
        # Fewer items (a) than slots (n)
        x = ([i,i] for i in a)
    else:
        x = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))   #return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))
    return [[y[0],y[-1],num+1] for num,y in enumerate(x)]                       # Just use the start and end of each chunk

def main(gridder_obj, proj1, ticker, tic1, inDriverName, in_basins, fieldname, OutputGridPolys, chunk):
    '''This script code is intended to take advantage of the above functions in order
    to generate regridding weights between a netCDF file (ideally a GEOGRID file) or
    a high resolution grid in netCDF but with a GEOGRID file to define the coordinate
    reference system. The process takes the following steps:

        1) Use GEOGRID file to define the input coordinate reference system
        2) Save the NetCDF grid to in-memory raster format (optionally save to disk)
        3) Build rtree index to speed up the intersection process
        4) Calculate the spatial intersection between the input grid and the input polygons
        5) Export weights to netCDF.
        '''

    core = chunk[-1]                                                            # Identify the core number

    # Open the basin shapefile file with OGR, read-only access
    driver = ogr.GetDriverByName(inDriverName)                                  # 'GPKG' / 'ESRI Shapefile'
    shp = driver.Open(in_basins, 0)                                            # 0 means read-only. 1 means writeable.
    if shp is None:
        print("Open failed %s\n" % (in_basins))
        raise SystemExit
    layer = shp.GetLayer()

    proj2 = layer.GetSpatialRef()
    proj2.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    coordTrans = osr.CoordinateTransformation(proj2, proj1)                     # Coordinate tansformation from proj2 to proj1

    FID_column = layer.GetFIDColumn()
    SQL = '(%s >= %s) AND (%s <= %s)' %(FID_column, chunk[0], FID_column, chunk[1])
    #print('[{0: >3}]    SQL Statement: {1: ^}'.format(core, SQL))
    layer.SetAttributeFilter(SQL)

    # Perform the intersection between the layers and gather the spatial weights
    allweights = perform_intersection(gridder_obj, proj1, coordTrans, layer, fieldname, ticker, corenum=core)
    layer = None
    del driver, shp
    #print('[{0: >3}]    Calculation of spatial correspondences completed in {1: 8.2f} s.'.format(core, (time.time()-tic1))
    return allweights

def work(chunk):
    '''This is the worker function which wraps all globals into one call to main()
    function. The only argument needed is the chunk, which tells OGR which features
    to grab from layer.'''
    allweights = main(gridder_obj, proj1, ticker, tic1, inDriverName, in_basins, fieldname, OutputGridPolys, chunk)
    return allweights

# ---------- End Functions ---------- #

# ---------- Begin Script Execution ---------- #
if __name__ == '__main__':

    print('Script initiated at %s' %time.ctime())
    tic1 = time.time()

    gdal.AllRegister()
    OutRaster = gdal.Open(in_raster, gdal.GA_ReadOnly)                                     # Opening the file with GDAL, with read only acces

    # Getting raster dataset information
    print('Input Raster Size: %s x %s x %s' %(OutRaster.RasterXSize, OutRaster.RasterYSize, OutRaster.RasterCount))
    print('Projection of input raster: %s' %OutRaster.GetProjection())
    geotransform = OutRaster.GetGeoTransform()
    x0 = geotransform[0]                                                        # upper left corner of upper left pixel x
    y0 = geotransform[3]                                                        # upper left corner of upper left pixel y
    pwidth = geotransform[1]                                                    # pixel width, if north is up.
    pheight = geotransform[5]                                                   # pixel height is negative because it's measured from top, if north is up.
    if not geotransform is None:
        print('Origin (x,y): %s,%s' %(x0, y0))
        print('Pixel Size (x,y): %s,%s' %(pwidth, pheight))

    # Create the SpatialReference
    proj1 = osr.SpatialReference()
    proj1.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    proj1.ImportFromWkt(OutRaster.GetProjection())                     # Use projection from input raster

    # Get projection information as an OSR SpatialReference object from raster
    DX = abs(geotransform[1])                                                       # pixel width, if north is up.
    DY = abs(geotransform[5])                                                       # pixel height is negative because it's measured from top, if north is up.
    x00 = geotransform[0]                                                           # upper left corner of upper left pixel x
    y00 = geotransform[3]                                                           # upper left corner of upper left pixel y

    # Initiate the gridder class object
    ncols = OutRaster.RasterXSize
    nrows = OutRaster.RasterYSize
    print('Raster will have nrows: %s and ncols: %s' %(nrows, ncols))
    gridder_obj = Gridder_Layer(DX, DY, x00, y00, nrows, ncols)                           # Initiate grider class object which will perform the gridding for each basin
    del ncols, nrows, DX, DY, x00, y00

    # Option to save geogrid variable to raster format
    if SaveRaster == True:
        target_ds = gdal.GetDriverByName('GTiff').CreateCopy(OutGTiff, OutRaster)
        target_ds = None
    del OutRaster

    if OutputGridPolys == True:
        create_polygons_from_info(gridder_obj, proj1, OutGridFile, outDriverName, 10000)
        print('Created output grid vector file: %s' %(OutGridFile))

    # Open the basins file in order to pull out the necessary basin geometries
    driver = ogr.GetDriverByName(inDriverName)                                  # 'GPKG' / 'ESRI Shapefile'     # Open the basin shapefile file with OGR, read-only access
    shp = driver.Open(in_basins, 0)                                             # 0 means read-only. 1 means writeable.
    if shp is None:
        print("Open failed-%s\n" % (in_basins))
        raise SystemExit
    layer = shp.GetLayer()
    numbasins = layer.GetFeatureCount()

    # Check for existence of provided fieldnames
    field_defn, fieldslist = checkfield(layer, fieldname, 'shapefile2')
    fieldtype2 = getfieldinfo(field_defn, fieldname)                            # Get information about field types for buildng the output NetCDF file later
    fieldtype1 = fieldtype2
    layer = shp = driver = None
    del driver, shp, layer, field_defn, fieldslist
    print('Found %s polygons in layer. Time elapsed: %s' %(numbasins, time.time()-tic1))

    print('Found %s processors. Distributing processing across %s core(s) using %s worker processes.' %(CPU_Count, Processors, pool_size))
    pool = multiprocessing.Pool(Processors)
    chunks = list(split(range(1, numbasins+1), pool_size))                      # Split the list of basin ids into x chunks

    # The following command works
    results = pool.map(work, [chunk for chunk in chunks], chunksize=1)          # This farms the function 'work' out to the number of processors defined by 'Processors'
    pool.close()
    pool.join()
    print('Length of results: %s . Time elapsed: %s' %(len(results), time.time()-tic1))

    # Get the size of the dimensions for constructing the netCDF file
    print('Beginning to get the size of the dictionaries.')
    dim1size = 0
    dim2size = 0
    counter = 1
    for dictionary in results:
        #print('%s: Dictionary file: %s' %(counter, dictionary))
        allweights = loadPickle(dictionary)
        dim1size += len(allweights[0])
        dim2size += sum([len(item) for item in allweights[0].values()])
        allweights = None
        print('  [%s] Finished gathering dictionary length information from dictionary: %s' %(counter, dictionary))
        counter += 1
    print('Number of pickled dictionaries found: %s' %(counter-1))

    '''Create a long-vector netCDF file. '''
    # variables for compatability with the code below, which was formerly from a function
    gridflag = 1

    print('Beginning to build weights netCDF file: %s . Time elapsed: %s' %(regridweightnc, time.time()-tic1))
    tic = time.time()

    # Create netcdf file for this simulation
    rootgrp = netCDF4.Dataset(regridweightnc, 'w', format=NC_format)

    # Create dimensions and set other attribute information
    dim1name = 'polyid'
    dim2name = 'data'
    dim1 = rootgrp.createDimension(dim1name, dim1size)
    dim2 = rootgrp.createDimension(dim2name, dim2size)
    print('    Dimensions created after {0: 8.2f} seconds.'.format(time.time()-tic))

    # Handle the data type of the polygon identifier
    if fieldtype1 == 'integer':
        ids = rootgrp.createVariable(dim1name, 'i4', (dim1name))                # Coordinate Variable (32-bit signed integer)
        masks = rootgrp.createVariable('IDmask', 'i4', (dim2name))              # (32-bit signed integer)
    elif fieldtype1 == 'integer64':
        ids = rootgrp.createVariable(dim1name, 'i8', (dim1name))                # Coordinate Variable (64-bit signed integer)
        masks = rootgrp.createVariable('IDmask', 'i8', (dim2name))              # (64-bit signed integer)
    elif fieldtype1 == 'string':
        ids = rootgrp.createVariable(dim1name, str, (dim1name))                 # Coordinate Variable (string type character)
        masks = rootgrp.createVariable('IDmask', str, (dim2name))               # (string type character)
    print('    Coordinate variable created after {0: 8.2f} seconds.'.format(time.time()-tic))

    # Create fixed-length variables
    overlaps = rootgrp.createVariable('overlaps', 'i4', (dim1name))             # 32-bit signed integer
    weights = rootgrp.createVariable('weight', 'f8', (dim2name))                # (64-bit floating point)
    rweights = rootgrp.createVariable('regridweight', 'f8', (dim2name))         # (64-bit floating point)

    if gridflag == 1:
        iindex = rootgrp.createVariable('i_index', 'i4', (dim2name))            # (32-bit signed integer)
        jindex = rootgrp.createVariable('j_index', 'i4', (dim2name))            # (32-bit signed integer)
        iindex.long_name = 'Index in the x dimension of the raster grid (starting with 1,1 in LL corner)'
        jindex.long_name = 'Index in the y dimension of the raster grid (starting with 1,1 in LL corner)'
    print('    Variables created after {0: 8.2f} seconds.'.format(time.time()-tic))

    # Set variable descriptions
    masks.long_name = 'Polygon ID (polyid) associated with each record'
    weights.long_name = 'fraction of polygon(polyid) intersected by polygon identified by poly2'
    rweights.long_name = 'fraction of intersecting polyid(overlapper) intersected by polygon(polyid)'
    ids.long_name = 'ID of polygon'
    overlaps.long_name = 'Number of intersecting polygons'
    print('    Variable attributes set after {0: 8.2f} seconds.'.format(time.time()-tic))

    # Fill in global attributes
    rootgrp.history = 'Created %s' %time.ctime()

    # Iterate over dictionaries and begin filling in NC variable arrays
    dim1len = 0
    dim2len = 0
    counter = 1
    for dictionary in results:
        tic2 = time.time()

        # Create dictionaries
        allweights = loadPickle(dictionary)
        spatialweights = allweights[0].copy()                                       # Updates new dictionary with another one
        regridweights = allweights[1].copy()                                        # Updates new dictionary with another one
        other_attributes = allweights[2].copy()                                     # Updates new dictionary with another one
        allweights = None

        # Set dimensions for this slice
        dim1start = dim1len
        dim2start = dim2len
        dim1len += len(spatialweights)
        dim2len += sum([len(item) for item in spatialweights.values()])

        # Start filling in elements
        if fieldtype1 == 'integer':
            #ids[dim1start:dim1len] = numpy.array(spatialweights.keys())                         # Ths method works for int-type netcdf variable
            ids[dim1start:dim1len] = numpy.array([x[0] for x in iter(spatialweights.items())])    # Test to fix ordering of ID values
        if fieldtype1 == 'integer64':
            #ids[dim1start:dim1len] = numpy.array(spatialweights.keys())                         # Ths method works for int-type netcdf variable
            ids[dim1start:dim1len] = numpy.array([x[0] for x in iter(spatialweights.items())], dtype=numpy.int_)    # Test to fix ordering of ID values
        elif fieldtype1 == 'string':
            #ids[dim1start:dim1len] = numpy.array(spatialweights.keys(), dtype=numpy.object)     # This method works for a string-type netcdf variable
            ids[dim1start:dim1len] = numpy.array([x[0] for x in iter(spatialweights.items())], dtype=numpy.object_)    # Test to fix ordering of ID values

        overlaps[dim1start:dim1len] = numpy.array([len(x) for x in spatialweights.values()])

        masklist = [[x[0] for y in x[1]] for x in iter(spatialweights.items())]       # Get all the keys for each list of weights
        masks[dim2start:dim2len] = numpy.array([item for sublist in masklist for item in sublist], dtype=numpy.object_)  # Flatten to 1 list (get rid of lists of lists)
        del masklist

        weightslist = [[item[1] for item in weight] for weight in iter(spatialweights.values())]
        weights[dim2start:dim2len] = numpy.array([item for sublist in weightslist for item in sublist], dtype=numpy.object_)
        del weightslist

        rweightlist = [[item[1] for item in rweight] for rweight in iter(regridweights.values())]
        rweights[dim2start:dim2len] = numpy.array([item for sublist in rweightlist for item in sublist], dtype=numpy.object_)
        del rweightlist

        if gridflag == 1:
            iindexlist= [[item[1] for item in attribute] for attribute in iter(other_attributes.values())]
            iindex[dim2start:dim2len] = numpy.array([item for sublist in iindexlist for item in sublist], dtype=numpy.object_)
            del iindexlist
            jindexlist = [[item[2] for item in attribute] for attribute in iter(other_attributes.values())]
            jindex[dim2start:dim2len] = numpy.array([item for sublist in jindexlist for item in sublist], dtype=numpy.object_)
            del jindexlist

        spatialweights = regridweights = other_attributes = None
        print('  [%s] Done setting dictionary: %s in %s seconds.' %(counter, os.path.basename(dictionary), time.time()-tic2))
        counter += 1
        os.remove(dictionary)                                                       # Delete the picled dictionary file

    # Close file
    rootgrp.close()
    del fieldtype1
    print('NetCDF correspondence file created in {0: 8.2f} seconds.'.format(time.time()-tic))

    # Build regrid and spatial weights NetCDF file.
    print('Finished building weights netCDF file: %s . Time elapsed: %s' %(regridweightnc, time.time()-tic1))
    # ---------- End Script Execution ---------- #
