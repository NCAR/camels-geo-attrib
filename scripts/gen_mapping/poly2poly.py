#!/usr/bin/env python

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# ** Copyright UCAR (c) 1992 - 2012
# ** University Corporation for Atmospheric Research(UCAR)
# ** National Center for Atmospheric Research(NCAR)
# ** Research Applications Laboratory(RAL)
# ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
# ** 2015/1/28
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

'''
 Name:         poly2poly_py3.py
 Author:       Kevin Sampson
               Associate Scientist
               National Center for Atmospheric Research
               ksampson@ucar.edu

 Created:      10/23/2013
 Modified:     1/20/2016
               2023ish:  AWW: converted to python3 and set to work with ncar pylibs; also now handles int64 hruIds
'''

'''This script is a standalone python tool which is intended to be run on the command
line. The tool takes two polygon vector geometry files in GeoPackage (.gpkg) format
and computes the geometric intersection between all polygons in the first feature class
and each feature in the second feature class. The intersecting areas are written to a
custom netCDF correspondence file format to be used for aggregating/disaggregating
data from one set of polygons to another in a spatial-statistical fasion.'''

ALP = False                                                                     # If running on Windows machine ALP, must start python VirtualEnv
if ALP == True:
    # Activate Virtual Environment so we can use Numpy 1.9.x, which GDAl is compiled against
    activate_this_file = r'C:\Python27\VirtualEnv_x64\Scripts\activate_this.py'     # Path to Virtual Environment python activate script
    execfile(activate_this_file, dict(__file__=activate_this_file))

# Import modules
import os, sys, time, getopt
from osgeo import ogr
from osgeo import osr
import numpy
import multiprocessing                                                          # Added 1/20/2016
import netCDF4 as nc4
import collections
from operator import itemgetter
try:
   import cPickle as pickle
except:
   import pickle

# Module Settings
tic = time.time()
sys.dont_write_bytecode = True                                                  # Place here so bytecode is not written
multiprocessing.freeze_support()

# Screen print in case invalid parameters are given
use = '''
Usage: %s <Polygon1> <FieldName1> <Polygon2> <FieldName2> [<gridflag>] [<outputNC>]
      <Polygon1>     -> Polygon GeoPackage (full path)
      <FieldName1>   -> Polygon identifier field name for Polygon1
      <Polygon2>     -> Polygon GeoPackage2 (full path)
      <FieldName2>   -> Polygon identifier field name for Polygon 2
      <gridflag>     -> Optional - Type 'GRID' to indicate Polygon2 was generated from grid2shp.py
      <outputNC>     -> Optional - Full path to output netCDF file (.nc extentsion required)
'''

# ---------- Global variables ---------- #

# Vector layer format
DriverName = 'GPKG'                                                             # 'GPKG' / 'ESRI Shapefile'

# Multiprocessing parameters
ticker = 1000                                                                    # Ticker for how often to print a message
CPU_Count = multiprocessing.cpu_count()                                         # Find the number of CPUs to chunk the data into
Processors = CPU_Count-1                                                        # To spread the processing over a set number of cores, otherwise use Processors = CPU_Count
pool_multiplier = 100                                                            # How many workers to have per CPU
pool_size = Processors*pool_multiplier                                          # Having a large number of worker processes can help with overly slow cores
print('Found %s processors. Using %s worker prcocesses.' % (CPU_Count, pool_size))

# Input arguments - very bad programming not to trap these in the if __name__ == "main": section!
infeatures1 = sys.argv[1]
fieldname1 = sys.argv[2]
infeatures2 = sys.argv[3]
fieldname2 = sys.argv[4]
gridf = 0
outnc = None
OutDir = None                                                                   # Set this global and change it below
if len(sys.argv) > 5:
    if 'GRID' in sys.argv[:]:
        gridf = 1
    else:
        outnc = sys.argv[-1]
        OutDir = os.path.dirname(outnc)
# Conditionals for optional input arguments
if len(sys.argv) > 6:
    if os.path.exists(os.path.dirname(sys.argv[6])):
        outnc = sys.argv[6]
        OutDir = os.path.dirname(outnc)
indir = os.path.dirname(infeatures2)                                            # Use directory of output file
if OutDir is None:
    OutDir = indir
projfeatures = os.path.join(indir, os.path.basename(infeatures2).replace(".gpkg", "_proj.gpkg"))    # Come up with the projected feature class file name based on inputs

# ---------- Global variables ---------- #

# ---------- Functions ---------- #

def clean_up(infeatures1, infeatures2, outputnc=None):
    '''Function to check inputs and delete existing files for outputs.'''

    tic1 = time.time()

    # Output Files
    if outputnc is None:
        outdir = os.path.dirname(infeatures2)                                    # Use directory of input file
        outncFile = os.path.join(outdir, os.path.basename(infeatures1)[:-5]+'.nc')
    else:
        outdir = os.path.dirname(outputnc)
        outncFile = outputnc
    print('Output Directory: %s' % outdir)
    projfeatures = os.path.join(outdir, os.path.basename(infeatures2).replace(".gpkg", "_proj.gpkg"))

    # Check if files exist and delete existing files if necessary
    if os.path.isfile(projfeatures):
        print("Removing existing file: %s" % projfeatures)
        os.remove(projfeatures)
    if os.path.isfile(outncFile):
        print("Removing existing file: %s" % outncFile)
        os.remove(outncFile)

    print("Clean-up step completed in %s seconds." % (time.time()-tic1))
    return projfeatures, outncFile

def Read_VectorDS(inFC, DriverName):
    '''Function to read and return the OGR vector layer'''

    # Test for presence of appropriate driver
    drivers = list_drivers()
    if DriverName in drivers:
        driver = ogr.GetDriverByName(DriverName)                                # Set driver for data source
    else:
        driver = None
        print('Driver %s not found in list of OGR supported drivers.' % (DriverName))

    # Open the basin vector file with OGR, with read only access
    data_source = driver.Open(inFC, 0)                                             # 0 means read-only. 1 means writeable.
    if data_source is None:
        print("Open failed on input %s." % (inFC))
        raise SystemExit
    layer = data_source.GetLayer()
    layername = layer.GetName()
    return layer, layername, data_source                                        #, driver

def project_to_input(infeatures, outputFile, to_project, DriverName):
    '''This function projects the input features (to_project) in GeoPackage
    format and uses ogr2ogr.py (if present in the same directory as this script)
    to the coordinate system of 'infeatures' GeoPackage.'''

    to_layer, to_layerName, to_data_source = Read_VectorDS(to_project, DriverName)
    to_spatialref = to_layer.GetSpatialRef()
    to_spatialref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    layer_defn = to_layer.GetLayerDefn()
    geom_type = layer_defn.GetGeomType()

    layer, layerName, data_source = Read_VectorDS(infeatures, DriverName)
    #spatialref = layer.GetSpatialRef().ExportToWkt()
    spatialref = layer.GetSpatialRef()
    spatialref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transform = osr.CoordinateTransformation(to_spatialref, spatialref)

    # create the output layer
    driver = ogr.GetDriverByName(DriverName)
    outDataSet = driver.CreateDataSource(outputFile)
    outLayer = outDataSet.CreateLayer(to_layerName, geom_type=geom_type)

    # add fields
    layerDefn = to_layer.GetLayerDefn()
    for i in range(0, layerDefn.GetFieldCount()):
        fieldDefn = layerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()

    # loop through the input features
    features = to_layer.GetNextFeature()
    while features:
        # get the input geometry
        geom = features.GetGeometryRef()
        # reproject the geometry
        geom.Transform(transform)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), features.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # dereference the features and get the next input feature
        outFeature = None
        features = to_layer.GetNextFeature()

    return outputFile

def checkfield(layer, fieldname, string1):
    '''Check for existence of provided fieldnames'''
    layerDefinition = layer.GetLayerDefn()
    fieldslist = []
    for i in range(layerDefinition.GetFieldCount()):
        fieldslist.append(layerDefinition.GetFieldDefn(i).GetName())
    if fieldname in fieldslist:
        i = fieldslist.index(fieldname)
        field_defn = layerDefinition.GetFieldDefn(i)
    else:
        print('    Field %s not found in input %s. Terminating...' % (fieldname, string1))
        raise SystemExit
    return field_defn, fieldslist

def getfieldinfo(field_defn, fieldname):
    '''Get information about field type for buildng the output NetCDF file later'''
    if field_defn.GetType() == ogr.OFTInteger:
        fieldtype = 'integer'
        print("found ID type of Integer")
    elif field_defn.GetType() == ogr.OFTReal:
        fieldtype = 'real'
        print("found ID type of Real")
        #print "field type: OFTReal not currently supported in output NetCDF file."
        #raise SystemExit
    elif field_defn.GetType() == ogr.OFTInteger64:
        fieldtype = 'integer64'
        print("found ID type of Integer64")       #        print "%d" % feat.GetFieldAsInteger(i)
    elif field_defn.GetType() == ogr.OFTString:
        fieldtype = 'string'
        print("found ID type of String")
    else:
        print("ID Type not found ... Exiting")
        raise SystemExit
    print("    Field Type for field '%s': %s (%s)" %(fieldname, field_defn.GetType(), fieldtype))
    return fieldtype

def list_drivers():
    '''This function lists all driver available to OGR.'''
    cnt = ogr.GetDriverCount()
    formatsList = []  # Empty List

    for i in range(cnt):
        driver = ogr.GetDriver(i)
        driverName = driver.GetName()
        if not driverName in formatsList:
            formatsList.append(driverName)
    return formatsList

def split(a, n):
    '''Turn a list into a number of chunks. This function is used to split up a job
    into discrete parts and handle lits of uneven length.'''
    k, m = len(a) // n, len(a) % n
    if k == 0:
        # Fewer items (a) than slots (n)
        x = ([i,i] for i in a)
    else:
        x = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))   #return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))
    return [[y[0],y[-1]] for y in x]                                            # Just use the start and end of each chunk

def perform_intersection(layer1, layer2, fieldname1, fieldname2, name, OutDir, gridf, ticker=10000, corenum=0):
    '''This function takes two input OGR layers (both must be in the same coordinate
    reference sysetm) and performs polygon-by-polygon analysis to compute the
    individual weight of each grid cell using OGR routines.'''

    # Initiate counters
    counter = 0
    counter2 = 0

    # Set up clocks for time printing
    tic2 = time.time()
    tic3 = time.time()

    # Initiate dictionaries to store results
    spatialweights = {}                                                         # This yields the fraction of the key polygon that each overlapping polygon contributes
    regridweights = {}                                                          # This yields the fraction of each overlapping polygon that intersects the key polygon - for regridding
    allweights = {}                                                             # This dictionary will store the final returned data
    other_attributes = {}

    print('[{0: >3}]    Layer feature count: {1: ^8}'.format(corenum, layer1.GetFeatureCount()))
    for feature1 in layer1:
        id1 = feature1.GetField(fieldname1)
        spatialweights[id1] = []
        regridweights[id1] = []
        other_attributes[id1] = []
        counter2 += 1
        geometry1 = feature1.GetGeometryRef()
        geometry1_area = geometry1.GetArea()

        # Set spatial filter (optional, but may increase/decrease speed of execution)
        layer2.SetSpatialFilter(geometry1)                                      # Spatial filter layer2 based on layer1

        if layer2.GetFeatureCount() == 0:
            '''Handle the case where there are no intersectors. Must still provide input data to the
            correspondence file in the form of null entries.'''
            print('[{0: >3}]  --    Found an input polygon from input1 with no intersector from input2: %s:%s'.format(corenum, fieldname1, id1))
            id2 = 0

            # Calculate area weights - for averaging
            existinglist = spatialweights[id1]
            existinglist.append((id2, 0))

            # Calculate regrid weights - for conservative regridding
            existinglist2 = regridweights[id1]
            existinglist2.append((id2, 0))

            existinglist3 = other_attributes[id1]
            existinglist3.append((id2, 0, 0, 0, 0))
            continue

        # Gather information from layer2
        for feature2 in layer2:
            geometry2 = feature2.GetGeometryRef()
            if geometry1.Intersects(geometry2):
                counter += 1
                id2 = feature2.GetField(fieldname2)

                # Latitude and Longtitude calculation
                geometry2_area = geometry2.GetArea()

                # Calculate area weights - for averaging
                existinglist = spatialweights[id1]
                existinglist.append((id2, (geometry2.Intersection(geometry1).GetArea()/geometry1_area)))

                # Calculate regrid weights - for conservative regridding
                existinglist2 = regridweights[id1]
                existinglist2.append((id2, (geometry1.Intersection(geometry2).GetArea()/geometry2_area)))

                existinglist3 = other_attributes[id1]
                if gridf == 1:
                    # Store i,j variables
                    existinglist3.append((id2, feature2.GetField('i_index'), feature2.GetField('j_index'), feature2.GetField('lon_cen'), feature2.GetField('lat_cen')))
                else:
                    existinglist3.append((id2, 0, 0, 0, 0))

                # Counter and printed information below
                if counter % ticker == 0:
                    print('[{0: >3}]        [{1: ^10} feature intersections in {2: ^4.2f} s] [{3: ^10} total at {4: ^8.2f} features per second] [Processed {5: ^8} features in dest grid]'.format(corenum, ticker, time.time()-tic3, counter, (counter/(time.time()-tic2)), counter2))
                    tic3 = time.time()
                geometry2 = None                              # Alternatively call .Destroy()

    # Print run information
    print('[{0: >3}]    Done gathering intersection information between layer 1 and layer 2 in {1: 8.2f} seconds'.format(corenum, time.time()-tic2))
    print('[{0: >3}]    {1: ^10} polygons processed for intersection with grid. {2: ^10} total polygon intersections processed.'.format(corenum, counter2, counter))
    allweights[0] = spatialweights
    allweights[1] = regridweights
    allweights[2] = other_attributes

    # Pickle the dictionary into a file (saves memory)
    ##allweightsfile = os.path.join(OutDir, "%s.p" %corenum)
    ##with open(allweightsfile, "wb") as fp:
    ##    cPickle.dump(allweights, fp, protocol=cPickle.HIGHEST_PROTOCOL)

    # Clean up and return
    print('[{0: >3}]  Finished creating OGR layers for the input feature classes after {1: 8.2f} seconds. -- '.format(corenum, (time.time()-tic)))
    #return allweightsfile
    return allweights

def create_longvector_ncfile(spatialweights, regridweights, other_attributes, outputfile, fieldtype1, fieldtype2, gridflag):
    '''This function creates a ragged-array netCDF file, with variable length dimensions for each basin. '''

    print('    NetCDF correspondence file creation started.')
    tic1 = time.time()

    # Create netcdf file for this simulation
    rootgrp = nc4.Dataset(outputfile, 'w', format='NETCDF4')

    # Create dimensions and set other attribute information
    dim1name = 'polyid'
    dim2name = 'data'
    dim1size = len(spatialweights)
    dim2size = sum([len(x) for x in spatialweights.values()])
    dim1 = rootgrp.createDimension(dim1name, dim1size)
    dim2 = rootgrp.createDimension(dim2name, dim2size)
    print('        Dimensions created after {0: 8.2f} seconds.'.format(time.time()-tic1))

    # Handle the data type of the polygon identifier
    if fieldtype1 == 'integer':
        ids = rootgrp.createVariable(dim1name, 'i4', (dim1name))                # Coordinate Variable (32-bit signed integer)
    elif fieldtype1 == 'integer64':
        ids = rootgrp.createVariable(dim1name, 'i8', (dim1name))                # Coordinate Variable (64-bit signed integer)
    elif fieldtype1 == 'string':
        ids = rootgrp.createVariable(dim1name, str, (dim1name))                 # Coordinate Variable (string type character)
    print('        Coordinate variable created after {0: 8.2f} seconds.'.format(time.time()-tic1))

    if fieldtype2 == 'integer':
        ids2 = rootgrp.createVariable('intersector', 'i4', (dim2name))          # Coordinate Variable (32-bit signed integer)
    elif fieldtype2 == 'integer64':
        ids2 = rootgrp.createVariable('intersector', 'i8', (dim2name))          # Coordinate Variable (64-bit signed integer)
    elif fieldtype2 == 'string':
        ids2 = rootgrp.createVariable('intersector', str, (dim2name))           # Coordinate Variable (string type character)

    # Create fixed-length variables
    overlaps = rootgrp.createVariable('overlaps', 'i4', (dim1name))             # 32-bit signed integer
    weights = rootgrp.createVariable('weight', 'f8', (dim2name))                # (64-bit floating point)
    rweights = rootgrp.createVariable('regridweight', 'f8', (dim2name))         # (64-bit floating point)
    # made masks definition flexible (int or string) - AWW ## Comment SG July 15
    if fieldtype1 == 'integer':
        masks = rootgrp.createVariable('IDmask', 'i4', (dim2name))              # (32-bit signed integer) ORIG
    elif fieldtype1 == 'integer64':
        masks = rootgrp.createVariable('IDmask', 'i8', (dim2name))              # (64-bit signed integer)
    elif fieldtype1 == 'string':
        masks = rootgrp.createVariable('IDmask', str, (dim2name))               # Coord. Variable (string type character)

    if gridflag == 1:
        iindex = rootgrp.createVariable('i_index', 'i4', (dim2name))            # (32-bit signed integer)
        jindex = rootgrp.createVariable('j_index', 'i4', (dim2name))            # (32-bit signed integer)
        lats = rootgrp.createVariable('latitude', 'f8', (dim2name))             # (64-bit floating point)
        lons = rootgrp.createVariable('longitude', 'f8', (dim2name))            # (64-bit floating point)
        # iindex.long_name = 'Index in the x dimension of the raster grid (starting with 1,1 in LL corner)' ORIG
        # jindex.long_name = 'Index in the y dimension of the raster grid (starting with 1,1 in LL corner)' ORIG
        iindex.long_name = 'Index in the x dimension of the raster grid (starting with 1,1 in UL/NW corner)' # AWW
        jindex.long_name = 'Index in the y dimension of the raster grid (starting with 1,1 in UL/NW corner)' # AWW
        lats.long_name = 'Centroid latitude of intersecting polygon in degrees north in WGS84 EPSG:4326'
        lons.long_name = 'Centroid longitude of intersecting polygon in degrees east in WGS84 EPSG:4326'
        lats.units = 'degrees north'
        lons.units = 'degrees east'
    print('        Variables created after {0: 8.2f} seconds.'.format(time.time()-tic1))

    # Set variable descriptions
    masks.long_name = 'Polygon ID (polyid) associated with each record'
    weights.long_name = 'fraction of polygon(polyid) intersected by polygon identified by poly2'
    rweights.long_name = 'fraction of intersecting polyid(overlapper) intersected by polygon(polyid)'
    ids.long_name = 'ID of polygon'
    overlaps.long_name = 'Number of intersecting polygons'
    ids2.long_name = 'ID of the polygon that intersetcs polyid'
    print('        Variable attributes set after {0: 8.2f} seconds.'.format(time.time()-tic1))

    # Fill in global attributes
    rootgrp.history = 'Created %s' %time.ctime()

    # Quick check to ensure all dictionaries have the same keys
    #print '        All dictionaries found to have the same keys: %s' %(spatialweights.keys() == regridweights.keys())
    #print '        All dictionaries found to have the same shapes: %s' %([len(x) for x in spatialweights.values()] == [len(x) for x in regridweights.values()])

    # Start filling in elements
    id_list = list(spatialweights.keys())
    if fieldtype1 == 'integer':
        ids[:] = numpy.array(id_list)          # Ths method works for int-type netcdf variable
    elif fieldtype1 == 'integer64':
        ids[:] = numpy.array(id_list, dtype=numpy.int_)          # Ths method works for int-type netcdf variable
    elif fieldtype1 == 'string':
        ids[:] = numpy.array(id_list, dtype=numpy.object_)         # This method works for a string-type netcdf variable
    print('        polyid data set after {0: 8.2f} seconds.'.format(time.time()-tic1))

    ids2list = [[y[0] for y in x[1]] for x in spatialweights.items()]       # Get all the intersecting polygon ids
    if fieldtype2 == 'integer':
        ids2[:] = numpy.array([item for sublist in ids2list for item in sublist])   # Flatten to 1 list (get rid of lists of lists)
    elif fieldtype2 == 'integer64':
        ids2[:] = numpy.array([item for sublist in ids2list for item in sublist], dtype=numpy.int_)   # Flatten to 1 list (get rid of lists of lists)
    elif fieldtype2 == 'string':
        ids2[:] = numpy.array([item for sublist in ids2list for item in sublist], dtype=numpy.str_)   # object_ This method works for a string-type netcdf variable
    print('        intersector data set after {0: 8.2f} seconds.'.format(time.time()-tic1))

    #overlaps[:] = numpy.array([len(x) for x in spatialweights.values()])
    overlaps[:] = numpy.array([0 if len(x)==1 and x[0][0]==0 else len(x) for x in spatialweights.values()])        # Handle issue of when there is a 0 in the weight
    print('        overlaps data set after {0: 8.2f} seconds.'.format(time.time()-tic1))

    masklist = [[x[0] for y in x[1]] for x in spatialweights.items()]       # Get all the keys for each list of weights
    masks[:] = numpy.array([item for sublist in masklist for item in sublist])  # Flatten to 1 list (get rid of lists of lists)
    del masklist
    print('        mask data set after {0: 8.2f} seconds.'.format(time.time()-tic1))

    weightslist = [[item[1] for item in weight] for weight in spatialweights.values()]
    weights[:] = numpy.array([item for sublist in weightslist for item in sublist], dtype=numpy.object_)
    del weightslist
    print('        weight data set after {0: 8.2f} seconds.'.format(time.time()-tic1))

    rweightlist = [[item[1] for item in rweight] for rweight in regridweights.values()]
    rweights[:] = numpy.array([item for sublist in rweightlist for item in sublist], dtype=numpy.object_)
    del rweightlist
    print('        regridweight data set after {0: 8.2f} seconds.'.format(time.time()-tic1))

    if gridflag == 1:
        iindexlist= [[item[1] for item in attribute] for attribute in other_attributes.values()]
        iindex[:] = numpy.array([item for sublist in iindexlist for item in sublist], dtype=numpy.object_)
        del iindexlist
        print('        i_index data set after {0: 8.2f} seconds.'.format(time.time()-tic))
        jindexlist = [[item[2] for item in attribute] for attribute in other_attributes.values()]
        jindex[:] = numpy.array([item for sublist in jindexlist for item in sublist], dtype=numpy.object_)
        del jindexlist
        print('        j_index data set after {0: 8.2f} seconds.'.format(time.time()-tic))
        lonslist = [[item[3] for item in attribute] for attribute in other_attributes.values()]
        lons[:] = numpy.array([item for sublist in lonslist for item in sublist], dtype=numpy.object_)
        del lonslist
        print('        lon_cen data set after {0: 8.2f} seconds.'.format(time.time()-tic))
        latslist = [[item[4] for item in attribute] for attribute in other_attributes.values()]
        lats[:] = numpy.array([item for sublist in latslist for item in sublist], dtype=numpy.object_)
        print('        lat_cen data set after {0: 8.2f} seconds.'.format(time.time()-tic))
        del latslist

    # Close file
    rootgrp.close()
    print('    NetCDF correspondence file created in {0: 8.2f} seconds.'.format(time.time()-tic1))

def main(infeatures1, infeatures2, ticker, tic, DriverName, fieldname1, fieldname2, name, OutDir, gridf, chunk):
    '''This script code is intended to take advantage of the above functions in order
    to generate regridding weights between two polygon vector feature classes. This
    function reads the input layer and prepares it for spatial-weight derivation.'''

    tic1 = time.time()

    # Attempt to find the core number (not accurate for the last several cores with unequal range of values)
    core = chunk[-1]/(chunk[-1]-(chunk[0]-1))

    # Open the first input feature class with OGR, read-only access
    layer1, layername1, data_source1 = Read_VectorDS(infeatures1, DriverName)
    SQL = '(FID >= %s) AND (FID <= %s)' %(chunk[0], chunk[-1])
    print('[{0: >3}]    SQL Statement: {1: ^}'.format(core, SQL))
    #print '[{0: >15}]    SQL Statement: {1: ^}'.format(name, SQL)
    layer1.SetAttributeFilter(SQL)                                              # Limit features to a particular chunk

    # Open the second input feature class with OGR, read-only access
    layer2, layername2, data_source2 = Read_VectorDS(infeatures2, DriverName)

    # Create a Polygon from the extent tuple
    extent1 = layer1.GetExtent()
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(extent1[0], extent1[2])
    ring.AddPoint(extent1[1], extent1[2])
    ring.AddPoint(extent1[1], extent1[3])
    ring.AddPoint(extent1[0], extent1[3])
    ring.AddPoint(extent1[0], extent1[2])
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    layer2.SetSpatialFilter(poly)                                               # Spatial filter layer2 based on layer1
    poly = ring = extent1 = None                                                # Eliminate the polygon part

    # Perform the intersection between the layers and gather the spatial weights
    allweights = perform_intersection(layer1, layer2, fieldname1, fieldname2, name, OutDir, gridf, ticker, corenum=core)
    layer1 = layername1 = data_source1 = None
    layer2 = layername2 = data_source2 = None
    print('[{0: >3}]    Calculation of spatial correspondences completed in {1: 8.2f} s.'.format(core, (time.time()-tic1)))
    #print '[{0: >15}]    Calculation of spatial correspondences completed in {1: 8.2f} s.'.format(name, (time.time()-tic1))
    return allweights

def work(chunk):
    '''This is the worker function which wraps all globals into one call to main()
    function. The only argument allowed is the chunk, which tells OGR which features
    to grab from layer. This is because the multiprocessing Pool.map can only pass
    one argument to the function it calls.'''

    name = multiprocessing.current_process().name
    allweights = main(infeatures1, projfeatures, ticker, tic, DriverName, fieldname1, fieldname2, name, OutDir, gridf, chunk)
    return allweights

def loadPickle(fp):
    with open(fp, 'rb') as fh:
        listOfObj = pickle.load(fh)
    return listOfObj

def check_correspondence_file(in_nc):
    '''This function will check the correspondence file and print information to determine
    the validity of the file variables.'''

    print('    Beginning to run correspondence file through checking routines.')

    # Function for gathering sum from 'groups'
    def group_sum(l, g):
        groups = collections.defaultdict(int)
        for li, gi in zip(l, g):
            groups[gi] += li
        #return map(itemgetter(1), sorted(groups.items()))
        return map(itemgetter(1), groups.items())

    # Find if the spatial weights sum to 1
    rootgrp = nc4.Dataset(in_nc, 'r', format='NETCDF4')

    polyids = rootgrp.variables['polyid']
    overlaps = rootgrp.variables['overlaps']
    weights = rootgrp.variables['weight']
    IDmask = rootgrp.variables['IDmask']

    weightsum = group_sum(weights[:], IDmask[:])
    weightlen = group_sum(numpy.ones(weights.shape), IDmask[:])                 # Hack using group_sum function and weights of 1.0 to get counts
    # Print diagnosis information
    print('        Number of basins: %s' %polyids.shape[0])
#    print('        Number of intersections: %s' %weights[:][weights[:]>0].shape[0])    # Fix for weights of 0   old: weights.shape[0]
    print('        Sum of overlaps: %s' %overlaps[:].sum() )
#    print('        Minimum sum of weights: %s' %min(weightsum))
#    #print('        Number of weights with sum=0: %s' %weightsum.count(0))
#    print('        Maximum sum of weights: %s' %max(weightsum))
#    print('        Mean sum of weights: %s' %numpy.mean(weightsum))
#    print('        Number of weights within .001 of 1.0: %s of %s' %(sum([1 for x in weightsum if x < 1.001 and x > 0.999]), len(weightsum)))
#    print('        Maximum number of overlaps: %s' %max(weightlen))
#    print('        Minimum number of overlaps: %s' %min(weightlen))
    print('        Mean number of overlaps: %s' %overlaps[:].mean())

    # Create dicitonary to store key:frequency pairs
    collection = collections.Counter(IDmask[:])

    # Extra checking of overlaps variable and IDmask variable similarities - takes some time for large data dimensions
    # This method will make sure that the ordering and overlaps values are correct

    # A. Wood -- this part is failing but it is after the output file is written -- need to debug later

    cumsum = numpy.cumsum(overlaps[:])
    cumsum2 = numpy.insert(cumsum, 0, 0)                                            # Insert 0 index at beginning of array
    del cumsum                                                                      # Free up memory
    myarray = IDmask[cumsum2[:-1]]                                                  # Use cumulative sum as index to check values against
    differences = polyids.shape[0] - numpy.array(myarray == polyids[:]).sum()
    print('        Found %s differences between overlaps value and number of the same polyid in IDmask' %differences)
    del cumsum2, myarray, differences

    # test ordering - too slow
    mydict = {x:y for x, y in zip(polyids[:], overlaps[:])}
    samelist = [1 if x[1]==collection[x[0]] else 0 for x in mydict.items()]
    print('        %s of %s values have a match between overlaps varable and IDmasks variable' %(sum(samelist), len(samelist)))

# ---------- Functions ---------- #

if __name__ == '__main__':
    print("Starting __main__ function")

    def usage():
      sys.stderr.write(use % sys.argv[0])
      sys.exit(1)
    try:
      (opts, args) = getopt.getopt(sys.argv[1:], 'h')
    except getopt.error:
      usage()

    if len(sys.argv) < 5 or len(sys.argv) > 7:
      usage()
    else:

        # Input arguments
        infeatures1 = sys.argv[1]     # target shapes
        fieldname1 = sys.argv[2]      # target id
        infeatures2 = sys.argv[3]     # source/grid shapes
        fieldname2 = sys.argv[4]      # source grid ID

        # Default arguments
        gridf = 0
        outnc = None

        # Conditionals for optional input arguments
        if len(sys.argv) > 5:
            if 'GRID' in sys.argv[:]:
                gridf = 1
            else:
                outnc = sys.argv[-1]
        if len(sys.argv) > 6:
            if os.path.exists(os.path.dirname(sys.argv[6])):
                outnc = sys.argv[6]

        print('Script initiated at %s' %time.ctime())

        # Clean up existing files
        projfeatures, outncFile = clean_up(infeatures1, infeatures2, outnc)
        OutDir = os.path.dirname(outncFile)

        # Project gridded polygons to coordinate system of input features
        tic1 = time.time()
        print('    Step 1: Projecting <Polygon2> to the coordinate system of <Polygon1> ...')
        projfeatures = project_to_input(infeatures1, projfeatures, infeatures2, DriverName) # Project to CRS of input feature class 1
        print('    Step 1 completed in %s seconds.' %(time.time()-tic1))

        layer1, layername1, data_source1 = Read_VectorDS(infeatures1, DriverName)
        layer2, layername2, data_source2 = Read_VectorDS(infeatures2, DriverName)
        numfeatures1  = layer1.GetFeatureCount()

        # Check for existence of provided fieldnames
        field_defn1, fieldslist1 = checkfield(layer1, fieldname1, 'infeatures1')
        field_defn2, fieldslist2 = checkfield(layer2, fieldname2, 'infeatures2')
        fieldtype1 = getfieldinfo(field_defn1, fieldname1)
        fieldtype2 = getfieldinfo(field_defn2, fieldname2)
        del field_defn1, field_defn2, fieldslist1, fieldslist2
        layer1 = layername1 = data_source1 = layer2 = layername2 = data_source2 = None

        # Begin to determine process pool
        print('    Step 2: Compute spatial weights starting.')
        print('      Found %s processors. Distributing processing across %s core(s) using %s worker processes.' %(CPU_Count, Processors, pool_size))
        tic1 = time.time()
        pool = multiprocessing.Pool(Processors)
        chunks = list(split(range(1, numfeatures1+1), pool_size))               # Split the list of basin ids into x chunks
        results = pool.map(work, [chunk for chunk in chunks], chunksize=1)      # This farms the function 'work' out to the number of processors defined by 'Processors'
        pool.close()
        pool.join()
        print('    Step 2: Compute spatial weights completed in %s seconds.' %(time.time()-tic1))

        # Step 3: Write weights to ragged array netcdf file
        print('    Step 3: Create correspondence netCDF file starting.')
        spatialweights = results[0][0].copy()                                   # Makes a copy of the first dictionary
        regridweights = results[0][1].copy()                                    # Makes a copy of the first dictionary
        other_attributes = results[0][2].copy()                                 # Makes a copy of the first dictionary
        keylist = spatialweights.keys()

        # Merge all spatial and regrid dictionaries into one large one
        print('    Beginning to grow all returned dictionaries')
        for dictionary in results[1:]:
            spatialweights.update(dictionary[0].copy())                         # Updates new dictionary with another one
            regridweights.update(dictionary[1].copy())                          # Updates new dictionary with another one
            other_attributes.update(dictionary[2].copy())                       # Updates new dictionary with another one
        print('      Size of spatialweights: %s' %len(spatialweights))
        print('      Size of regridweights: %s' %len(regridweights))
        print('      Size of other_attributes: %s' %len(other_attributes))
        print(r'      Size of min/max of dictionary lists: min=%s, max=%s' %(min([len(spatialweights[x]) for x in spatialweights]), max([len(spatialweights[x]) for x in spatialweights])))    # Check that all output dictionaries have similar sizes

        # Build regrid and spatial weights NetCDF file.
        tic1 = time.time()
        print('    Beginning to build weights netCDF file: %s . Time elapsed: %s' %(outncFile, time.time()-tic))
        create_longvector_ncfile(spatialweights, regridweights, other_attributes, outncFile, fieldtype1, fieldtype2, gridf)
        print('    Finished building weights netCDF file: %s in %s seconds' %(outncFile, time.time()-tic1))
        check_correspondence_file(outncFile)
        del fieldtype1, fieldtype2, pool, chunks, results, keylist, spatialweights, regridweights, other_attributes

        # Remove intermediate projected polygons
        if os.path.isfile(projfeatures):
            print('    Removing intermediate projected GeoPackage: %s' %projfeatures)
            os.remove(projfeatures)
        print('Process completed. Total time elapsed: %s seconds.' %(time.time()-tic))
