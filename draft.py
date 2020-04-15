%spark.pyspark
import h5py
import tables
import os
import sys
sys.path.insert(0, "/home/hadoop/10605_Group8")
from hdf5_getters import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline

vectorizer = VectorAssembler()

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

# TODO: add more attribute
def read_h5_to_list(filename):
    import sys
    import h5py
    import tables
    sys.path.insert(0, "/home/hadoop/10605_Group8")
    from hdf5_getters import *
    h5tocopy = open_h5_file_read(filename)
    song_num = get_num_songs(h5tocopy)
    result = []
    for songidx in range(song_num):
        song_info = []
        # METADATA
        song_info.append(float(get_artist_familiarity(h5tocopy,songidx)))
        song_info.append(float(get_artist_hotttnesss(h5tocopy,songidx)))
        # song_info.append(get_artist_id(h5tocopy,songidx))
        # song_info.append(get_artist_mbid(h5tocopy,songidx))
        # song_info.append(get_artist_playmeid(h5tocopy,songidx))
        # song_info.append(get_artist_7digitalid(h5tocopy,songidx))
        # song_info.append(get_artist_latitude(h5tocopy,songidx))
        # song_info.append(get_artist_location(h5tocopy,songidx))
        # song_info.append(get_artist_longitude(h5tocopy,songidx))
        # song_info.append(get_artist_name(h5tocopy,songidx))
        # song_info.append(get_release(h5tocopy,songidx))
        # song_info.append(get_release_7digitalid(h5tocopy,songidx))
        # song_info.append(get_song_id(h5tocopy,songidx))
        # song_info.append(get_song_hotttnesss(h5tocopy,songidx))
        # song_info.append(get_title(h5tocopy,songidx))
        song_info.append(float(get_track_7digitalid(h5tocopy,songidx)))
        
        result.append(song_info)
    h5tocopy.close()
    return result
    
num_nodes = 2

# data_path = "/home/hadoop/MillionSongSubset/data/A/A/A"
data_path = '/mnt/snap/data'
# TODO: fix with nested dir
# filenames = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]
filenames = getListOfFiles(data_path)
rdd = sc.parallelize(filenames, num_nodes)
rdd1 = rdd.flatMap(lambda x: read_h5_to_list(x))
# TODO: modified with attribute name
df1 = rdd1.toDF(["a", "b", "c"])
vectorizer.setInputCols(["a", "b", "c"])
vectorizer.setOutputCol("features")


lrPipeline = Pipeline()
kmeans = KMeans().setK(2).setSeed(1)
lrPipeline.setStages([vectorizer, kmeans])
model = lrPipeline.fit(df1)
