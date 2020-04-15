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
        song_info.append(str(get_artist_id(h5tocopy,songidx)))
        song_info.append(str(get_artist_location(h5tocopy,songidx)))
        song_info.append(get_artist_mbtags(h5tocopy,songidx).tolist())
        song_info.append(get_artist_mbtags_count(h5tocopy,songidx).tolist())
        song_info.append(str(get_artist_name(h5tocopy,songidx)))
        song_info.append(get_artist_terms(h5tocopy,songidx).tolist())
        song_info.append(get_artist_terms_freq(h5tocopy,songidx).tolist())
        song_info.append(get_artist_terms_weight(h5tocopy,songidx).tolist())
        song_info.append(float(get_danceability(h5tocopy,songidx)))
        song_info.append(float(get_duration(h5tocopy,songidx)))
        song_info.append(float(get_end_of_fade_in(h5tocopy,songidx)))
        song_info.append(float(get_energy(h5tocopy,songidx)))
        song_info.append(float(get_key(h5tocopy,songidx)))
        song_info.append(float(get_key_confidence(h5tocopy,songidx)))
        song_info.append(float(get_loudness(h5tocopy,songidx)))
        song_info.append(float(get_mode(h5tocopy,songidx)))
        song_info.append(float(get_mode_confidence(h5tocopy,songidx)))
        song_info.append(str(get_release(h5tocopy,songidx)))
        song_info.append(get_segments_confidence(h5tocopy,songidx).tolist())        
        song_info.append(get_segments_loudness_max(h5tocopy,songidx).tolist())        
        song_info.append(get_segments_loudness_max_time(h5tocopy,songidx).tolist())    
        song_info.append(get_segments_pitches(h5tocopy,songidx).tolist())    
        song_info.append(get_segments_timbre(h5tocopy,songidx).tolist())    
        song_info.append(get_similar_artists(h5tocopy,songidx).tolist())   
        song_info.append(float(get_artist_hotttnesss(h5tocopy,songidx)))
        song_info.append(str(get_song_id(h5tocopy,songidx)))
        song_info.append(float(get_start_of_fade_out(h5tocopy,songidx)))
        song_info.append(float(get_tempo(h5tocopy,songidx)))
        song_info.append(int(get_time_signature(h5tocopy,songidx)))
        song_info.append(float(get_time_signature_confidence(h5tocopy,songidx)))
        song_info.append(str(get_title(h5tocopy,songidx)))
        song_info.append(str(get_track_id(h5tocopy,songidx)))
        song_info.append(int(get_year(h5tocopy,songidx)))

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
col_name = ["artist familiarity", "artist hotttnesss", "artist id", "artist location", "artist mbtags", 
 "artist mbtags count", "artist name", "artist terms", "artist terms freq", "artist terms weight", 
 "danceability", "duration", "end of fade in", "energy", "key",
"key confidence", "loudness", "mode", "mode confidence", "release", 
 "segments confidence", "segments loudness max", "segments loudness max time", 
"segments pitches", "segments timbre", "similar artists", 
"song hotttnesss", "song id", "start of fade out", "tempo", "time signature", 
"time signature confidence", "title", "track id", "year"]

df1 = rdd1.toDF(col_name)
vectorizer.setInputCols(col_name)
vectorizer.setOutputCol("features")


lrPipeline = Pipeline()
kmeans = KMeans().setK(2).setSeed(1)
lrPipeline.setStages([vectorizer, kmeans])
model = lrPipeline.fit(df1)
