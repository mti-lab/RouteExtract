# RouteExtract


## Setup

### Requirements
For the required python packages, simply install them from the provided `requirements.txt` file with pip or uv.
```shell
pip install uv && \
uv pip install -r requirements.txt
```

For route generation, an instance of GraphHopper with map data is also required, see the section *GraphHopper*.

### GraphHopper
GraphHopper is used to calculate routes. It's open source and uses OpenStreetMap data: by hosting it yourself, it is 
completely free to use. All you need to get started is a recent graphhopper.jar file, a correspondingly recent JDK and some
OSM data.

```shell
cd GraphHopper && \
wget https://repo1.maven.org/maven2/com/graphhopper/graphhopper-web/10.2/graphhopper-web-10.2.jar
```
GraphHopper/run_instructions.txt contains information on how to get the GraphHopper server running if following the same
directory structure as outlined in that file.

### OpenStreetMap (OSM) Data
The snapshots used for this project were obtained on https://download.openstreetmap.fr/ which hosts daily snapshots of
different regions and countries. However, OSM Data can be obtained in various ways and should be compatible with GraphHopper
in any of the commonly available file formats.

Example on how to get OSM data:
```shell
mkdir GraphHopper/Kanto; \
mkdir GraphHopper/Kanto/Pedestrian; \
cd GraphHopper/Kantoo/Pedestrian && \
wget 'https://download.openstreetmap.fr/extracts/asia/japan/kanto.osm.pbf'; \
cp ../../config-example.yml config.yml;
cd ../../../
```

### Map Dataset
We provide the scripts to generate our map dataset within this directory. The dataset contains 50 hiking and sightseeing
maps from around Japan. Each map has manually created masks for every trail in a map. We also provide Ground Control Points
for georeferencing the maps.
Note that we don't host the maps ourselves but only provide links to publicly accessible maps. Please let us know if any
of the provided download links don't work anymore!

Here's how to download and prepare the dataset:
```shell
cd dataset/ && \
python download_maps.py && \
python convert_pdf_to_image.py && \
python crop_and_resize.py && \
cd ../
```

## How to run
### Trail Segmentation
To train the segmentation, run `src/train_trail_segmentation.py`. The model can be evaluated with `src/evaluate_trail_segmentation.py`.

### Route Generation
Routes can be generated using either `src/generate_gpx.py` or `src/batch_generate_gpx.py`.
Similarly, evaluation can be done using `src/evaluate_gpx.py` or `src/batch_evaluate_gpx.py`.

