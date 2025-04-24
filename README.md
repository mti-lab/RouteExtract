# RouteExtract_kremser
Extracting routes from paper maps so they can be used in navigation software


## Setup

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

### OSM Data
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
```

