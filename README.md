# Hot Topic

Use python satellite imagery, historical weather data, and historical fire perimeters to predict wildfire spreading using artificial neural network.

# Requirements
the following python3 libraries should be installed
- numpy
- opencv
- keras (with Tensorflow backend)
- scipy
- matplotlib (optional, for visualization)

# Data sources
- Historical weather data: [NOAA READY Archive](https://www.ready.noaa.gov/READYamet.php) for the HRRR (High Resolution Rapid Refresh) weather model. 3km, 1 hr resolution
- Historical fire perimeters: [GeoMAC Database](https://www.geomac.gov/), a conglomeration of daily fire perimeters from many different land management agencies. Perimeters are typically organized into burns, so you can follow the progress of a fire as it grows day to day.
- Historical satellite imagery:
  - landsat
  - ndvi
  - dem (digital elevation model)
