/**
@author: Mateus H Fulan;
@date: Last updated 07/03/2026.
@description: Extracts environmental variables for each year between 'startYear' and 
              'endYear' using FIRMS as the fire layers. 'buffer_size_meters' is the 
              variable used to extract GHG emissions.
*/

// ==============================================
// Configuration Section
// ==============================================
var CONFIG = {
  buffer_size_meters: 2500,
  scales: {
    viirs_fire: 375,
    ndvi_evi: 500,
    lai: 500,
    nbr: 500,
  },
  assets: {
    cerrado: 'projects/ee-mateusfulan-research/assets/fiat-firms/IBGE_biomaCerrado_1p250000',
    mapbiomas:
      'projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_coverage_v2',
  },
  time_series: {
    periods: 11,
    days_interval: 17,
  },
};

// ==============================================
// Helper Functions
// ==============================================
/**
 * Extracts bits from an image band.
 */
function bitwiseExtract(image, fromBit, toBit) {
  if (toBit === undefined || toBit === null) {
    toBit = fromBit;
  }
  var maskSize = ee.Number(1).add(toBit).subtract(fromBit);
  var mask = ee.Number(1).leftShift(maskSize).subtract(1);
  return image.rightShift(fromBit).bitwiseAnd(mask);
}

/**
 * Extracts bitmask values from an image for a specific geometry.
 */
function extractBitmaskValue(image, bitmask, fromBit, toBit, prefix, scale, geometry) {
  if (toBit === undefined || toBit === null) {
    toBit = fromBit;
  }
  if (prefix === undefined || prefix === null) {
    prefix = '';
  }

  var mask = image.select([bitmask]);
  var maskBit = bitwiseExtract(mask, fromBit, toBit);

  var bandName = ee.String(bitmask)
    .cat('_bitmask_')
    .cat(ee.Number(fromBit).format())
    .cat('-')
    .cat(ee.Number(toBit).format());
  if (prefix) {
    bandName = ee.String(prefix).cat('_').cat(bandName);
  }

  var maskedImage = image.addBands(maskBit.rename(bandName));

  return maskedImage
    .select([bandName])
    .reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: geometry,
      scale: scale,
      maxPixels: 1e13,
    })
    .get(bandName);
}

// ==============================================
// Core Processing Functions
// ==============================================
/**
 * Processes VIIRS image to extract hotspots.
 * VIIRS confidence: 0=low, 1=nominal, 2=high.
 */
function processViirsImage(image, region, buffer_size, scale) {
  var date = ee.Date(image.get('system:time_start'));
  
  // Select confidence band
  var confidence = image.select('confidence');
  
  // Create hotspots: nominal (1) and high (2) confidence
  var hotspots = confidence.gte(1);

  var createHotspotFeature = function (feat) {
    var geometry = feat.geometry();
    var coords = geometry.coordinates();
    var lon = coords.get(0);
    var lat = coords.get(1);

    var props = image.reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: feat.geometry(),
      scale: scale,
    });

    var buffer = geometry.buffer(buffer_size);

    return ee.Feature(buffer, props)
      .set('DATE', date.format('YYYY-MM-dd'))
      .set('hotspot_time', date.millis())
      .set('LAT', lat)
      .set('LON', lon)
      .set('frp', props.get('frp'))
      .set('confidence', props.get('confidence'))
      .set('DayNight', props.get('DayNight'))
      .set('geometry_type', 'buffer');
  };

  return hotspots
    .reduceToVectors({
      geometry: region,
      scale: scale,
      geometryType: 'centroid',
      maxPixels: 1e9,
    })
    .map(createHotspotFeature);
}

/**
 * Extracts MODIS Terra FireMask bitmask values (Renamed from extractAquaFireMaskBitmasks)
 */
function extractTerraFireMaskBitmasks(image, feature, prefix) {
  var point = ee.Geometry.Point([feature.get('LON'), feature.get('LAT')]);
  var scale = CONFIG.scales.terra_fire;
  var properties = {};
  
  var fireMask_0_3 = extractBitmaskValue(image, 'FireMask', 0, 3, prefix, scale, point);
  properties[prefix + '_FireMask_0_3'] = fireMask_0_3;
  
  return properties;
}

/**
 * Extracts MODIS Terra QA bitmask values (Renamed from extractAquaQaBitmasks)
 */
function extractTerraQaBitmasks(image, feature, prefix) {
  var point = ee.Geometry.Point([feature.get('LON'), feature.get('LAT')]);
  var scale = CONFIG.scales.terra_fire;
  var properties = {};
  
  var qa_0_1 = extractBitmaskValue(image, 'QA', 0, 1, prefix, scale, point);
  properties[prefix + '_QA_0_1'] = qa_0_1;
  
  var qa_2 = extractBitmaskValue(image, 'QA', 2, 2, prefix, scale, point);
  properties[prefix + '_QA_2'] = qa_2;
  
  return properties;
}

/**
 * Adds GHG time series to feature
 */
function addGhgTimeSeries(feature, gasCollections) {
  var buffer = feature.geometry();
  var date = ee.Date(feature.get('hotspot_time'));
  var properties = {};

  var periods = 3;
  var intervalDays = 2;

  for (var gas in gasCollections) {
    var collection = gasCollections[gas];

    for (var i = 0; i < periods; i++) {
      ['b', 'a'].forEach(function(direction) {
        var offset = (i + 1) * intervalDays;
        var targetDate = date.advance(
            direction === 'b' ? -offset : offset,
            'day'
        );

        var images = collection.filterDate(targetDate, targetDate.advance(1, 'day'));
        var empty = isCollectionEmpty(images);

        var bandName = ee.Algorithms.If(
          empty,
          'none',
          ee.Image(images.first()).bandNames().get(0)
        );
        
        var value = ee.Algorithms.If(
          empty,
          -9999,
          ee.Image(images.mean()).reduceRegion({
            reducer: ee.Reducer.mean(),
            geometry: buffer,
            scale: 1000,
            maxPixels: 1e9
          }).get(bandName)
        );

        var propName = gas + '_' + i + direction + '_MEAN_2500m';
        properties[propName] = value;
      });
    }
  }

  return feature.set(properties);
}

/**
 * Adds basic properties like MapBiomas classification.
 */
function addBasicProperties(feature, mapbiomasImage) {
  var point = ee.Geometry.Point([feature.get('LON'), feature.get('LAT')]);
  var date = ee.Date(feature.get('hotspot_time'));
  var year = date.get('year');

  var mbClass = ee.Algorithms.If(
    year.gt(2024),
    'NA',
    mapbiomasImage
      .select(ee.String('classification_').cat(ee.Number(year).format('%d')))
      .reduceRegion(ee.Reducer.first(), point, 30)
      .get(ee.String('classification_').cat(ee.Number(year).format('%d')))
  );

  return feature.set({
    MaxFRP: feature.get('MaxFRP'),
    MB_CLASS: mbClass,
  });
}

/**
 * Checks if collection is empty
 */
function isCollectionEmpty(collection) {
  return ee.Algorithms.If(
    ee.Algorithms.IsEqual(collection, null),
    true,
    ee.Algorithms.If(
      ee.Number(collection.size()).gt(0),
      false,
      true
    )
  );
}

/**
 * Extracts MOD13A1 bitmask values for a feature
 */
function extractMod13a1Bitmasks(image, feature, prefix) {
  var point = ee.Geometry.Point([feature.get('LON'), feature.get('LAT')]);
  var scale = CONFIG.scales.ndvi_evi;
  var properties = {};
  
  // Extract SummaryQA bits 0-1
  var summaryQA_0_1 = extractBitmaskValue(image, 'SummaryQA', 0, 1, prefix, scale, point);
  properties[prefix + '_SummaryQA_0_1'] = summaryQA_0_1;
  
  // Extract DetailedQA bits 0-1
  var detailedQA_0_1 = extractBitmaskValue(image, 'DetailedQA', 0, 1, prefix, scale, point);
  properties[prefix + '_DetailedQA_0_1'] = detailedQA_0_1;
  
  // Extract DetailedQA bits 6-7 (Aerosol)
  var detailedQA_6_7 = extractBitmaskValue(image, 'DetailedQA', 6, 7, prefix, scale, point);
  properties[prefix + '_DetailedQA_6_7'] = detailedQA_6_7;
  
  // Extract DetailedQA bit 8 (Adjacent cloud)
  var detailedQA_8 = extractBitmaskValue(image, 'DetailedQA', 8, 8, prefix, scale, point);
  properties[prefix + '_DetailedQA_8'] = detailedQA_8;
  
  // Extract DetailedQA bits 11-13 (Land use)
  var detailedQA_11_13 = extractBitmaskValue(image, 'DetailedQA', 11, 13, prefix, scale, point);
  properties[prefix + '_DetailedQA_11_13'] = detailedQA_11_13;
  
  // Extract DetailedQA bit 15 (Shadow)
  var detailedQA_15 = extractBitmaskValue(image, 'DetailedQA', 15, 15, prefix, scale, point);
  properties[prefix + '_DetailedQA_15'] = detailedQA_15;
  
  return properties;
}

/**
 * Extracts MOD15A2H FparLai_QC bitmask values for a feature
 */
function extractMod15a2hFparLaiBitmasks(image, feature, prefix) {
  var point = ee.Geometry.Point([feature.get('LON'), feature.get('LAT')]);
  var scale = CONFIG.scales.lai;
  var properties = {};
  
  // Extract FparLai_QC bit 0 (Good quality/Other quality)
  var fparLai_0 = extractBitmaskValue(image, 'FparLai_QC', 0, 0, prefix, scale, point);
  properties[prefix + '_FparLai_QC_0'] = fparLai_0;
  
  // Extract FparLai_QC bit 2 (Detector status)
  var fparLai_2 = extractBitmaskValue(image, 'FparLai_QC', 2, 2, prefix, scale, point);
  properties[prefix + '_FparLai_QC_2'] = fparLai_2;
  
  // Extract FparLai_QC bits 3-4 (Cloud state)
  var fparLai_3_4 = extractBitmaskValue(image, 'FparLai_QC', 3, 4, prefix, scale, point);
  properties[prefix + '_FparLai_QC_3_4'] = fparLai_3_4;
  
  // Extract FparLai_QC bits 5-7 (Calculation quality)
  var fparLai_5_7 = extractBitmaskValue(image, 'FparLai_QC', 5, 7, prefix, scale, point);
  properties[prefix + '_FparLai_QC_5_7'] = fparLai_5_7;
  
  return properties;
}

/**
 * Extracts MOD15A2H FparExtra_QC bitmask values for a feature
 */
function extractMod15a2hFparExtraBitmasks(image, feature, prefix) {
  var point = ee.Geometry.Point([feature.get('LON'), feature.get('LAT')]);
  var scale = CONFIG.scales.lai;
  var properties = {};
  
  // Extract FparExtra_QC bits 0-1 (Land/water)
  var fparExtra_0_1 = extractBitmaskValue(image, 'FparExtra_QC', 0, 1, prefix, scale, point);
  properties[prefix + '_FparExtra_QC_0_1'] = fparExtra_0_1;
  
  // Extract FparExtra_QC bit 3 (Aerosol)
  var fparExtra_3 = extractBitmaskValue(image, 'FparExtra_QC', 3, 3, prefix, scale, point);
  properties[prefix + '_FparExtra_QC_3'] = fparExtra_3;
  
  return properties;
}

/**
 * Extracts MOD09A1 QA bitmask values for a feature
 */
function extractMod09a1QaBitmasks(image, feature, prefix) {
  var point = ee.Geometry.Point([feature.get('LON'), feature.get('LAT')]);
  var scale = CONFIG.scales.nbr;
  var properties = {};
  
  // Extract QA bits 0-1 (Quality)
  var qa_0_1 = extractBitmaskValue(image, 'QA', 0, 1, prefix, scale, point);
  properties[prefix + '_QA_0_1'] = qa_0_1;
  
  return properties;
}

/**
 * Extracts MOD09A1 StateQA bitmask values for a feature
 */
function extractMod09a1StateQaBitmasks(image, feature, prefix) {
  var point = ee.Geometry.Point([feature.get('LON'), feature.get('LAT')]);
  var scale = CONFIG.scales.nbr;
  var properties = {};
  
  // Extract StateQA bits 0-1 (Clouds)
  var stateQA_0_1 = extractBitmaskValue(image, 'StateQA', 0, 1, prefix, scale, point);
  properties[prefix + '_StateQA_0_1'] = stateQA_0_1;
  
  // Extract StateQA bit 2 (Cloud shadow)
  var stateQA_2 = extractBitmaskValue(image, 'StateQA', 2, 2, prefix, scale, point);
  properties[prefix + '_StateQA_2'] = stateQA_2;
  
  // Extract StateQA bits 3-5 (Land/water)
  var stateQA_3_5 = extractBitmaskValue(image, 'StateQA', 3, 5, prefix, scale, point);
  properties[prefix + '_StateQA_3_5'] = stateQA_3_5;
  
  // Extract StateQA bits 6-7 (Aerosol)
  var stateQA_6_7 = extractBitmaskValue(image, 'StateQA', 6, 7, prefix, scale, point);
  properties[prefix + '_StateQA_6_7'] = stateQA_6_7;
  
  // Extract StateQA bits 8-9 (Cirrus)
  var stateQA_8_9 = extractBitmaskValue(image, 'StateQA', 8, 9, prefix, scale, point);
  properties[prefix + '_StateQA_8_9'] = stateQA_8_9;
  
  // Extract StateQA bit 13 (Adjacent to clouds)
  var stateQA_13 = extractBitmaskValue(image, 'StateQA', 13, 13, prefix, scale, point);
  properties[prefix + '_StateQA_13'] = stateQA_13;
  
  return properties;
}

/**
 * Generic time series creation function for MOD13A1 vegetation indices
 */
function createVegetationTimeSeries(collection, bandName, prefix, feature, scale) {
  var point = ee.Geometry.Point([feature.get('LON'), feature.get('LAT')]);
  var date = ee.Date(feature.get('hotspot_time'));

  // Create empty dictionaries to start
  var properties = ee.Dictionary({});
  var dates = ee.Dictionary({});

  for (var i = 0; i < CONFIG.time_series.periods; i++) {
    ['B', 'A'].forEach(function (direction) {
      var days = ee.Number(i + 1).multiply(CONFIG.time_series.days_interval);
      var targetDate = date.advance(
        ee.Algorithms.If(direction === 'B', days.multiply(-1), days),
        'day'
      );
      
      var filteredCollection = collection.filterDate(
        targetDate.advance(-8, 'day'),
        targetDate.advance(8, 'day')
      );

      // Get the closest image
      var closestImage = filteredCollection
        .map(function(image) {
          var diff = ee.Number(image.date().difference(targetDate, 'day')).abs();
          return image.set('date_diff', diff);
        })
        .sort('date_diff')
        .first();

      // Check if collection is empty
      var empty = filteredCollection.size().eq(0);
      
      // Get the value
      var val = ee.Algorithms.If(
        empty,
        ee.Number(-9999),
        ee.Image(closestImage).reduceRegion({
          reducer: ee.Reducer.first(),
          geometry: point,
          scale: scale
        }).get(bandName)
      );
      
      // NULL CHECK: If reduceRegion returns null, replace with -9999
      var isValNull = ee.Algorithms.IsEqual(val, null);
      val = ee.Algorithms.If(isValNull, ee.Number(-9999), val);

      var propName = prefix + '_' + i + direction;
      properties = properties.set(propName, val);

      // Extract bitmask values if image exists
      var bitmaskProps = ee.Algorithms.If(
        empty,
        ee.Dictionary({}),
        extractMod13a1Bitmasks(ee.Image(closestImage), feature, propName)
      );
      
      properties = properties.combine(bitmaskProps);

      // Set the date
      var dateVal = ee.Algorithms.If(
        empty,
        ee.String('NA'),
        ee.Date(ee.Image(closestImage).get('system:time_start')).format('YYYY-MM-dd')
      );
      
      // NULL CHECK: If the date value is null, replace with 'NA'
      var isDateValNull = ee.Algorithms.IsEqual(dateVal, null);
      dateVal = ee.Algorithms.If(isDateValNull, ee.String('NA'), dateVal);
      
      dates = dates.set(propName + '_DATE', dateVal);
    });
  }

  return properties.combine(dates);
}

/**
 * Generic time series creation function for MOD15A2H (LAI/FPAR)
 */
function createLaiFparTimeSeries(collection, bandName, prefix, feature, scale) {
  var point = ee.Geometry.Point([feature.get('LON'), feature.get('LAT')]);
  var date = ee.Date(feature.get('hotspot_time'));

  // Initialize as server-side ee.Dictionary
  var properties = ee.Dictionary({});
  var dates = ee.Dictionary({});

  for (var i = 0; i < CONFIG.time_series.periods; i++) {
    ['B', 'A'].forEach(function (direction) {
      var propPrefix = prefix + '_' + i + direction;
      var days = ee.Number(i + 1).multiply(CONFIG.time_series.days_interval);
      var targetDate = date.advance(
        ee.Algorithms.If(direction === 'B', days.multiply(-1), days),
        'day'
      );
      var filteredCollection = collection.filterDate(
        targetDate.advance(-8, 'day'),
        targetDate.advance(8, 'day')
      );

      var closestImage = filteredCollection
        .map(function(image) {
          var diff = ee.Number(image.date().difference(targetDate, 'day')).abs();
          return image.set('date_diff', diff);
        })
        .sort('date_diff')
        .first();

      var empty = isCollectionEmpty(filteredCollection);
      var img = ee.Algorithms.If(empty, null, closestImage);

      var val = ee.Algorithms.If(
        empty,
        ee.Number(-9999),
        ee.Image(img).reduceRegion(ee.Reducer.first(), point, scale).get(bandName)
      );

      // If null, set it to the designated missing value (-9999).
      var isValNull = ee.Algorithms.IsEqual(val, null);
      val = ee.Algorithms.If(isValNull, ee.Number(-9999), val);
      properties = properties.set(propPrefix, val);

      var fparLaiProps = ee.Algorithms.If(
        empty,
        ee.Dictionary({}),
        extractMod15a2hFparLaiBitmasks(ee.Image(img), feature, propPrefix)
      );
      
      var fparExtraProps = ee.Algorithms.If(
        empty,
        ee.Dictionary({}),
        extractMod15a2hFparExtraBitmasks(ee.Image(img), feature, propPrefix)
      );
      
      properties = properties
        .combine(ee.Dictionary(fparLaiProps))
        .combine(ee.Dictionary(fparExtraProps));

      var dateVal = ee.Algorithms.If(
        img,
        ee.Date(ee.Image(img).get('system:time_start')).format('YYYY-MM-dd'),
        ee.String('NA') 
      );
      
      // If null, set it to the designated missing value ('NA').
      var isDateValNull = ee.Algorithms.IsEqual(dateVal, null);
      dateVal = ee.Algorithms.If(isDateValNull, ee.String('NA'), dateVal);

      dates = dates.set(propPrefix + '_DATE', dateVal);
    });
  }

  // Combine the final server-side Dictionaries
  return properties.combine(dates);
}

/**
 * Creates reflectance time series with MOD09A1 bitmask extraction
 */
function createReflectanceTimeSeries(collection, feature, scale) {
  var point = ee.Geometry.Point([feature.get('LON'), feature.get('LAT')]);
  var date = ee.Date(feature.get('hotspot_time'));

  // Create empty dictionaries to start
  var properties = ee.Dictionary({});
  var dates = ee.Dictionary({});

  for (var i = 0; i < CONFIG.time_series.periods; i++) {
    ['B', 'A'].forEach(function (direction) {
      var days = ee.Number(i + 1).multiply(CONFIG.time_series.days_interval);
      var targetDate = date.advance(
        ee.Algorithms.If(direction === 'B', days.multiply(-1), days),
        'day'
      );
      
      var filteredCollection = collection.filterDate(
        targetDate.advance(-8, 'day'),
        targetDate.advance(8, 'day')
      );

      // Get the closest image
      var closestImage = filteredCollection
        .map(function(image) {
          var diff = ee.Number(image.date().difference(targetDate, 'day')).abs();
          return image.set('date_diff', diff);
        })
        .sort('date_diff')
        .first();

      // Check if collection is empty
      var empty = filteredCollection.size().eq(0);
      
      // Get reflectance values
      var reflectanceValues = ee.Algorithms.If(
        empty,
        ee.Dictionary({}),
        ee.Image(closestImage)
          .select([
            'sur_refl_b01',
            'sur_refl_b02',
            'sur_refl_b03',
            'sur_refl_b04',
            'sur_refl_b05',
            'sur_refl_b06',
            'sur_refl_b07',
          ])
          .reduceRegion(ee.Reducer.first(), point, scale)
      );

      var bands = [
        'sur_refl_b01',
        'sur_refl_b02',
        'sur_refl_b03',
        'sur_refl_b04',
        'sur_refl_b05',
        'sur_refl_b06',
        'sur_refl_b07',
      ];

      for (var k = 0; k < bands.length; k++) {
        var band = bands[k];
        var propName = 'REFL_' + i + direction + '_' + band;
        
        var bandValue = ee.Algorithms.If(
          empty,
          ee.Number(-9999),
          ee.Dictionary(reflectanceValues).get(band)
        );
        
        // NULL CHECK: If reduceRegion returns null, replace with -9999
        var isBandValNull = ee.Algorithms.IsEqual(bandValue, null);
        bandValue = ee.Algorithms.If(isBandValNull, ee.Number(-9999), bandValue);
        
        properties = properties.set(propName, bandValue);
      }

      // Extract bitmask values if image exists
      var qaProps = ee.Algorithms.If(
        empty,
        ee.Dictionary({}),
        extractMod09a1QaBitmasks(ee.Image(closestImage), feature, 'REFL_' + i + direction)
      );
      
      var stateQaProps = ee.Algorithms.If(
        empty,
        ee.Dictionary({}),
        extractMod09a1StateQaBitmasks(ee.Image(closestImage), feature, 'REFL_' + i + direction)
      );
      
      properties = properties.combine(qaProps).combine(stateQaProps);

      // Set the date
      var dateVal = ee.Algorithms.If(
        empty,
        ee.String('NA'),
        ee.Date(ee.Image(closestImage).get('system:time_start')).format('YYYY-MM-dd')
      );
      
      // NULL CHECK: If the date value is null, replace with 'NA'
      var isDateValNull = ee.Algorithms.IsEqual(dateVal, null);
      dateVal = ee.Algorithms.If(isDateValNull, ee.String('NA'), dateVal);
      
      dates = dates.set('REFL_' + i + direction + '_DATE', dateVal);
    });
  }

  return properties.combine(dates);
}

// ==============================================
// Main Execution
// ==============================================
function processYear(startDate, endDate) {
  var cerrado = ee.FeatureCollection(CONFIG.assets.cerrado);
  var mapbiomas = ee.Image(CONFIG.assets.mapbiomas);

  // CHANGED: Using VIIRS SNPP instead of MODIS
  var viirs = ee.ImageCollection("NASA/LANCE/SNPP_VIIRS/C2")
    .filterDate(startDate, endDate)
    .map(function (image) {
      return image.clip(cerrado);
    });

  var hotspotPoints = viirs
    .map(function (img) {
      return processViirsImage(
        img,
        cerrado,
        CONFIG.buffer_size_meters,
        CONFIG.scales.viirs_fire
      );
    })
    .flatten();

  // Collections for GHG and MODIS Vegetation (NDVI, LAI, Reflectance)
  var earliestGasDate = ee.Date(startDate).advance(-7, 'day');
  var latestGasDate = ee.Date(endDate).advance(7, 'day');
  
  var gasCollections = {
    CO: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CO')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('CO_column_number_density'),
    H2O: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CO')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('H2O_column_number_density'),
    AltNuvem: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CO')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('cloud_height'),
    
    NO2: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('NO2_column_number_density'),
    NO2_tropo: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('tropospheric_NO2_column_number_density'),
    NO2_strato: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('stratospheric_NO2_column_number_density'),
    NO2_slant: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('NO2_slant_column_number_density'),
    Tropopause_Press: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('tropopause_pressure'),
    
    CH4: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('CH4_column_volume_mixing_ratio_dry_air'),
    CH4_corr_bias: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('CH4_column_volume_mixing_ratio_dry_air_bias_corrected'),
    Alt_aerosol: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('aerosol_height'),
    Thick_aerosol: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('aerosol_optical_depth'),
    
    HCHO_tropo: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_HCHO')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('tropospheric_HCHO_column_number_density'),
    HCHO_amf: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_HCHO')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('tropospheric_HCHO_column_number_density_amf'),
    HCHO_slant: ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_HCHO')
      .filterDate(earliestGasDate, latestGasDate)
      .filterBounds(cerrado)
      .select('HCHO_slant_column_number_density'),
  };

  var earliestDate = ee.Date(startDate).advance(-12 * 16, 'day');
  var latestDate = ee.Date(endDate).advance(12 * 16, 'day');
  
  // MOD13A1 with QA bands for bitmask extraction
  var modisNdviEvi = ee.ImageCollection('MODIS/061/MOD13A1')
    .filterDate(earliestDate, latestDate)
    .filterBounds(cerrado)
    .select(['NDVI', 'EVI', 'SummaryQA', 'DetailedQA']);

  // MOD15A2H with QA bands for bitmask extraction
  var modisLaiFpar = ee.ImageCollection('MODIS/061/MOD15A2H')
    .filterDate(earliestDate, latestDate)
    .filterBounds(cerrado)
    .select(['Lai_500m', 'Fpar_500m', 'FparLai_QC', 'FparExtra_QC']);

  // MOD09A1 with QA bands for bitmask extraction
  var modisReflectance = ee.ImageCollection('MODIS/061/MOD09A1')
    .filterDate(earliestDate, latestDate)
    .filterBounds(cerrado)
    .select([
      'sur_refl_b01',
      'sur_refl_b02',
      'sur_refl_b03',
      'sur_refl_b04',
      'sur_refl_b05',
      'sur_refl_b06',
      'sur_refl_b07',
      'QA',
      'StateQA',
    ]);

var processFeature = function (feature) {
    // Add basic and GHG data
    feature = addBasicProperties(feature, mapbiomas);
    feature = addGhgTimeSeries(feature, gasCollections);

    // Vegetation Indices Time Series
    var ndvi = createVegetationTimeSeries(modisNdviEvi, 'NDVI', 'NDVI', feature, CONFIG.scales.ndvi_evi);
    var evi = createVegetationTimeSeries(modisNdviEvi, 'EVI', 'EVI', feature, CONFIG.scales.ndvi_evi);
    var lai = createLaiFparTimeSeries(modisLaiFpar, 'Lai_500m', 'LAI', feature, CONFIG.scales.lai);
    var fpar = createLaiFparTimeSeries(modisLaiFpar, 'Fpar_500m', 'FPAR', feature, CONFIG.scales.lai);
    var reflectance = createReflectanceTimeSeries(modisReflectance, feature, CONFIG.scales.nbr);

    return feature.set(ndvi).set(evi).set(lai).set(fpar).set(reflectance);
  };

  var finalData = hotspotPoints.map(processFeature);
  
  var year = ee.Date(startDate).get('year').getInfo();
  var exportDescription = 'VIIRS_SNPP_' + year;
  
  Export.table.toDrive({
    collection: finalData,
    description: exportDescription,
    folder: 'ic-fiat_firms/VIIRS',
    fileFormat: 'CSV'
  });
  
  print('Export task created for year: ' + year);
}

function main() {
  var startYear = 2000; 
  var endYear = 2025;
  
  for (var year = startYear; year <= endYear; year++) {
    var startDate = ee.Date.fromYMD(year, 1, 1);
    var endDate = ee.Date.fromYMD(year + 1, 1, 1);
    
    processYear(startDate, endDate);
  }
}

main();
