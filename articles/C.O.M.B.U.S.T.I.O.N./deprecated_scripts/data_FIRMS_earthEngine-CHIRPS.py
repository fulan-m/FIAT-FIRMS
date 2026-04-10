# %%
import ee

# Initialize Earth Engine
# ee.Authenticate() # Uncomment if first time running
# ee.Authenticate() 
ee.Initialize(project='ee-mateusfulan-research')

# 1. Configuration
START_DATE = '2000-10-01'
END_DATE = '2026-03-01'
CERRADO = ee.FeatureCollection("projects/ee-mateusfulan-research/assets/fiat-firms/IBGE_biomaCerrado_1p250000")

PERIODS = {
    '5B': -96, '4B': -80, '3B': -64, '2B': -48, '1B': -32, '0B': -16
}

# -----------------------------------------------------------------------
# BIT EXTRACTION HELPERS
# -----------------------------------------------------------------------
def extract_bits(image, start_bit, end_bit, new_name):
    """Extracts a bit range from a QA band and returns it as a new band."""
    pattern = 0
    for b in range(start_bit, end_bit + 1):
        pattern += 2 ** b
    return image.select([0]).toInt().bitwiseAnd(pattern).rightShift(start_bit).rename(new_name)

def get_qa_value(image, qa_band, start_bit, end_bit, geometry, scale):
    """
    Extracts a specific bit range from a QA band and reduces to mode over
    the buffer geometry. Returns None if the image is missing.
    """
    def _extract(img):
        qa_img = img.select(qa_band)
        bit_img = extract_bits(qa_img, start_bit, end_bit, 'qa_val')
        return bit_img.reduceRegion(
            reducer=ee.Reducer.mode(), geometry=geometry, scale=scale, bestEffort=True
        ).get('qa_val')

    return ee.Algorithms.If(image, _extract(image), None)


def get_accumulated_precipitation(start_date, end_date, geometry, scale):
    """Calculate accumulated precipitation from CHIRPS between start_date and end_date."""
    chirps_collection = ee.ImageCollection("UCSB-CHC/CHIRPS/V3/DAILY_SAT") \
        .filterDate(start_date, end_date) \
        .select('precipitation')
    total_precip = chirps_collection.sum()
    return total_precip.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=scale,
        bestEffort=True
    ).get('precipitation')


def get_data_for_period(day_offset, fire_date):
    """Retrieves MODIS images for a specific time window relative to the fire."""
    period_date = fire_date.advance(day_offset, 'day')

    # NDVI/EVI (MOD13Q1) — 16-day composite, ±16 days
    ndvi = (ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate(period_date.advance(-16, 'day'), period_date.advance(16, 'day'))
            .sort('system:time_start', False).first())

    # LAI/FPAR (MCD15A3H) — 4-day composite, ±4 days
    lai = (ee.ImageCollection("MODIS/061/MCD15A3H")
           .filterDate(period_date.advance(-4, 'day'), period_date.advance(4, 'day'))
           .sort('system:time_start', False).first())

    # LST (MOD21A1D) — mean composite over ±8-day window to handle swath gaps
    temp = (ee.ImageCollection("MODIS/061/MOD21A1D")
            .filterDate(period_date.advance(-8, 'day'), period_date.advance(8, 'day'))
            .mean())

    # SW Radiation (MCD18A1) — mean composite over ±8-day window to handle sparse overpasses
    rad = (ee.ImageCollection("MODIS/062/MCD18A1")
        .filterDate(period_date.advance(-8, 'day'), period_date.advance(8, 'day'))
        .mean())

    return {'ndvi': ndvi, 'lai': lai, 'temp': temp, 'rad': rad}


def get_band_value(image, band_name, geometry, scale):
    """Safely reduces a region to a mean value for a specific band."""
    return ee.Algorithms.If(
        image,
        image.reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=scale, bestEffort=True).get(band_name),
        None
    )


# --- 1.5 Create the "Never Burned" Mask ---
total_fire_mask = ee.ImageCollection("FIRMS") \
    .filterDate(START_DATE, END_DATE) \
    .select('confidence') \
    .max() \
    .gt(0) \
    .clip(CERRADO) \
    .unmask(0)

never_burned_mask = total_fire_mask.Not().selfMask().clip(CERRADO)


def process_date(fire_date):
    target_date_str = fire_date.format('YYYY-MM-dd')
    firms = ee.ImageCollection("FIRMS") \
        .filterDate(fire_date, fire_date.advance(1, 'day')).first()

    def get_hotspots():
        hotspot_points = firms.select('confidence').gt(0).reduceToVectors(
            geometry=CERRADO, scale=1000, geometryType='centroid',
            maxPixels=1e13, bestEffort=True
        ).map(lambda f: f.set('is_fire', 1))

        num_points = hotspot_points.size()

        non_fire_points = never_burned_mask.stratifiedSample(
            numPoints=num_points,
            region=CERRADO,
            scale=1000,
            geometries=True
        ).map(lambda f: f.set('is_fire', 0))

        combined_samples = hotspot_points.merge(non_fire_points)

        dem_image = ee.ImageCollection("COPERNICUS/DEM/GLO30").mean()

        year = fire_date.get('year')
        lulc_year = ee.Number(ee.Algorithms.If(year.gt(2024), 2024, year))
        lulc_image = ee.ImageCollection("projects/mapbiomas-public/assets/brazil/lulc/v1") \
            .filter(ee.Filter.eq('year', lulc_year)).first()

        def map_over_hotspots(feature):
            point = feature.geometry()
            buffer = point.buffer(1000)

            firms_vals = firms.reduceRegion(ee.Reducer.mean(), buffer, 1000, bestEffort=True)
            elev_val = dem_image.select('DEM').reduceRegion(ee.Reducer.mean(), buffer, 30, bestEffort=True)
            lulc_val = lulc_image.reduceRegion(ee.Reducer.mode(), buffer, 30, bestEffort=True)
            is_fire_val = feature.get('is_fire')

            props = ee.Dictionary({
                'is_fire':          is_fire_val,
                'date':             target_date_str,
                'lon':              point.coordinates().get(0),
                'lat':              point.coordinates().get(1),
                'confidence':       firms_vals.get('confidence'),
                'line_number':      firms_vals.get('line_number'),
                'elevation':        elev_val.get('DEM'),
                'land_cover_class': lulc_val.get('classification')
            })

            for label, offset in PERIODS.items():
                p_data = get_data_for_period(offset, fire_date)
                ndvi_img = p_data['ndvi']
                lai_img  = p_data['lai']
                temp_img = p_data['temp']
                rad_img  = p_data['rad']

                # CHIRPS accumulated precipitation for this 16-day period
                period_end_date   = fire_date.advance(offset, 'day')
                period_start_date = period_end_date.advance(-16, 'day')
                acc_prec_value = get_accumulated_precipitation(
                    period_start_date, period_end_date, buffer, 5566
                )

                period_vals = {
                    # Spectral
                    f'NDVI_{label}':         get_band_value(ndvi_img, 'NDVI',    buffer, 250),
                    f'EVI_{label}':          get_band_value(ndvi_img, 'EVI',     buffer, 250),
                    f'LAI_{label}':          get_band_value(lai_img,  'Lai',     buffer, 500),
                    f'FPAR_{label}':         get_band_value(lai_img,  'Fpar',    buffer, 500),
                    f'LST_{label}':          get_band_value(temp_img, 'LST_1KM', buffer, 1000),
                    f'SW_radiation_{label}': get_band_value(rad_img,  'DSR',     buffer, 1000),
                    f'acc_prec_{label}':     acc_prec_value,

                    # QA bitmasks
                    # MOD13Q1 — SummaryQA bits 0-1 (0=Good, 1=Marginal, 2=Snow/ice, 3=Cloudy)
                    f'MOD13Q1_SummaryQA_0_1-{label}': get_qa_value(
                        ndvi_img, 'SummaryQA', 0, 1, buffer, 250),

                    # MCD15A3H — FparLai_QC bit 0 (0=Good/main algo, 1=Back-up/fill)
                    f'MCD15A3H_FparLai_QC_bit0-{label}': get_qa_value(
                        lai_img, 'FparLai_QC', 0, 0, buffer, 500),

                    # MCD15A3H — FparLai_QC bits 3-4 (Cloud state)
                    f'MCD15A3H_FparLai_QC_3_4-{label}': get_qa_value(
                        lai_img, 'FparLai_QC', 3, 4, buffer, 500),

                    # MOD21A1D — QC bits 0-1 (Mandatory QA: 0=Good → 3=Not produced)
                    f'MOD21A1D_QC_0_1-{label}': get_qa_value(
                        temp_img, 'QC', 0, 1, buffer, 1000),

                    # MOD21A1D — QC bits 4-5 (Cloud flag: 0=Clear → 3=Cloudy)
                    f'MOD21A1D_QC_4_5-{label}': get_qa_value(
                        temp_img, 'QC', 4, 5, buffer, 1000),

                    # MCD18A1 — DSR_Quality bits 0-1 (SR source: 0=None, 1=MCD43, 2=Climatology)
                    f'MCD18A1_DSR_Quality_0_1-{label}': get_qa_value(
                        rad_img, 'DSR_Quality', 0, 1, buffer, 500),
                }

                props = props.combine(period_vals)

            return ee.Feature(point, props)

        return combined_samples.map(map_over_hotspots)

    return ee.FeatureCollection(ee.Algorithms.If(firms, get_hotspots(), ee.FeatureCollection([])))


# 2. Main Loop (Month by Month)
start_ee = ee.Date(START_DATE)
end_ee   = ee.Date(END_DATE)
months_diff = end_ee.difference(start_ee, 'month').ceil().getInfo()

selectors = ['is_fire', 'date', 'lon', 'lat', 'confidence', 'line_number',
             'elevation', 'land_cover_class']

for p in PERIODS.keys():
    selectors.extend([
        # Spectral
        f'NDVI_{p}', f'EVI_{p}', f'LAI_{p}', f'FPAR_{p}',
        f'LST_{p}', f'SW_radiation_{p}', f'acc_prec_{p}',
        # QA bitmasks
        f'MOD13Q1_SummaryQA_0_1-{p}',
        f'MCD15A3H_FparLai_QC_bit0-{p}',
        f'MCD15A3H_FparLai_QC_3_4-{p}',
        f'MOD21A1D_QC_0_1-{p}',
        f'MOD21A1D_QC_4_5-{p}',
        f'MCD18A1_DSR_Quality_0_1-{p}',
    ])

for i in range(int(months_diff)):
    month_start = start_ee.advance(i, 'month')
    month_str   = month_start.format('YYYY-MM').getInfo()

    days_in_month = month_start.advance(1, 'month').difference(month_start, 'day')
    day_list = ee.List.sequence(0, days_in_month.subtract(1))

    def process_day(d):
        return process_date(month_start.advance(d, 'day'))

    month_results = ee.FeatureCollection(day_list.map(process_day)).flatten()

    task = ee.batch.Export.table.toDrive(
        collection=month_results,
        description=f'firms_monthly_{month_str.replace("-", "_")}',
        folder='ic-fiat_firms/FIRMS_FIATs-RF_CHIRPS',
        fileFormat='CSV',
        selectors=selectors
    )
    task.start()
    print(f"Started export for {month_str}")
