import datetime as dt
import ee
ee.Authenticate()
ee.Initialize(project='your_project')

# Generates all dates for 03/2026
dates = [dt.date(2026, 3, day) for day in range(1, 32)]
dates_str = [d.strftime('%Y-%m-%d') for d in dates]


for date in dates_str:
    try:
        TARGET_DATE = date

        # Options: 'FIRMS', 'FIRMS-CHIRPS', 'VIIRS', 'VIIRS-CHIRPS'
        SENSOR = 'VIIRS-CHIRPS'

        # Output resolution (base on sensor)
        EXPORT_SCALE = 1000

        # Goggle drive
        DRIVE_FOLDER = 'COMBUSTION_raster'

        USE_CHIRPS = SENSOR in ('FIRMS-CHIRPS', 'VIIRS-CHIRPS')
        USE_VIIRS  = SENSOR in ('VIIRS', 'VIIRS-CHIRPS')

        CERRADO = ee.FeatureCollection(
            'projects/ee-mateusfulan-research/assets/fiat-firms/IBGE_biomaCerrado_1p250000'
        )
        CERRADO_GEOM = CERRADO.geometry()

        PERIODS = {'5B': -96, '4B': -80, '3B': -64, '2B': -48, '1B': -32, '0B': -16}

        print(f'Sensor : {SENSOR}  |  CHIRPS: {USE_CHIRPS}  |  VIIRS: {USE_VIIRS}')
        print(f'Data   : {TARGET_DATE}')
        print(f'Escala : {EXPORT_SCALE} m')

        def extract_bits(image, start_bit, end_bit, new_name):
            pattern = sum(2 ** b for b in range(start_bit, end_bit + 1))
            return image.select([0]).toInt().bitwiseAnd(pattern).rightShift(start_bit).rename(new_name)


        def get_modis_for_period(day_offset, target_date_ee):
            pd_ = target_date_ee.advance(day_offset, 'day')
            ndvi = (ee.ImageCollection('MODIS/061/MOD13Q1')
                    .filterDate(pd_.advance(-16, 'day'), pd_.advance(15, 'day'))
                    .sort('system:time_start', False).first())
            lai  = (ee.ImageCollection('MODIS/061/MCD15A3H')
                    .filterDate(pd_.advance(-4,  'day'), pd_.advance(4,  'day'))
                    .sort('system:time_start', False).first())
            temp = (ee.ImageCollection('MODIS/061/MOD21A1D')
                    .filterDate(pd_.advance(-8,  'day'), pd_.advance(8,  'day')).mean())
            rad  = (ee.ImageCollection('MODIS/062/MCD18A1')
                    .filterDate(pd_.advance(-8,  'day'), pd_.advance(8,  'day')).mean())
            return ndvi, lai, temp, rad


        def make_qa_band(image, qa_band_name, start_bit, end_bit, out_name):
            return extract_bits(image.select(qa_band_name), start_bit, end_bit, out_name).toFloat()

        target_date_ee = ee.Date(TARGET_DATE)

        year = target_date_ee.get('year')
        lulc_year  = ee.Number(ee.Algorithms.If(year.gt(2024), 2024, year))
        lulc_image = (ee.ImageCollection('projects/mapbiomas-public/assets/brazil/lulc/v1')
                    .filter(ee.Filter.eq('year', lulc_year)).first()
                    .select('classification').rename('land_cover_class').toFloat())

        dem_image  = (ee.ImageCollection('COPERNICUS/DEM/GLO30').mean()
                    .select('DEM').rename('elevation').toFloat())

        stack = lulc_image.addBands(dem_image)

        band_names = ['land_cover_class', 'elevation']

        for label, offset in PERIODS.items():
            ndvi_img, lai_img, temp_img, rad_img = get_modis_for_period(offset, target_date_ee)
            pd_ = target_date_ee.advance(offset, 'day')

            b_ndvi = ndvi_img.select('NDVI').rename(f'NDVI_{label}').toFloat()
            b_evi  = ndvi_img.select('EVI' ).rename(f'EVI_{label}' ).toFloat()
            b_lai  = lai_img .select('Lai' ).rename(f'LAI_{label}' ).toFloat()
            b_fpar = lai_img .select('Fpar').rename(f'FPAR_{label}').toFloat()
            b_lst  = temp_img.select('LST_1KM').rename(f'LST_{label}').toFloat()
            b_swr  = rad_img .select('DSR').rename(f'SW_radiation_{label}').toFloat()

            b_qa_ndvi1 = make_qa_band(ndvi_img, 'SummaryQA',  0, 1, f'MOD13Q1_SummaryQA_0_1-{label}')
            b_qa_lai0  = make_qa_band(lai_img,  'FparLai_QC', 0, 0, f'MCD15A3H_FparLai_QC_bit0-{label}')
            b_qa_lai34 = make_qa_band(lai_img,  'FparLai_QC', 3, 4, f'MCD15A3H_FparLai_QC_3_4-{label}')
            b_qa_lst01 = make_qa_band(temp_img, 'QC',         0, 1, f'MOD21A1D_QC_0_1-{label}')
            b_qa_lst45 = make_qa_band(temp_img, 'QC',         4, 5, f'MOD21A1D_QC_4_5-{label}')
            b_qa_rad01 = make_qa_band(rad_img,  'DSR_Quality',0, 1, f'MCD18A1_DSR_Quality_0_1-{label}')

            period_bands = [
                b_ndvi, b_evi, b_lai, b_fpar, b_lst, b_swr,
                b_qa_ndvi1, b_qa_lai0, b_qa_lai34,
                b_qa_lst01, b_qa_lst45, b_qa_rad01
            ]
            period_names = [
                f'NDVI_{label}', f'EVI_{label}', f'LAI_{label}', f'FPAR_{label}',
                f'LST_{label}', f'SW_radiation_{label}',
                f'MOD13Q1_SummaryQA_0_1-{label}',
                f'MCD15A3H_FparLai_QC_bit0-{label}',
                f'MCD15A3H_FparLai_QC_3_4-{label}',
                f'MOD21A1D_QC_0_1-{label}',
                f'MOD21A1D_QC_4_5-{label}',
                f'MCD18A1_DSR_Quality_0_1-{label}',
            ]

            if USE_CHIRPS:
                precip_start = pd_.advance(-16, 'day')
                precip_img   = (ee.ImageCollection('UCSB-CHC/CHIRPS/V3/DAILY_SAT')
                                .filterDate(precip_start, pd_)
                                .select('precipitation').sum()
                                .rename(f'acc_prec_{label}').toFloat())
                period_bands.append(precip_img)
                period_names.append(f'acc_prec_{label}')

            for b in period_bands:
                stack = stack.addBands(b)
            band_names += period_names

        stack = stack.clip(CERRADO_GEOM)

        print(f'Stack montado: {len(band_names)} bandas')
        print('Bandas:', band_names)

        sensor_tag  = SENSOR.replace('-', '_')
        date_tag    = TARGET_DATE.replace('-', '')
        description = f'COMBUSTION_{sensor_tag}_{date_tag}_stack'

        task = ee.batch.Export.image.toDrive(
            image=stack,
            description=description,
            folder=DRIVE_FOLDER,
            fileNamePrefix=description,
            region=CERRADO_GEOM,
            scale=EXPORT_SCALE,
            crs='EPSG:4326',
            maxPixels=1e13,
            fileFormat='GeoTIFF',
        )
        task.start()

        import json

        # This JSON is used in 'test_real_life_scenario.py' but is also available in this repo
        with open('band_names.json', 'w') as f:
            json.dump(band_names, f)

    except Exception as e:
        print('Erro:', e)
