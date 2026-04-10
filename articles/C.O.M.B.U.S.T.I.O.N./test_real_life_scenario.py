import glob
import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import ee
import datetime as dt

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODELS_BASE_PATH = './results/models/'
RASTER_INPUT_DIR = './multiband-raster/'
OUT_DIR = './test_results'
BAND_NAMES_JSON = 'band_names.json'
EE_PROJECT = 'your_project'

# Mapping for VIIRS categorical confidence
VIIRS_CONF_MAP = {
    'nominal_conf': 1,
    'above-nominal_conf': 1, # Logic: >= 1
    'high_conf': 2
}

os.makedirs(OUT_DIR, exist_ok=True)
ee.Initialize(project=EE_PROJECT)

def preprocess_features(df, variant, periods, qc_patterns, redux_cols):
    """Standardizes preprocessing for both sensors."""
    df_p = df.copy()
    for p in periods:
        # MODIS NDVI/EVI QA masking
        qa = f'MOD13Q1_SummaryQA_0_1-{p}'
        if qa in df_p.columns:
            bad = df_p[qa] != 0
            for v in ['NDVI', 'EVI']:
                if f'{v}_{p}' in df_p.columns:
                    df_p.loc[bad, f'{v}_{p}'] = np.nan
        
        # LAI/FPAR QA masking
        qa0, qa1 = f'MCD15A3H_FparLai_QC_bit0-{p}', f'MCD15A3H_FparLai_QC_3_4-{p}'
        if qa0 in df_p.columns and qa1 in df_p.columns:
            bad = (df_p[qa0] != 0) | (df_p[qa1] != 0)
            for v in ['LAI', 'FPAR']:
                if f'{v}_{p}' in df_p.columns:
                    df_p.loc[bad, f'{v}_{p}'] = np.nan

    # Drop non-feature columns
    cols_to_drop = [c for c in df_p.columns if any(pat in c for pat in qc_patterns) or c.startswith('LST_')]
    if variant == 'reduced':
        cols_to_drop.extend([c for c in redux_cols if c in df_p.columns])
    
    return df_p.drop(columns=cols_to_drop)

def get_combinations():
    """Generates the full matrix of expected model filenames."""
    combos = []
    # Process FIRMS
    for ds in ['FIRMS', 'FIRMS$CHIRPS']:
        for thr in ['80', '90', '95']:
            for m_type in ['full', 'reduced']:
                combos.append({'ds': ds.replace('$', '-'), 'thr': f"{thr}_conf", 'type': m_type, 'sensor': 'FIRMS'})
    
    # Process VIIRS
    for ds in ['VIIRS', 'VIIRS$CHIRPS']:
        for thr in ['nominal_conf', 'above$nominal_conf', 'high_conf']:
            for m_type in ['full', 'reduced']:
                combos.append({'ds': ds.replace('$', '-'), 'thr': thr.replace('$', '-'), 'type': m_type, 'sensor': 'VIIRS'})
    return combos

def run_pipeline():
    with open(BAND_NAMES_JSON) as f:
        band_names = json.load(f)
    
    combinations = get_combinations()
    cerrado = ee.FeatureCollection(f"projects/{EE_PROJECT}/assets/fiat-firms/IBGE_biomaCerrado_1p250000")

    for combo in combinations:
        model_name = f"RF_{combo['ds']}_{combo['thr']}_{combo['type']}.joblib"
        model_path = os.path.join(MODELS_BASE_PATH, model_name)
        
        if not os.path.exists(model_path):
            continue

        model = joblib.load(model_path)
        # Match rasters based on sensor name in filename
        pattern = f"COMBUSTION_{combo['ds'].replace('-', '_')}_*_stack.tif"
        rasters = glob.glob(os.path.join(RASTER_INPUT_DIR, pattern))

        for r_path in rasters:
            with rasterio.open(r_path) as src:
                meta = src.profile
                data = src.read().astype(np.float32)
                
            # Vectorize
            df = pd.DataFrame(data.reshape(data.shape[0], -1).T, columns=band_names)
            if src.nodata is not None: df.replace(src.nodata, np.nan, inplace=True)
            
            valid_mask = df.notna().any(axis=1).values
            df_proc = preprocess_features(df, combo['type'], ['5B','4B','3B','2B','1B','0B'], 
                                        ['MOD13Q1', 'MCD15A3H', 'MOD21A1D', 'MCD18A1'], ['land_cover_class', 'elevation'])

            # Alignment
            if hasattr(model, 'feature_names_in_'):
                df_proc = df_proc.reindex(columns=list(model.feature_names_in_))

            # Run Inference
            probs = np.full(data.shape[1] * data.shape[2], np.nan, dtype=np.float32)
            if valid_mask.any():
                probs[valid_mask] = model.predict_proba(df_proc[valid_mask])[:, 1]
            
            # Export Result
            meta.update(count=1, dtype='float32', nodata=-9999, compress='lzw')
            out_tif = os.path.join(OUT_DIR, f"pred_{os.path.basename(r_path)}")
            with rasterio.open(out_tif, 'w', **meta) as dst:
                dst.write(np.where(np.isnan(probs), -9999, probs).reshape(1, data.shape[1], data.shape[2]), 1)

            # --- Earth Engine Validation Data ---
            date_str = os.path.basename(r_path).split('_')[-2]
            start_date = dt.datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            end_date = (dt.datetime.strptime(date_str, '%Y%m%d') + dt.timedelta(days=1)).strftime('%Y-%m-%d')
            
            if combo['sensor'] == 'FIRMS':
                ee_coll = ee.ImageCollection("FIRMS").select('confidence').filterDate(start_date, end_date)
                thresh_val = int(combo['thr'].split('_')[0])
                fire_img = ee_coll.max().clip(cerrado)
                fire_mask = fire_img.updateMask(fire_img.gte(thresh_val))
                scale = 1000
            else:
                ee_coll = ee.ImageCollection("FIRMS").select('confidence').filterDate(start_date, end_date) # Note: VIIRS is often inside FIRMS collection in EE
                thresh_val = VIIRS_CONF_MAP.get(combo['thr'], 1)
                fire_img = ee_coll.max().clip(cerrado)
                # Handle 'above-nominal' logic
                op = fire_img.gte(thresh_val) if combo['thr'] != 'high_conf' else fire_img.eq(2)
                fire_mask = fire_img.updateMask(op)
                scale = 375

if __name__ == "__main__":
    run_pipeline()
