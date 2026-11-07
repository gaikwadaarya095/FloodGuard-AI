# FloodGuard AI — village-level ML pipeline (research prototype)

## Important scientific limitation

The public India Flood Inventory (IFI-Impacts, 1967–2023) contains district-level flood records/impacts, not comprehensive village-by-day ground-truth labels. It is useful for historical context, but **it cannot by itself establish whether an individual village flooded on a particular day**. This pipeline downloads IFI and real historical weather, but it will not pretend district labels are village labels. A true village-level model requires `village_flood_labels.csv` with verified village-specific labels.

The model predicts: **whether a recorded flood event occurs at the same village within the next 3 calendar days**, given current/recent weather and static location features. It is a research prototype, not an official warning system.

## Files

- `download_data.py`: downloads the public India Flood Inventory CSV and Open-Meteo historical daily weather for locations in `locations_template.csv`.
- `locations_template.csv`: sample location coordinates for testing the pipeline. Replace/add real village locations for your study area; town examples are only starter coordinates.
- `village_flood_labels_TEMPLATE.csv`: format for verified village/date labels. Each labelled village-date must be represented with `flood_event` 0 or 1; do not assume unreported dates are non-flood dates.
- `train_model.py`: creates rolling rainfall features, shifts labels forward 1–3 days, does a chronological holdout, trains a Random Forest, writes metrics, model and feature metadata. It refuses to train without enough labeled examples from both classes.
- `app.py`: local API used by the HTML dashboard. Returns `model_not_trained` until a model has actually been trained.
- `run_windows.bat`: Windows helper to install dependencies, download data, and run the local API after training.

## Quick start (Windows)

1. Install Python 3.10+ from python.org and select “Add Python to PATH”.
2. Extract this ZIP to a normal folder.
3. Open Command Prompt in that folder and run:

   ```bat
   py -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   python download_data.py --start 2017-01-01 --end 2023-12-31
   ```

4. Create `village_flood_labels.csv` using the template. Only use verified village-level historical flood event sources. Include positive and genuinely assessed negative dates. The target event is in the following 1–3 days, so label coverage must include dates after each feature date too.
5. Train and evaluate:

   ```bat
   python train_model.py
   ```

6. If and only if training finishes and writes `model/floodguard_rf.joblib`, start the API:

   ```bat
   python app.py
   ```

7. Open the bundled `FloodGuard_AI_India_Map_ML.html` in Chrome/Edge. Keep the API terminal open. The page checks `http://127.0.0.1:8000/health` and displays a probability only if a trained model file is present.

## Data sources

- IFI-Impacts v4, India Flood Inventory, 1967–2023, IIT Delhi / IMD-derived inventory: https://zenodo.org/records/16994648
- Open-Meteo Historical Weather API (ERA5/ERA5-Land reanalysis; gridded estimates, not village rain gauges): https://open-meteo.com/en/docs/historical-weather-api

The IFI CSV is downloaded for reference and dataset inspection. The training labels are not auto-created from district-level IFI rows because that would produce misleading village-level labels.

## Features

Latitude, longitude, elevation, calendar month, and previous 1-, 3-, 7-, and 14-day precipitation totals. Weather data comes from Open-Meteo archive API. The current dashboard passes the selected point's coordinates, elevation and forecast API's recent precipitation to the local API. The forecast API may have missing recent hours; the dashboard will show an unavailable state rather than invent data.

## Validation

A chronological holdout keeps later dates out of training. The report contains precision, recall, F1, average precision (PR-AUC-style metric), ROC-AUC when defined, Brier score and confusion matrix. Because nearby villages and events can be spatially correlated, this is not sufficient for operational use: follow up with a spatial holdout by district/river basin and independent event seasons. Report sample counts and class balance before interpreting any metric.

## Browser / API note

The HTML calls `http://127.0.0.1:8000`. This is local to the computer running the browser. The API must be running on that same laptop; opening the HTML on a phone will not connect to a server running on the laptop unless you deliberately configure a LAN address and firewall. Internet is required for map/weather/data downloads; ML inference after training is local.

## Map access / laptop fix

- The dashboard starts with a standard street map; it does **not** request satellite or device-location permission.
- If OpenStreetMap tiles are blocked, it automatically tries CARTO, then Esri Topographic tiles. If optional satellite imagery is blocked, it switches back to the street map.
- A website cannot silently grant browser or network permissions. Map tiles normally do not require device-location permission; a tile error usually means the provider or network is blocking the request.
- For the most reliable laptop launch, double-click `Open_Dashboard_Laptop.bat`. It serves the HTML at `http://127.0.0.1:8080/` instead of using a `file://` URL. Keep its terminal window open. This dashboard server is separate from the ML API, which uses port 8000.
- Internet is still needed for live map tiles, geocoding, weather and public data APIs. If all providers are blocked by the network, try a different network/hotspot; code cannot bypass a network block.

## Flood Alert Explanation Assistant
The dashboard now includes a rule-based explanation assistant. It summarizes currently displayed forecast rainfall, elevation, historical inventory status, land-cover status, and (only when available) the local model prediction. It includes quick prompts for why an area is flagged, practical next steps, data definitions, and model reliability. This is not a generative AI service or an official alert system; it never invents an ML probability and does not replace IMD or local disaster-management instructions.

## Deploy as an online website (Render)

The Flask app now serves the dashboard at `/` and the model API on the same origin. `render.yaml` and `Procfile` are included for deployment.

1. Put this project folder in a GitHub repository (do not upload private keys or personal data).
2. In Render, choose **New + → Blueprint** and connect the repository. Render reads `render.yaml` and builds the web service.
3. After deployment, open the HTTPS URL provided by Render. The dashboard and API share the same website origin, so the browser no longer calls `127.0.0.1` on a visitor's device.
4. The deployed service will still show **NOT TRAINED** until a valid `model/floodguard_rf.joblib` and `model/metadata.json` are included in the deployed build or loaded from configured persistent storage. Do not create fake model artifacts. Train only after verified village-level labels exist.
5. If you retrain the model, redeploy the model artifact and its matching metadata together. Protect the repository if the artifacts or data should not be public.

## Browser notifications

The dashboard includes opt-in browser notifications. The user must click **Enable flood alerts** and allow notifications; a site cannot grant its own permission. While the dashboard remains open, it checks the current forecast every 15 minutes and can notify on a heuristic weather signal (forecast rainfall >= 50 mm/24h or max hourly precipitation probability >= 70%). These are configurable demonstration thresholds, **not validated flood thresholds or official alerts**. The browser may suspend background tabs, and these notifications do not work when the page is fully closed. True closed-app push alerts require a push service, HTTPS service worker, VAPID keys, a subscription store and a reliable scheduled backend worker; those are not configured in this starter project.

## Model performance and safety

No 95% accuracy target is guaranteed. Flood events are rare and false negatives are costly; evaluation must include event recall, precision, average precision, Brier score/calibration and independent temporal/spatial holdouts. A high accuracy score alone can be misleading. The current pipeline intentionally refuses to train without enough verified village-level labels from both classes. The public district-level India Flood Inventory is not silently converted into village-level ground truth.


## FloodGuard AI Assistant
The dashboard now accepts broad flood-related questions. It has a local flood knowledge fallback, so common questions work without an AI API key. To enable generative answers when hosted, configure `OPENAI_API_KEY` as a private environment secret in your hosting provider; optionally set `OPENAI_MODEL` (default `gpt-4.1-mini`). Never put API keys in the HTML or commit them to GitHub. The `/assistant` backend route uses the key server-side. If the provider is unavailable, the dashboard falls back to its local flood knowledge response.

The assistant is informational and not an official flood warning service. It does not confirm floods or make evacuation decisions.
