@echo off
cd /d %~dp0
py -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_data.py --start 2017-01-01 --end 2023-12-31
if not exist village_flood_labels.csv (
  copy village_flood_labels_TEMPLATE.csv village_flood_labels.csv
  echo Add verified village-level labels to village_flood_labels.csv, then run: python train_model.py
  pause
  exit /b 0
)
python train_model.py
if exist model\floodguard_rf.joblib python app.py
pause
