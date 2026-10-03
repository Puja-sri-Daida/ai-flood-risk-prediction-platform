# AI Disaster Intelligence & Early Warning Platform

A 100% software-only flood-risk decision-support prototype for a B.Tech workshop/demo.

## What it does
- Predicts flood probability from environmental/geographical inputs
- Classifies LOW / MODERATE / HIGH / CRITICAL risk
- Shows an interactive GIS map
- Explains major contributing factors
- Shows model performance and historical demo analytics
- Runs locally with no external API dependency

## Important
The bundled dataset is a **reproducible synthetic demo dataset**, designed to make the workshop project self-contained and reliable. It must not be presented as a validated operational flood-forecasting system. For real deployment, replace it with validated historical observations from authoritative sources.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Suggested workshop demo
1. Choose a city.
2. Set rainfall to a high value.
3. Increase soil moisture and lower elevation.
4. Show the probability/risk change.
5. Explain the risk factors.
6. Show the interactive risk map.
7. Open Model Performance and explain precision/recall/ROC-AUC.
