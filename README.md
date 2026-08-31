# AI Finance Controller — Razorpay Track 04

AI-powered multi-source finance reconciliation for a synthetic 100-invoice / 115-payment batch.

## Run locally

pip install -r requirements.txt
streamlit run app.py

## Streamlit Community Cloud

1. Push this folder to GitHub.
2. Deploy `app.py` from Streamlit Community Cloud.
3. Add the Mistral API key as a secret:

MISTRAL_API_KEY = "your-key"

4. The bundled sample dataset is available under `sample_data/`.

## Notes

The ground-truth metrics are explicitly described in the app as inferred synthetic ground truth, not as an independently labeled benchmark.
