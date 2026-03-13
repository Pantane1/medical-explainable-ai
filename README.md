# MedXAI — Explainable AI for Medicine

A production-grade Explainable AI (XAI) system for clinical decision support.
Provides transparent, interpretable predictions with SHAP, LIME, uncertainty
quantification, guideline validation, and regulatory audit trails.

## Quick Start

```bash
# 1. Clone and set up
git clone <repo-url>
cd medical-explainable-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Train a model
python src/models/train_model.py

# 6. Run API server
python api/app.py

# 7. Open dashboard
open frontend/templates/dashboard.html
```

## Architecture

```
Patient Input → Preprocessor → Model → Prediction
                                  ↓
                           SHAP Explainer
                           LIME Explainer
                           Guideline Validator
                           Uncertainty Estimator
                                  ↓
                         Clinical Report + Audit Log
```

## Project Structure

See `docs/architecture.md` for full structure documentation.

## Regulatory

This system is intended as a **clinical decision support tool** only.
It does not replace clinical judgment. All predictions must be reviewed
by a qualified healthcare professional.

- Audit logging: enabled by default
- HIPAA: no PHI stored in logs (patient IDs only)
- Fairness monitoring: demographic parity + equalized odds checks
- Model versioning: all models tracked in `models/`

## License

Research use only. See ![LICENSE] for details.
