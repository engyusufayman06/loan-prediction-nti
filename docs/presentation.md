# Presentation Experience

The project now includes a dedicated Streamlit presentation page at `pages/01_Presentation.py`.

## Presentation features

- 16-slide technical narrative covering the complete ML lifecycle.
- Animated slide entrance and benchmark bars.
- Progress indicator and slide counter.
- Previous / next / home navigation.
- Jump directly to any chapter.
- **Pause Motion** control for a static presentation mode.
- **Reveal Detail** control for deeper technical notes without overcrowding the main slide.
- Live confusion-matrix values taken from the trained XGBoost evaluation.
- **Live Model Lab** embedded in the presentation: edit applicant attributes and run the actual `predict_one()` inference path.
- Responsive dark FinTech visual language designed for demos and projector-style delivery.

## Recommended demo sequence

1. Open the Presentation page.
2. Start on the opening slide and explain the objective.
3. Walk through data → preprocessing → imbalance → benchmark.
4. Pause on the scoreboard and discuss why XGBoost was selected.
5. Open the error-profile slide and explain TN / FP / FN / TP.
6. On **Live Model Lab**, change the applicant profile and click **RUN REAL MODEL**.
7. Show the product architecture and limitations.
8. Finish on the production roadmap and team slide.

## Important technical distinction

The presentation is not a fake prediction animation. The Live Model Lab calls the repository's reusable inference layer in `src/model.py`, so the displayed prediction and probabilities come from the actual trained XGBoost model used by the application.
