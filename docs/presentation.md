# Interactive Presentation Experience

The repository includes a dedicated Streamlit presentation page at `pages/01_🚀_Interactive_Presentation.py`.

## 26-slide narrative

The deck follows the actual project lifecycle:

1. Opening
2. The story
3. Problem framing
4. Dataset
5. Data exploration
6. Data quality
7. Feature representation
8. Scaling
9. Class imbalance
10. Model arena
11. XGBoost selection
12. Live model lab
13. Decision / probability interpretation
14. Confusion matrix
15. Precision vs recall
16. Overfitting / generalization
17. What-if experiment
18. Feature importance
19. Live data explorer
20. Product architecture
21. Batch Lab workflow
22. Explainability
23. Responsible ML
24. Production roadmap
25. Takeaway
26. Closing / demo script

## Interaction design

- Animated slide entrance.
- Gradient progress bar and slide counter.
- Previous / next controls.
- Jump directly to any slide.
- Pause / resume motion.
- Live dataset sampling on the stage.
- Real XGBoost **Live Model Lab** inside the presentation.
- Controlled **What-if** experiment for model sensitivity.
- Dynamic XGBoost feature-importance chart from the fitted model.
- Dynamic confusion-matrix values from the evaluation result.
- Benchmark chart comparing all six models.
- Batch workflow explanation for CSV scoring.
- Embedded SVG system-flow and model-arena visuals.

## The most important demo moment

On slide 12, the presenter can stop the narrative and run a real prediction:

```text
Applicant inputs
      ↓
project preprocessing
      ↓
RobustScaler
      ↓
XGBoost
      ↓
class + probabilities
```

The result is produced by `predict_one()` from `src/model.py`; it is not a hard-coded animation.

## Recommended live-demo sequence

1. Open **Interactive Presentation**.
2. Start with the opening and problem framing.
3. Show the dataset and live data explorer.
4. Explain preprocessing and SMOTETomek.
5. Enter the Model Arena and select XGBoost from the benchmark.
6. Show the confusion matrix and precision/recall trade-off.
7. **Stop the deck on Live Model Lab and run an applicant live.**
8. Move to What-if and change a feature to demonstrate model sensitivity.
9. Open Feature Importance and explain model-level reliance vs causality.
10. Finish with architecture, responsible ML, and the production roadmap.

## Presentation philosophy

This is designed as an **interactive technical demo**, not a static slide dump. The presenter should alternate between narrative slides, live data, and live inference so the audience sees both the research and the working product.
