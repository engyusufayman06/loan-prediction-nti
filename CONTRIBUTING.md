# Contributing

Thanks for contributing to **Loan Intelligence**.

## Development flow

1. Create a focused branch.
2. Keep ML transformations consistent with `src/model.py`.
3. Add or update tests for behavior changes.
4. Run the test suite locally.
5. Run a Streamlit smoke test before opening a pull request.

## Pull requests

Please include:

- what changed;
- why it changed;
- how it was tested;
- screenshots for UI changes;
- any effect on model metrics or inference schema.

## ML changes

Any change to preprocessing, class balancing, features, hyperparameters, or evaluation should document the expected effect on the benchmark. Avoid presenting a metric improvement without a held-out evaluation.

## Dataset

Do not commit private, regulated, or personally identifiable financial data. The included dataset is for the educational project context.
