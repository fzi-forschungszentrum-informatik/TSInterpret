# Contributing Code

We welcome PRs from the community. By contributing to TSInterpret, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.

## What can I submit? 
- Resolve Issues/Bugs
- new Features/Methods
- fix typos, improve code quality, code coverage

## Basic Development Workflow

1. Fork the main branch from the GitHub repository.
2. Clone your fork locally.
3. Commit changes.
4. Push the changes to your fork.
5. Send a pull request from your fork back to the original main branch.

## Development Installation
```
git clone https://github.com/fzi-forschungszentrum-informatik/TSInterpret.git
pip install -e .[dev]
```
Please add an extra branch for your development. Do not develop on the main branch.


## Adding a new Algorithm

1. Pick a class from `FeatureAttribution`, `InstanceBased` or `InterpretabilityBase`, depending on the Algorithms Features. More information on the taxonomy can be found <a href="https://fzi-forschungszentrum-informatik.github.io/TSInterpret/Interpretability/#taxonomy">here</a>.
2.  Make sure you implement the required method `explain`. For Algorithms that inherit from `InterpretabilityBase`, you will also need to implement `plot`. In all other cases a default plot function is available.
3. Add to  `__init__` method.
4. If possible provide a default value for each parameter. 
5. Write a comprehensive docstring with example usage.
6. Write tests in `./tests`.
7. Ideally, add a notebook with sample usage to `./docs/Notebooks`.

## Git pre-commit hooks

Before submitting a PR, run flake8, mypy and pyupgrade hooks before every commit with `pre-commit run --all-files` . If there are errors, the commit will fail and you will see the changes that need to be made.

## Testing
We use pytest to run tests. Run all tests :

```
pytest .
```

Test files can be found / should be added in folder ./tests.


## PR Checklist 

- [ ] All functions/methods/classes/modules have docstrings and all parameters are documented.
- [ ] All functions/methods have type hints for arguments and return types.
- [ ] New functionality has tests.
- [ ] Documentation is built locally and checked for errors.
- [ ] For any new functionality or new examples, appropriate links are added.
- [ ] For any changes to existing algorithms, run the example notebooks and tests manually and check that everything still works.
- [ ] Any changes to dependencies are reflected in the appropriate place.

## LISENCE
By contributing to TSInterpret, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.
