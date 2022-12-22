# Contributing Code

We welcome PRs from the community. By contributing to TSInterpret, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.

## What can I submit? 
- Resolve Issues/Bugs
- new Features/Methods
- fix typos, improve code quality, code coverage

## Development Installation
```
git clone https://github.com/fzi-forschungszentrum-informatik/TSInterpret.git
pip install -e .[dev]
```
Please add an extra branch for your development. Do not develop on the main branch.

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
