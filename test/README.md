# Dataset loading and training testing

## Requirements

To write tests, you’ll need to install pytest. As with most Python packages, 
you can install pytest in a virtual environment from PyPI using pip:
```shell
$ python -m pip install pytest
```
The pytest command will now be available in your installation environment.

## Testing

Run in comnetx/test directory:
```shell
$ pytest
```
If we need more details, run
```shell
$ pytest -v
```
### Run specific tests

File test_dataset.py contains tests for dataset loading which is executed when pytest runs with option `-m short`.
File test_training.py contains tests for dataset loading which is executed when pytest runs with option `-m long`.


## Write tests

Your test files
- should be located in directory `test` or its subdirectories;
- must have unique names of the form `test_*.py` or `*_test.py`, because pytest will search for these files and import them as top-level modules;
- must contain test items:
   * `test` prefixed test functions or methods outside of class,
   * `test` prefixed test functions or methods inside `Test` prefixed test classes (without an `__init__` method).