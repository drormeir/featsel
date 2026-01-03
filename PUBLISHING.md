# Publishing featsel to PyPI

This guide explains how to build and publish the `featsel` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on both:
   - [TestPyPI](https://test.pypi.org/account/register/) (for testing)
   - [PyPI](https://pypi.org/account/register/) (for production)

2. **API Token**: Generate API tokens for both:
   - TestPyPI: https://test.pypi.org/manage/account/token/
   - PyPI: https://pypi.org/manage/account/token/

3. **Install build tools**:
   ```bash
   pip install --upgrade build twine
   ```

## Building the Package

1. **Clean previous builds**:
   ```bash
   rm -rf build/ dist/ *.egg-info/
   ```

2. **Build the package**:
   ```bash
   python -m build
   ```

   This creates:
   - `dist/featsel-<version>.tar.gz` (source distribution)
   - `dist/featsel-<version>-py3-none-any.whl` (wheel distribution)

3. **Verify the build**:
   ```bash
   twine check dist/*
   ```

## Testing on TestPyPI (Recommended First)

1. **Upload to TestPyPI**:
   ```bash
   twine upload --repository testpypi dist/*
   ```

   Enter your TestPyPI username and API token when prompted.

2. **Test installation**:
   ```bash
   # Create a fresh virtual environment
   python -m venv test_env
   source test_env/bin/activate

   # Install from TestPyPI
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ featsel

   # Test import
   python -c "from featsel import DataLoader; print('Success!')"

   # Clean up
   deactivate
   rm -rf test_env
   ```

## Publishing to PyPI

Once you've tested on TestPyPI and everything works:

1. **Upload to PyPI**:
   ```bash
   twine upload dist/*
   ```

   Enter your PyPI username and API token when prompted.

2. **Verify installation**:
   ```bash
   pip install featsel
   python -c "from featsel import DataLoader; print('Success!')"
   ```

## Using API Tokens with .pypirc

To avoid entering credentials each time, create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-testpypi-token-here
```

**Important**: Keep this file secure (it contains your API tokens).

## Version Management

Update the version in `pyproject.toml` before each release:

```toml
[project]
version = "0.1.0"  # Update this
```

Follow [Semantic Versioning](https://semver.org/):
- **0.1.0** → **0.1.1**: Bug fixes
- **0.1.0** → **0.2.0**: New features (backward compatible)
- **0.1.0** → **1.0.0**: Breaking changes

## Automated Publishing (Optional)

Consider setting up GitHub Actions for automated publishing:

1. Add PyPI API token to GitHub Secrets
2. Create `.github/workflows/publish.yml`
3. Trigger on new tags/releases

## Troubleshooting

### Package name already taken
If "featsel" is taken, choose another name in `pyproject.toml`.

### Import errors after installation
Make sure `featsel/` is properly structured as a package with `__init__.py`.

### Dependencies not installing
Verify `dependencies` list in `pyproject.toml` matches `requirements.txt`.

## Useful Commands

```bash
# Check package metadata
python -m build --sdist --wheel --outdir dist/ .
tar -tzf dist/featsel-*.tar.gz

# List what will be included
python setup.py sdist
tar -tzf dist/featsel-*.tar.gz

# Uninstall package
pip uninstall featsel
```

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Publishing Tutorial](https://packaging.python.org/tutorials/packaging-projects/)
- [Twine Documentation](https://twine.readthedocs.io/)
