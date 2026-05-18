from setuptools import setup, find_packages

# Find all packages inside ToolBox/ (core, methods, search, visualization, ...)
# and remap them to UQ_Toolbox.* so existing import paths stay intact.
subpackages = find_packages(where="ToolBox")
packages = ["UQ_Toolbox"] + ["UQ_Toolbox." + p for p in subpackages]

# "UQ_Toolbox" -> ToolBox/; setuptools derives UQ_Toolbox.core -> ToolBox/core/, etc.
package_dir = {"UQ_Toolbox": "ToolBox"}

setup(
    name="FailCatcher",
    version="2.0.0",
    description="Post-hoc uncertainty quantification toolkit for PyTorch deep learning models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="See LICENCE",
    python_requires=">=3.8",
    packages=packages,
    package_dir=package_dir,
    install_requires=[
        "torch>=1.10",
        "torchvision>=0.11",
        "numpy>=1.20",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "shap>=0.40",
        "monai>=0.9",
    ],
)
