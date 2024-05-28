from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='mlops project',
    author='Artem Ponomarenko',
    license='',
    install_requires=[
            'numpy',
            'pandas',
            'scikit-learn',
            'catboost'
        ],
    entry_points={
            'console_scripts': [
                #'make_dataset=src.data.make_dataset:load_data',
                'build_features=src.features.build_features:preprocess',
                'train_model=src.models.train_model:train_model',
                'predict_model=src.models.predict_model:make_predictions',
            ],
        },
)
