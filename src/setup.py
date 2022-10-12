from setuptools import setup, find_packages

# git clone https://mrcabbage972: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@github.com/mrcabbage972/ai4code.git

setup(name='AI4Code',
      version='1.0',
      packages=find_packages(), #['ai4code'],
      entry_points={
          'console_scripts': ['run_ai4code=ai4code.run_workflow:main']
      },
    package_data={
      'ai4code': ['config/*.yaml', 'config/env/*.yaml', 'config/job_logging/*.yaml', 'config/model/*.yaml', 'config/workflow/*.yaml'],
     },
      install_requires=[
          'pyyaml==6.0',
          'transformers',
          'torch',
          'rouge-score',
          'requests',
          'scikit-learn',
          'scipy',
          'pandas',
          'numpy',
          'hydra-core',
          'tensorboardx',
          'xgboost',
          'datasets',
          'wandb',
          'sentence-transformers'
      ],
     )