from setuptools import setup, find_packages

setup(name='ctalearn',
      version='0.1',
      description='Deep learning for analysis and classification of image data for Imaging Atmospheric Cherenkov Telescopes, especially the Cherenkov Telescope Array (CTA).',
      url='https://github.com/ctlearn-project/ctlearn',
      license='BSD-3-Clause',
      packages=['ctalearn', 'ctalearn.data'],
      include_package_data=True,
      dependencies=[],
      dependency_links=[],
      zip_safe=False)
