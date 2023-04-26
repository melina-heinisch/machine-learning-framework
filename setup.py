import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='machine_learning_framework',
    version='0.0.1',
    author='Mike Huls',
    author_email='melina.heinisch@gmail.com',
    description='Machine Learning package created throught WS 22/23 Machine Learning course',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws22/Team-1/machine-learning-framework',
    license='MIT',
    packages=['machine_learning_framework'],
    install_requires=['numpy', 'pandas'],
)