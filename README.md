# 2024 Computer Graphics (DCCS310)

Prof. Shinhoo Kang

## Development Environment

### Python + OpenGL 

#### Windows

* Install Python
  - Download stable Python 3.11.8 from [here](https://www.python.org/downloads/windows/).
  - Choose Windows installer (64-bit)
  - Select 'Add python.exe to PATH'
  - Customize installation: C:\Users\KOREA\Python\Python311
  - Optional Features: Documents, pip, tcl/tk and IDLE, Python test suite
  - Add Python to environment variables
  - Create shortcuts for installed applications
  - In a terminal, type 'set' to see Python environment variables are correctly set. PATH variable should have C:\Users\KOREA\Python\Python311 and C:\Users\KOREA\Python\Python311\Scripts

* Install virtualenv
  - ``` pip install virtualenv ```
  
* Create Virtual environment
  - Create new directory 'test_cg'
  ``` mkdir test_cg ```
  - Change the working directory to 'test_cg'
  ``` cd test_cg ``` 
  - Create virtual environment
  ``` virtualenv --python=python3.11 .venv ```

* Activate and Deactivate virtual environment
  - Go to the working directory 'test_cg'
  - ``` call .venv\Scripts\activate ``` 
  - ``` deactivate ```
  
* Install `PyOpenGL` through `pipwin`
  - ``` pip install pipwin ```
  - ``` pipwin PyOpenGL ```

* Install `GLFW`, `PyGLM`, and `Numpy`
  - ``` pip install glfw ```
  - ``` pip install PyGLM ```
  - ``` pip install numpy ```

* Test 
  ```python
  import OpenGL
  import glfw
  import glm
  import numpy
  ```

#### Mac 

* Install `PyOpenGL`, `GLFW`, `PyGLM`, and `Numpy`
  - ``` pip install PyOpenGL ```
  - ``` pip install glfw ```
  - ``` pip install PyGLM ```
  - ``` pip install numpy ```


