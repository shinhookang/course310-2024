# 2024 Computer Graphics (DCCS310)

Prof. Shinhoo Kang

## Development Environment

### Python + OpenGL 

#### Windows

* Install Python
  - Download stable Python 3.11.8 from [here](https://www.python.org/downloads/windows/).
  - Choose Windows installer (64-bit)
  - Select 'Add python.exe to PATH'
  - Customize installation: `C:\Users\user\Python\Python311`
  - Optional Features: Documents, pip, tcl/tk and IDLE, Python test suite
  - Add Python to environment variables
  - Create shortcuts for installed applications
  - In a terminal, type 'set' to see Python environment variables are correctly set. PATH variable should have `C:\Users\user\Python\Python311` and `C:\Users\user\Python\Python311\Scripts`

* Install virtualenv
  - ``` pip install virtualenv ```
  
* Create Virtual environment
  - Create new directory 'test_cg'
  ``` mkdir test_cg ```
  - Change the working directory to 'test_cg'
  ``` cd test_cg ``` 
  - Create virtual environment
  ``` virtualenv --python=python3.11 .venv ```

* Activate virtual environment
  - Go to the working directory 'test_cg'
  - ``` call .venv\Scripts\activate ``` 
  
* Install `PyOpenGL` through `pipwin`
  - ``` pip install pipwin ```
  - ``` pipwin install PyOpenGL ```

* Install `GLFW`, `PyGLM`, `pillow`, and `Numpy`
  - ``` pip install glfw ```
  - ``` pip install PyGLM ```
  - ``` pip install numpy ```
  - ``` pip install pillow ```

* Test 
  ```python
  import OpenGL
  import glfw
  import glm
  import numpy
  import PIL
  ```

* Deactivate virtual environment
  - ``` deactivate ```

* You can choose your preferred editor for working with Python.
  - Visual Studio Code 
    - from Microsoft Store 
  - Vim 
    - https://www.vim.org/download.php
    - Download gvim_9.1.0_x64_signed.exe 
    - Update PATH variable: `win+x`-> system-> advanced system settings -> environment variables

#### Mac 
* Under construction ...
* Install `PyOpenGL`, `GLFW`, `PyGLM`, and `Numpy`
  - ``` pip install PyOpenGL ```
  - ``` pip install glfw ```
  - ``` pip install PyGLM ```
  - ``` pip install numpy ```
  - ``` pip install pillow ```


## OpenGL Reference
* https://registry.khronos.org/OpenGL-Refpages/
* https://www.khronos.org/files/opengl-quick-reference-card.pdf
