## Running on Windows
* Linux box uses python 3.10.12. Can't get that on windows... trying 3.10.11 so I can get the right tensorflow (2.15.0 is what's on Linux machine) version
* Created a new venv in ./.venv
* To activate, either:
  * Open git bash terminal which seems to automatically enable the venv
  * Open command prompt and run:
    ```bash
    .\.venv\Scripts\activate.bat
    echo %VIRTUAL_ENV%
    ```
* Installed the following things in the venv
  ```bash
  pip install tensorflow==2.15.0
  pip install opencv-python
  pip install matplotlib
  pip install torch
  pip install torchvision
  pip install scipy; # Needed for StanfordCars
  pip install kaggle; # Needed for manually downloading StanfordCars dataset
  pip install kagglehub; # Trying to download dataset manually without auth
  pip install PyQt5; # Debugging issues with matplotlib plots not showint
  ```