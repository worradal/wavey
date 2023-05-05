# wavey
Wavey is a package to manipulate time varying spectroscopic data.

## Run wavey
`python wavey.py config.yml`

## To run UI

- ##### With python

    `cd src`
    `python ui.py`

- ##### With install media
    run the installer named wavey_#_#_#.exe 
    open the installed directory wavey and run the ui_#_#_#.exe to open the UI



## UI usage
1. Select a folder containing the data files
2. Select output folder
3. Select the type of data e.g. Raman (specific CSV format), UV vis (specific TXT format), IR (standard CSV)
Optional: Select a file to use as a weighting file in the frequency domain
Optional: Perfom a baseline correction
4. Click Run

![Alt text](./UI_image.PNG?raw=true "UI")

## Requirements
1. [numpy](https://numpy.org/)
2. [scipy](https://scipy.org/)
3. [pyyaml](https://pyyaml.org/)
4. [pandas](https://pandas.pydata.org/)
5. [natsort](https://github.com/SethMMorton/natsort)

