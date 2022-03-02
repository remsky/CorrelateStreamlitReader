Create a conda environment 3.7

> conda create -n streamlitmvp python=3.7

Install requirements via Anaconda Prompt: (make sure your terminal is in the unzipped StreamlitExplorerMVP folder first)

> pip install -r requirements.txt

Then run the app via 

> conda activate streamlitmvp
> streamlit run "C:\Users\.[your path to the file here]..\StreamlitExplorerMVP\app.py"

If it complains about any dependencies missed in the requirements.txt file, just install with pip. 
Should launch in browser. 

*****

Currently hardcoded to read from the .pkl/.csv, and glitchy. 