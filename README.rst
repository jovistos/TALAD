**This American Life Audio Dataset Downloader**
===============


This script serves for downloading audio data used in this paper [1]. The data set consists of 663 podcasts from the This American Life radio program from 1995 to 2020, covering 637 hours of audio and an average of 18 unique speakers per conversation.

I've included multiple sources from which this script looks for files, because many of the links provided by the authors are dead. 

The data is divided in train, valid and test folders. There is an option for converting the original mp3 files to wav.

Since audio files are copyrighted, they can't be distributet.  Therefore, you can have these audios only as private dataset. Main goal of this notebook is making a private dataset on Kaggle. It can also be used for downloading data localy. Since data set sizes on Kaggle are limited to 20GB, I've included the option for spliting the data in four parts so it can fit when in wav format.

[1] Mao, H. H., Li, S., McAuley, J., & Cottrell, G. (2020). Speech Recognition and Multi-Speaker Diarization of Long Conversations. INTERSPEECH. 
https://arxiv.org/pdf/2005.08072.pdf


===============

Requirements
------------

1) Python 3.8

Installation
------------

.. code-block:: bash

    apt-get update && apt-get install -y ffmpeg
    git clone https://github.com/jovistos/TALAD
    cd TALAD
    activate your virtualenv
    pip install -r requirements.txt
    
    
Examples
--------

.. code-block:: bash

    python TAL_download_audio.py -p /home/jovis/Downloads/test -d test
