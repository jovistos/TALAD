
import subprocess
import glob
import os
import time
import codecs
import re
import json
import random
import argparse
import math

from pathlib import Path
from random import randrange

from tqdm import tqdm
import numpy as np
import pandas as pd

import pydub
from pydub.playback import play
from pydub import AudioSegment
from pydub.utils import mediainfo

from urllib.parse import quote
import urllib.request
import urllib.error
import urllib.parse

from bs4 import BeautifulSoup


class TALDownAudio:
    transcript_dir_path = '/kaggle/input/this-american-life-podcast-transcriptsalignments/'

    def __init__(self, data_parts, DOWNLOAD_DIR='./'):

        self.data_parts = data_parts
        self.download_dir_path = Path(DOWNLOAD_DIR)
        self.source_dir = Path('source')
        
    def _list_episodes(self, mode):
        self.working_dir +"/train_test_valid_split/"+mode+".txt"
        with open(self.working_dir +"/train_test_valid_split/"+mode+".txt") as f:
            episode_list = f.readlines()
        episode_list = [x.strip() for x in episode_list] 
        return episode_list

    def _family_of_links_2(self, ep_number):

        URL = 'http://audio.thisamericanlife.org/jomamashouse/ismymamashouse/'

        file_name = ep_number+'.mp3'
        link = URL+file_name

        return link

    def _family_of_links_6(self, ep_number):

        SOURCE_FILE_LINK = 'https://raw.githubusercontent.com/dcadata/this-american-life-archive/master/TALArchive.csv'

        source_dir = Path('self.working_dir')/Path(self.source_dir)
        source_file_path = source_dir/Path('TALArchive.csv')

        if not source_file_path.is_file():
            subprocess.run(['wget', SOURCE_FILE_LINK, '-P', source_dir])

        self.df_links = pd.read_csv(str(source_file_path))
        file_name = ep_number+'.mp3'
        link = self.df_links.loc[self.df_links['number']
                                 == int(ep_number)]['url'].values[0]

        return link

    def _family_of_links_4(self, ep_number):

        URL = 'https://podcast.thisamericanlife.org/podcast/'

        file_name = ep_number+'.mp3'
        link = URL+file_name

        return link

    def _family_of_links_5(self, ep_number):

        URL = 'https://stream.thisamericanlife.org/'

        file_name = ep_number+'.mp3'
        link = URL+ep_number+'/'+file_name

        return link

    def _family_of_links_1(self, ep_number):

        URL = 'https://stream.thisamericanlife.org/'

        g = codecs.open(self.transcript_dir_path +
                        'download_page_snapshot.html', "r", "utf-8")
        f = g.read()
        start = f.find(URL+ep_number)
        end = f.find(ep_number+".mp3")
        link = f[start:end]+ep_number+".mp3"
        g.close()

        return link

    def _family_of_links_3(self, ep_number):

        URL = 'https://www.reddit.com/r/ThisAmericanLife/wiki/download'

        file_name_mp3 = ep_number+'.mp3'
        source_dir = Path(self.working_dir)/Path(self.source_dir)
        file_path_txt = source_dir/Path('links.txt')

        if not Path(file_path_txt).is_file():
            response = urllib.request.urlopen(URL)
            webContent = response.read()
            soup = BeautifulSoup(webContent)
            links = [x.get('href') for x in soup.findAll('a')]
            with open(file_path_txt, 'w') as f:
                for item in links:
                    f.write("%s\n" % item)

        with open(file_path_txt, 'r') as read_obj:
            for line in read_obj:
                if file_name_mp3 in line:
                    return line[:-1]

        return "False"

    @staticmethod
    def _try_link(link):

        try:
            download = subprocess.run(['wget','-N', link])
            returncode = int(download.returncode)
        except:
            returncode = 1
        return returncode

    def _download_mp3(self, episode_list, destination_path, mode):

        print(f'..downloading {mode} dataset...\n')
        time.sleep(0.5)
        Path(self.source_dir).mkdir(parents=True, exist_ok=True)
        Path(destination_path).mkdir(parents=True, exist_ok=True)
        os.chdir(destination_path)
        
        for ep_number in tqdm(episode_list):

            file_name = ep_number+'.mp3'
            
            if not Path(file_name).is_file():
                returncode = self._try_link(self._family_of_links_2(ep_number))
                if not returncode == 0:
                    returncode = self._try_link(
                        self._family_of_links_3(ep_number))
                    if not returncode == 0:
                        returncode = self._try_link(
                            self._family_of_links_4(ep_number))
                        if not returncode == 0:
                            returncode = self._try_link(
                                self._family_of_links_5(ep_number))
                            if not returncode == 0:
                                returncode = self._try_link(
                                    self._family_of_links_5(ep_number))
                                if not returncode == 0:
                                    returncode = self._try_link(
                                        self._family_of_links_6(ep_number))

        os.chdir(self.download_dir_path)

    def _convert_mp3_to_wav(self, folder_path):

        os.chdir(folder_path)
        list_of_mp3_names = glob.glob("*.mp3")
        print(f'..converting {self.mode} dataset from mp3 to wav...')
        time.sleep(0.5)

        for file_name in tqdm(list_of_mp3_names):
            if not Path(file_name.split('.')[0]+'.wav').is_file():
                sound = AudioSegment.from_mp3(file_name)
                sound.export(file_name.split('.')[
                             0]+'.wav', format="wav", parameters=['-ar', self.ar, '-ac', self.ac, '-ab', self.ab])
                os.remove(file_name)

        os.chdir(self.download_dir_path)

    def _missing_data_percentage(self, mode):
        missing_ep = set(
            self.list_of_ep)-set([x.stem for x in (self.download_dir_path/mode).glob('*')])
        percentage = len(missing_ep)/len(self.list_of_ep)

        print(f"\n\n{int(percentage*100)}% of the {mode} data is missing")
        if percentage != 0:
            print(f"missing episodes {missing_ep}")
        print("\n*****************************************************************************\n")

    def _download_part(self, mode):

        if 'part' in mode:
            self.mode = 'train'
        else:
            self.mode = mode


        dir_path = self.download_dir_path/self.mode
        self.list_of_ep = self._list_episodes(self.mode)

        if 'part' in mode:
            part_numb = int(mode.split('_')[2])
            one_forth = int(math.ceil(len(self.list_of_ep)/4))
            self.list_of_ep = self.list_of_ep[(
                part_numb-1)*one_forth:part_numb*one_forth]

        self._download_mp3(self.list_of_ep, dir_path, mode)
        if self.convert_to_wav:
            self._convert_mp3_to_wav(dir_path)
        self._missing_data_percentage(self.mode)

    def download(self, convert_to_wav=False, ar='16000',ac='1',ab='16'):

        print("...starting...\n")
        self.ar = ar
        self.ac = ac
        self.ab = ab
        self.convert_to_wav = convert_to_wav
        self.working_dir =os.getcwd()
        os.chdir(self.download_dir_path)
        for part in self.data_parts:
            self._download_part(part)

    @staticmethod
    def rm_tree(pth):
        pth = Path(pth)
        for child in pth.glob('*'):
            if child.is_file():
                child.unlink()
            else:
                self.rm_tree(child)
        pth.rmdir()

    def remove_dir(self):

        os.chdir(self.download_dir_path)
        for part in self.data_parts:
            pth = self.download_dir_path/part
            for child in pth.glob('*'):
                child.unlink()

            pth.rmdir()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--download_path', required=True, help="apsolute folder path for dataset to be stored in")
    parser.add_argument('-d', '--data_parts', required=True, nargs='+', help="which parts of data to download. valid, test, train, train_part_{i}, i=1,2,3,4")
    parser.add_argument('-w', '--convert_to_wav', default=False, help="convert mp3 files to wav")
    parser.add_argument('-ar', '--sample_rate', default='16000', help="wav sample rate")
    parser.add_argument('-ac', '--channels', default='1', help="wav number of channels")
    parser.add_argument('-ab', '--bit_depth', default='16', help="wav bit_depth")

    args = parser.parse_args()
    if args.convert_to_wav=="True":
        TALDownAudio(args.data_parts,args.download_path).download(convert_to_wav=True, ar = args.sample_rate, ac = args.channels,ab = args.bit_depth )
    else:
        TALDownAudio(args.data_parts,args.download_path).download(convert_to_wav=False)
       
 




