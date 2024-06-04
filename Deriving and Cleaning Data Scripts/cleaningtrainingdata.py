# -*- coding: utf-8 -*-
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

import os
import csv
import subprocess
import re
import random
import numpy as np

with open('/content/drive/MyDrive/Spring2024/NaturalLanguageProcessing/FinalProject/rawresumes.csv') as f:
  csv_reader = csv.reader(f, delimiter=',')
  for row in csv_reader:
    gender = row[0]
    resume = row[1]
    resume = re.sub(r'[^a-zA-Z0-9\s]', ' ', resume)
    resume = resume.lower()
    resume = resume.strip()
    with open('/content/drive/MyDrive/Spring2024/NaturalLanguageProcessing/FinalProject/cleanedresumes.csv', 'a', encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile, escapechar='\\')
        writer.writerow([gender, resume])