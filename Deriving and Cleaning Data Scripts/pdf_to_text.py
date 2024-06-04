from pypdf import PdfReader
import csv 
import os


def sanitize_text(text):
    return text.encode('utf-8', 'ignore').decode('utf-8')

#create array of pdfs and pull text from each one
path = ".\pdfs"
filenames = os.listdir(path) 

for k in filenames:
    filename = '.\pdfs' + '\\' + k
    print("filename: " + filename + " gender: " + filename[(len(filename)-5)])
    gender = filename[(len(filename)-5)]
    reader = PdfReader(filename)
    page = reader.pages[0]
    text = page.extract_text()
    text = sanitize_text(text)



    #write text to raw resume file
    with open('raw_resumes.csv', 'a', encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile, escapechar='\\')
        writer.writerow([gender, text])

