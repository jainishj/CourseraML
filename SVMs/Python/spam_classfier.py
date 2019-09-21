import csv
import re

def vocab_list(vocab_file):
    with open(vocab_file) as vocab:
        vocab_list = csv.reader(vocab, delimiter = '\t')
    return vocab_list
def read_email(filename):
    with open(filename) as email:
        data = email.read()
    return data

def process_email(email_file):
    vocablist = vocab_list('data/vocab.txt')
    
    print('-----Read Email-----')
    email = read_email(email_file)
    print(email)
    
    print('-----Lower case-----')
    email = email.lower()
    print(email)
    
    print('-----Stripping HTML Tags-----')
    email = re.sub(r'<[^<>]+>', ' ', email)
    print(email)
    
    print('-----Normailizing Numbers-----')
    email = re.sub(r'[0-9]+', 'number', email)
    print(email)
    
    print('-----Normalizing HTTP links -----')
    email = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email)
    print(email)
    
    print('-----Normalizing Dollars-----')
    email = re.sub(r'[$]+', 'dollar', email)
    print(email)
    
    print('-----Normailizing Email Addresses-----')
    email = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email)
    print(email)
    
    words = email.split()
    for word in words:
        word = re.sub(r'', '', word)
                      
    
    
process_email('data/emailSample1.txt')