# Some useful tools

from IPython.display import display
import numpy as np
import time, datetime

def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def count_file_lines(filepath):
    '''
    Count lines in a text file
    '''
    
    L = 0
    with open(filepath, "r") as f:
        while f.readline():
            L+=1
    return L;

def head_and_tail_file(filepath, N=10, has_header=True):
    '''
    Show first N lines and last N lines
    in a text file
    '''
    
    L = count_file_lines(filepath)
    H = N + 1
    if has_header:
        M = N + 2
    T = L - N - 1
                
    with open(filepath, "r") as f:
        line = f.readline()
        i = 0
        while line:
            if i < H:
                print(line)
            if i == H:
                print("[...]\n")
            if i > T:
                print(line)
            line = f.readline()
            i += 1

    print("TOTAL lines:",L,'(',i,')')

def date_converter(fecha):
    # convertir de dd.mm.yyyy a yyyy-mm-dd
    dia = fecha[0:2]
    mes = fecha[3:5]
    anio = fecha[6:10]

    if int(dia) < 1 or int(dia) > 31:
        print("RARO DIA:",fecha)
    if int(mes) < 1 or int(mes) > 12:
        print("RARO MES:",fecha)

    return anio+'-'+mes+'-'+dia

def compare_lists(A,A_name,B,B_name,verbose=True):
    only_in_a = 0
    for a in A:
        if a not in B:
            if verbose:
                print(a,"-> only in "+A_name)
            only_in_a+=1
    only_in_b = 0
    for b in B:
        if b not in A:
            if verbose:
                print(b,"-> only in"+B_name)
            only_in_b+=1
            
    print(A_name,len(A),'items,',only_in_a,'only in it')
    print(B_name,len(B),'items,',only_in_b,'only in it')

def remove_from_list(A,B):
    for b in B:
        A.remove(b)
    return A
    