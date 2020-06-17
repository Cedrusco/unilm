import os, uuid, base64, sys
import re 

def find_URL(string): 
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)       
    return [x[0] for x in url] 
      
def adapt_data(path): # it might be better to check the directory in app.py?
    new_path = path

    # check if it's a directory
    if(os.path.isdir(path)):
        print("is directory")
        return new_path

    # check if it's a link
    elif(len(find_URL(path)) != 0 ):
        print("islink")
        # connect to a DB using the link and "load" collection into new_path
        # new_path = collection
        return new_path
    
    # if it's not 
    else:
        return {"error": "wrong path"}
