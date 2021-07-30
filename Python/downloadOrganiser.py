import os
import json

filesTypes={"Images":["jpg","png","gif","jpeg"],
            "Extracts":["rar","zip"],
            "Videos":["mp4","avi","m4a"],
            "Docs":["docx","ppt","pptx","xml","csv","xlsx","doc"],
            "Pdfs":["pdf"],
            "Unknowns":[""]}
if os.path.exists("filesTypes.json"):
    with open("filesTypes.json","r") as config:
        filesTypes = json.load(config)

#Making Folders 
for Folder in filesTypes:
    if not os.path.isdir(Folder):
        os.makedirs(Folder)

#organising files
for file in os.listdir("./"):
    fileType = file.split(".")[-1]
    if file!="downloadOrganiser.py" and file not in filesTypes.keys() and file!="filesTypes.json":
        fileMove=False
        for key in filesTypes:
            if fileType in filesTypes[key] :
                os.replace(file,f"./{key}/"+file)
                fileMove = True
                break
        if not fileMove:
            os.replace(file,"./Unknowns/"+file)

with open("filesTypes.json","w") as config:
    json.dump(filesTypes,config,indent=3)
