import os 

parent_aid_dir = "data/Million-AID/train"

cpt = sum([len(files) for r, d, files in os.walk(parent_aid_dir)])
print(f"Total Number of files : {cpt}")

print(list(os.walk(parent_aid_dir)))