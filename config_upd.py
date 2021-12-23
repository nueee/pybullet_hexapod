import configparser
# Add the structure to the file we will create
# https://blog.finxter.com/creating-reading-updating-a-config-file-with-python/ 

edit = configparser.ConfigParser()
edit.read("./configfile.ini")
#Get the postgresql section
postgresql = edit["postgresql"]
#Update the g
postgresql["g"] = "-12.0"
#Write changes back to file
with open('./configfile.ini', 'w') as configfile:
    edit.write(configfile)
