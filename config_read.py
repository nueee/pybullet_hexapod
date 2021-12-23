import configparser
# Add the structure to the file we will create
# https://blog.finxter.com/creating-reading-updating-a-config-file-with-python/ 

config_obj = configparser.ConfigParser()
            
config_obj.read("./configfile.ini")

print("2")
dbparam = config_obj["postgresql"]
print(dbparam["g"])
