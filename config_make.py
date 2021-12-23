import configparser
config = configparser.ConfigParser()
# Add the structure to the file we will create
# https://blog.finxter.com/creating-reading-updating-a-config-file-with-python/ 
config.add_section('postgresql')
config.set('postgresql', 'g', '-9.8')
# Write the new structure to the new file

with open("./configfile.ini", 'w') as configfile:
    config.write(configfile)
    
    

