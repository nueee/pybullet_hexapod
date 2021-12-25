import configparser
config = configparser.ConfigParser()
# Add the structure to the file we will create
# https://blog.finxter.com/creating-reading-updating-a-config-file-with-python/ 
config.add_section('alpha')
config.set('alpha', 'force', '0')
config.set('alpha','joint_damping','0')

# Write the new structure to the new file

with open("../configfile.ini", 'w') as configfile:
    config.write(configfile)
    
    

