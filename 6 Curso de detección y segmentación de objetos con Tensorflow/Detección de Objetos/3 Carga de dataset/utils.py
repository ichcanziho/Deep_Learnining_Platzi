import configparser


def read_ini():
    config = configparser.ConfigParser()
    config.read("config.ini")
    out = {}
    for section in config.sections():
        for key in config[section]:
            out[key] = config[section][key]
    return out
