import yaml,os
with open(os.environ['YAML_PATH']) as f: io = yaml.load(f.read(), yaml.FullLoader)
print(io)