import json
import os

test_json_path = '../data/test/outputs_json'
test_img_path = '../data/test/DCM'

pngs = {
    os.path.relpath(os.path.join(root, fname), start=test_img_path)
    for root, _dirs, files in os.walk(test_img_path)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}
pngs_fn_prefix = sorted([os.path.splitext(fname)[0] for fname in pngs])

if __name__ == "__main__":
    folders = os.listdir(test_img_path)
    json_file = {'boxes': []}
    for folder in folders:
        os.makedirs(os.path.join(test_json_path,folder),exist_ok=True)

    for prefix in pngs_fn_prefix:
         json_path = os.path.join(test_json_path,prefix) + '.json'
         with open(json_path,'w') as f:
            json.dump(json_file,f,indent=2)