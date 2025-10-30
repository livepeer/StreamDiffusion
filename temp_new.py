# 

import json

data = []
with open("temporalnet2_celebs_new.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

print(len(data))

# print(data[276]["negative_prompt"])
# print(data[277]["negative_prompt"])
# print(data[278]["negative_prompt"])
count = 0
for idx, d in enumerate(data):
    if type(d["negative_prompt"]) != str:
        print(d["video_id"], type(d["negative_prompt"]))
        count += 1
        if d["negative_prompt"] == None:
            d["negative_prompt"] = ""
        elif type(d["negative_prompt"]) == list:
            d["negative_prompt"] = ", ".join(d["negative_prompt"])
        data[idx] = d
print(count)

# with open("temporalnet2_celebs_new.jsonl", "w") as f:
#     for d in data:
#         f.write(json.dumps(d) + "\n")