# -*- coding: utf-8 -*-

import json
import codecs

reviews = []

with codecs.open('dataset/data.json', 'r', 'utf-8') as f:
    reviews = json.load(f)

print(reviews[0]['review'])