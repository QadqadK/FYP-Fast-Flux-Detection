import re, math
from collections import Counter
from statistics import mean, stdev

def shannon_entropy(items):
    counts = Counter(items)
    total = sum(counts.values())
    return -sum((c/total)*math.log2(c/total) for c in counts.values()) if total else 0

def extract_features(dig_text):
    a_records = re.findall(r'IN\s+A\s+(\d+\.\d+\.\d+\.\d+)', dig_text)
    ttl_vals = list(map(int, re.findall(r'(\d+)\s+IN\s+A\s+', dig_text)))
    cname_records = re.findall(r'IN\s+CNAME\s+(\S+)', dig_text)
    ns_records = re.findall(r'IN\s+NS\s+(\S+)', dig_text)

    subnets = {'.'.join(ip.split('.')[:3]) for ip in a_records}

    return [
        len(a_records),
        min(ttl_vals) if ttl_vals else 0,
        max(ttl_vals) if ttl_vals else 0,
        mean(ttl_vals) if ttl_vals else 0,
        stdev(ttl_vals) if len(ttl_vals) > 1 else 0,
        len(cname_records),
        len(ns_records),
        shannon_entropy(a_records),
        len(subnets)
    ]
