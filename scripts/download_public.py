#!/usr/bin/env python
"""Bounded, resumable HTTP range downloads for public author files."""
import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from maritime_robust.common import ROOT, digest, directory_bytes, write_json

REQUEST_PROXIES = None

def request_get(*args, **kwargs):
    if REQUEST_PROXIES is not None:
        kwargs.setdefault("proxies", REQUEST_PROXIES)
    error=None
    for attempt in range(5):
        try:
            return requests.get(*args, **kwargs)
        except requests.RequestException as e:
            error=e
            time.sleep(min(attempt,2))
    raise RuntimeError("Public HTTP endpoint unavailable after retries") from error


def drive_url(file_id):
    error=None
    for _ in range(4):
        try:
            r=request_get("https://drive.google.com/uc",params={"export":"download","id":file_id},timeout=30)
            r.raise_for_status()
            form=BeautifulSoup(r.text,"html.parser").find("form")
            break
        except requests.RequestException as e:error=e
    else:
        raise RuntimeError("Could not resolve public Drive URL") from error
    if form is None:
        raise ValueError("Expected public Drive confirmation form")
    return requests.Request("GET",form["action"],params={i["name"]:i.get("value","") for i in form.find_all("input") if i.get("name")}).prepare().url


def download(url,destination,prefix=None,threads=8,max_bytes=10*10**9,required_starts=None):
    import fcntl
    import psutil
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    part=destination.with_suffix(destination.suffix+".part").resolve()
    with destination.with_suffix(destination.suffix+".lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Also recognize a downloader launched before file locking was added.
        for process in psutil.process_iter(["pid"]):
            if process.pid==os.getpid():continue
            try:
                if any(Path(f.path)==part and f.mode in ("w","a","r+") for f in process.open_files()):
                    raise RuntimeError(f"An existing process is already downloading {destination.name}")
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                continue
        return _download(url,destination,prefix,threads,max_bytes,required_starts)


def _download(url,destination,prefix=None,threads=8,max_bytes=10*10**9,required_starts=None):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        return {"path":str(destination),"bytes":destination.stat().st_size,"sha256":digest(destination),"existing":True}
    r=request_get(url,headers={"Range":"bytes=0-0","Accept-Encoding":"identity"},stream=True,timeout=40)
    if r.status_code!=206 or not r.headers.get("Content-Range","").startswith("bytes 0-0/"):
        r.close()
        raise ValueError("Server must support verified byte ranges")
    size=int(r.headers["Content-Range"].split("/")[1]);r.close()
    part=destination.with_suffix(destination.suffix+".part")
    additional=max(0,size-part.stat().st_size) if part.exists() else size
    if size>max_bytes or directory_bytes(ROOT)+additional>49*10**9:
        raise ValueError("Download would exceed allocation")
    part=destination.with_suffix(destination.suffix+".part")
    state_path=destination.with_suffix(destination.suffix+".ranges.json")
    chunk=32*2**20
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state["size"]!=size:raise ValueError("Remote size changed")
        done=set(state["done"])
    else:
        prefix_size=0
        if prefix:
            prefix=Path(prefix)
            prefix_size=(prefix.stat().st_size//chunk)*chunk
            prefix.rename(part)
        done=set(range(0,prefix_size,chunk))
        state={"size":size,"chunk":chunk,"done":sorted(done),"started":time.time()}
        with part.open("ab") as f:f.truncate(size)
        write_json(state_path,state)
    fd=os.open(part,os.O_WRONLY)
    def fetch(start):
        end=min(start+chunk,size)-1
        error=None
        for attempt in range(4):
            try:
                with request_get(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"},stream=True,timeout=(20,60)) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get("Content-Range")!=f"bytes {start}-{end}/{size}":
                        raise ValueError("Server returned an incorrect range")
                    offset=start
                    for content in response.iter_content(2**20):
                        if offset+len(content)>end+1:raise ValueError("Oversized range")
                        written=os.pwrite(fd,content,offset)
                        if written!=len(content):raise IOError("Short file write")
                        offset+=len(content)
                    if offset!=end+1:raise IOError("Truncated range")
                return start
            except Exception as e:error=e
        raise RuntimeError(f"Range {start} failed") from error
    try:
        wanted=set(range(0,size,chunk)) if required_starts is None else set(required_starts)
        if any(start<0 or start>=size or start%chunk for start in wanted):
            raise ValueError("Invalid required range starts")
        pending=[start for start in range(0,size,chunk) if start not in done and start in wanted]
        # Fetch ZIP directory tail early, making metadata inspection possible before completion.
        if pending: pending=[pending[-1]]+pending[:-1]
        with concurrent.futures.ThreadPoolExecutor(threads) as pool:
            for future in concurrent.futures.as_completed([pool.submit(fetch,start) for start in pending]):
                done.add(future.result())
                state["done"]=sorted(done)
                write_json(state_path,state)
                print(f"{destination.name}: {len(done)}/{(size+chunk-1)//chunk} chunks",flush=True)
    finally:
        os.close(fd)
    if len(done)<(size+chunk-1)//chunk:
        return {"status":"subset_complete","path":str(part),"required_chunks":len(wanted),
                "required_completed":len(done&wanted),"whole_file_complete":False}
    part.replace(destination)
    result={"path":str(destination),"bytes":size,"sha256":digest(destination),"seconds":time.time()-state["started"]}
    write_json(destination.with_suffix(destination.suffix+".provenance.json"),result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--drive-id");parser.add_argument("--url");parser.add_argument("--output",required=True)
    parser.add_argument("--prefix");parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--proxy", help="HTTP proxy for this download only; global settings unchanged")
    parser.add_argument("--required-ranges", help="JSON array of 32MiB range starts; leaves an explicitly partial file")
    args=parser.parse_args()
    if args.proxy:
        REQUEST_PROXIES={"http":args.proxy,"https":args.proxy}
    required=json.loads(Path(args.required_ranges).read_text()) if args.required_ranges else None
    print(json.dumps(download(drive_url(args.drive_id) if args.drive_id else args.url,args.output,args.prefix,args.threads,required_starts=required),indent=2))
