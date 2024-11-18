#!/bin/bash

while true; do
    curl "http://0.0.0.0:5000/" >> sales.txt
    sleep 60
done 
