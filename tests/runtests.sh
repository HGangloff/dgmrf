#!/bin/bash
pytest *.py
if [ $? -ne 0 ]; then
   exit $?
fi

