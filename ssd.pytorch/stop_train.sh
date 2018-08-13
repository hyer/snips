#!/usr/bin/env bash

ps -ef | grep train.py | cut -c 9-15 | xargs kill -9